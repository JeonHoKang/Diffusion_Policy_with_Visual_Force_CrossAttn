from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import numpy as np
import math
import torch
import torch.nn as nn
import torchvision
import collections
import zarr
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import gdown
import os
from skvideo.io import vwrite
from env_util import PushTImageEnv
from network import ConditionalUnet1D, DiffusionPolicy
from data_util import data_utils
import cv2
from train_utils import train_utils

#@markdown ### **Loading Pretrained Checkpoint**
#@markdown Set `load_pretrained = True` to load pretrained weights.

#@markdown ### **Network Demo**


class EvaluatePushT:
    # construct ResNet18 encoder
    # if you have multiple camera views, use seperate encoder weights for each view.
    def __init__(self, max_steps):

        diffusion = DiffusionPolicy()
        # num_epochs = 100
        ema_nets = self.load_pretrained(diffusion)
        # ResNet18 has output dim of 512
        vision_feature_dim = 512
        # agent_pos is 2 dimensional
        # lowdim_obs_dim = 2
        # # observation feature has 514 dims in total per step
        # obs_dim = vision_feature_dim + lowdim_obs_dim
        # action_dim = 2
        #@markdown ### **Inference**

        # limit enviornment interaction to 200 steps before termination
        env = PushTImageEnv()
        # use a seed >200 to avoid initial states seen in the training dataset
        env.seed(100000)

        # get first observation
        obs, info = env.reset()
        
     # keep a queue of last 2 steps of observations
        obs_deque = collections.deque(
            [obs] * diffusion.obs_horizon, maxlen=diffusion.obs_horizon)
        # save visualization and rewards
        imgs = [env.render(mode='rgb_array')]
        
        rewards = list()
        step_idx = 0
        # device transfer
        device = torch.device('cuda')
        _ = diffusion.nets.to(device)

        
        self.diffusion = diffusion
        self.vision_feature_dim = vision_feature_dim
        self.env = env
        self.obs = obs
        self.info = info
        self.rewards = rewards
        self.device = device
        self.obs_deque = obs_deque
        self.imgs = imgs
        self.max_steps = max_steps
        self.ema_nets = ema_nets
        self.step_idx = step_idx

        # the final arch has 2 parts
    ###### Load Pretrained 
    def load_pretrained(self, diffusion):

        load_pretrained = True
        if load_pretrained:
            # ckpt_path = "/home/jeon/jeon_ws/diffusion_policy/src/diffusion_cam/checkpoints/checkpoint_100.pth"
            ckpt_path = "pusht_vision_100ep.ckpt"
            if not os.path.isfile(ckpt_path):
                id = "1XKpfNSlwYMGaF5CncoFaLKCDTWoLAHf1&confirm=t"
                gdown.download(id=id, output=ckpt_path, quiet=False)

            state_dict = torch.load(ckpt_path, map_location='cuda')
            #   noise_pred_net.load_state_dict(checkpoint['model_state_dict'])
            #   start_epoch = checkpoint['epoch'] + 1
            #   optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            #   lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            #   start_epoch = checkpoint['epoch'] + 1
            ema_nets = diffusion.nets
            ema_nets.load_state_dict(state_dict)
            print('Pretrained weights loaded.')
        else:
            print("Skipped pretrained weight loading.")
        return ema_nets
    
    def inference(self):
        diffusion = self.diffusion
        max_steps = self.max_steps
        device = self.device
        obs_deque = self.obs_deque
        imgs = self.imgs
        ema_nets = self.ema_nets
        env = self.env
        rewards = self.rewards
        step_idx = self.step_idx
        done = False
        with tqdm(total=max_steps, desc="Eval PushTImageEnv") as pbar:
            while not done:
                B = 1
                # stack the last obs_horizon number of observations
                images = np.stack([x['image'] for x in obs_deque])
                agent_poses = np.stack([x['agent_pos'] for x in obs_deque])

                # normalize observation
                nagent_poses = data_utils.normalize_data(agent_poses, stats=diffusion.stats['agent_pos'])
                # images are already normalized to [0,1]
                nimages = images

                # device transfer
                nimages = torch.from_numpy(nimages).to(device, dtype=torch.float32)
                # (2,3,96,96)
                nagent_poses = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32)
                # (2,2)

                # infer action
                with torch.no_grad():
                    # get image features
                    image_features = ema_nets['vision_encoder'](nimages)
                    # (2,512)

                    # concat with low-dim observations
                    obs_features = torch.cat([image_features, nagent_poses], dim=-1)

                    # reshape observation to (B,obs_horizon*obs_dim)
                    obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

                    # initialize action from Guassian noise
                    noisy_action = torch.randn(
                        (B, diffusion.pred_horizon, diffusion.action_dim), device=device)
                    naction = noisy_action

                    # init scheduler
                    diffusion.noise_scheduler.set_timesteps(diffusion.num_diffusion_iters)

                    for k in diffusion.noise_scheduler.timesteps:
                        # predict noise
                        noise_pred = ema_nets['noise_pred_net'](
                            sample=naction,
                            timestep=k,
                            global_cond=obs_cond
                        )

                        # inverse diffusion step (remove noise)
                        naction = diffusion.noise_scheduler.step(
                            model_output=noise_pred,
                            timestep=k,
                            sample=naction
                        ).prev_sample

                # unnormalize action
                naction = naction.detach().to('cpu').numpy()
                # (B, pred_horizon, action_dim)
                naction = naction[0]
                action_pred = data_utils.unnormalize_data(naction, stats=diffusion.stats['action'])

                # only take action_horizon number of actions5
                start = diffusion.obs_horizon - 1
                end = start + diffusion.action_horizon
                action = action_pred[start:end,:]
            # (action_horizon, action_dim)
    
                # execute action_horizon number of steps
                # without replanning
                for i in range(len(action)):
                    # stepping env
                    obs, reward, done, _, info = env.step(action[i])
                    # save observations
                    obs_deque.append(obs)
                    # and reward/vis
                    rewards.append(reward)
                    imgs.append(env.render(mode='rgb_array'))

                    # update progress bar
                    step_idx += 1
                    pbar.update(1)
                    pbar.set_postfix(reward=reward)
                    if step_idx > max_steps:
                        done = True
                    if done:
                        break
        # print out the maximum target coverage

        print('Score: ', max(rewards))
        return imgs


def main():
    max_steps = 300

    eval_pusht = EvaluatePushT(max_steps)
    imgs = eval_pusht.inference()
    height, width, layers = imgs[0].shape
    video = cv2.VideoWriter('/home/jeon/jeon_ws/diffusion_policy/src/diffusion_cam/checkpoints/vis_PUSHT.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (width, height))

    for img in imgs:
        video.write(np.uint8(img))

    video.release()
if __name__ == "__main__":
    main()