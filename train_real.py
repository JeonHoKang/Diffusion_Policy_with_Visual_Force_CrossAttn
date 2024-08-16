from network import DiffusionPolicy
from real_robot_network import DiffusionPolicy_Real
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import os
import time

def train_Real_Robot():
    # # for this demo, we use DDPMScheduler with 100 diffusion iterations
    diffusion = DiffusionPolicy_Real()
    # device transfer
    device = torch.device('cuda')
    _ = diffusion.nets.to(device)

    #@markdown ### **Training**
    #@markdown
    #@markdown Takes about 2.5 hours. If you don't want to wait, skip to the next cell
    #@markdown to load pre-trained weights

    num_epochs = 200

    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(
        parameters=diffusion.nets.parameters(),
        power=0.75)
    checkpoint_dir = "/home/jeon/jeon_ws/diffusion_policy/src/diffusion_cam/checkpoints"
    # Standard ADAM optimizer
    # Note that EMA parametesr are not optimized
    optimizer = torch.optim.AdamW(
        params=diffusion.nets.parameters(),
        lr=1e-4, weight_decay=1e-6)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(diffusion.dataloader) * num_epochs
    )

    with tqdm(range(num_epochs), desc='Epoch') as tglobal:
        # epoch loop
        for epoch_idx in tglobal:
            epoch_loss = list()
            # batch loop
            with tqdm(diffusion.dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    # data normalized in dataset
                    # device transfer
                    nimage = nbatch['image'][:,:diffusion.obs_horizon].to(device)
                    nimage_second_view = nbatch['image2'][:,:diffusion.obs_horizon].to(device)
                    nagent_pos = nbatch['agent_pos'][:,:diffusion.obs_horizon].to(device)
                    naction = nbatch['action'].to(device)
                    B = nagent_pos.shape[0]

                    # encoder vision features
                    image_features = diffusion.nets['vision_encoder'](
                        nimage.flatten(end_dim=1))
                    image_features = image_features.reshape(
                        *nimage.shape[:2],-1)
                    # (B,obs_horizon,D)

                    # encoder vision features
                    image_features_second_view = diffusion.nets['vision_encoder2'](
                        nimage_second_view.flatten(end_dim=1))
                    image_features_second_view = image_features_second_view.reshape(
                        *nimage_second_view.shape[:2],-1)
                    # (B,obs_horizon,D)

                    # concatenate vision feature and low-dim obs
                    obs_features = torch.cat([image_features, image_features_second_view, nagent_pos], dim=-1)
                    obs_cond = obs_features.flatten(start_dim=1)
                    # (B, obs_horizon * obs_dim)

                    # sample noise to add to actions
                    noise = torch.randn(naction.shape, device=device)

                    # sample a diffusion iteration for each data point
                    timesteps = torch.randint(
                        0, diffusion.noise_scheduler.config.num_train_timesteps,
                        (B,), device=device
                    ).long()

                    # add noise to the clean images according to the noise magnitude at each diffusion iteration
                    # (this is the forward diffusion process)
                    noisy_actions = diffusion.noise_scheduler.add_noise(
                        naction, noise, timesteps)

                    # predict the noise residual
                    noise_pred = diffusion.noise_pred_net(
                        noisy_actions, timesteps, global_cond=obs_cond)

                    # L2 loss
                    loss = nn.functional.mse_loss(noise_pred, noise)

                    # optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # step lr scheduler every batch
                    # this is different from standard pytorch behavior
                    lr_scheduler.step()

                    # update Exponential Moving Average of the model weights
                    ema.step(diffusion.nets.parameters())

                    # logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)

            tglobal.set_postfix(loss=np.mean(epoch_loss))
            avg_loss = np.mean(epoch_loss)
            tglobal.set_postfix(loss=avg_loss)
            
            # Save checkpoint every 10 epochs or at the end of training
            if (epoch_idx + 1) % 50 == 0 or (epoch_idx + 1) == num_epochs:
                # Save only the state_dict of the model, including relevant submodules
                torch.save(diffusion.nets.state_dict(),  os.path.join(checkpoint_dir, f'checkpoint_{epoch_idx+1}.pth'))

    # Weights of the EMA model
    # is used for inference
    ema_nets = diffusion.nets
    ema.copy_to(ema_nets.parameters())



if __name__ == "__main__":
    train_Real_Robot()