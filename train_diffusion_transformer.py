
import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil

from normalizer import dict_apply, optimizer_to
from real_robot_network import DiffusionTransformerLowdimPolicy

from diffusers.optimization import get_scheduler
from ema_for_diffusion import EMAModel
import gdown

from data_util import PushTImageDataset

# download demonstration data from Google Drive
dataset_path = "pusht_cchi_v7_replay.zarr.zip"
if not os.path.isfile(dataset_path):
    id = "1KY1InLurpMvJDRb14L9NlXT_fEsCvVUq&confirm=t"
    gdown.download(id=id, output=dataset_path, quiet=False)


# %%
class TrainDiffusionTransformerLowdimWorkspace:
    def __init__(self, resume_train=False):
        # set seed
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.resume_train = resume_train
        # configure model
        self.model = DiffusionTransformerLowdimPolicy(
            input_dim=1024+6,
            output_dim=6,
            horizon=8,
            n_obs_steps=3,
            # cond_dim=10,
            causal_attn=True,
        )
        self.ema_model: DiffusionTransformerLowdimPolicy = None
        self.ema_model = copy.deepcopy(self.model)

        weight_decay = 1.0e-4
        learning_rate  = 1.0e-3
        betas = (0.9, 0.95)

        # configure training state
        self.optimizer = self.model.get_optimizer(weight_decay=weight_decay, 
        learning_rate=learning_rate, 
        betas=betas)

        self.global_step = 0
        self.epoch = 0

    def run(self):

        action_dim = 6
        # parameters
        pred_horizon = 16
        obs_horizon = 2
        action_horizon = 8
        resume_train = self.resume_train
        # resume training
        checkpoint_dir = ""
        if resume_train:
            start_epoch = 59
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{start_epoch}.pth')  # Replace with the correct path
            # Load the saved state_dict into the model
            checkpoint = torch.load(checkpoint_path)
            # diffusion.nets.load_state_dict(checkpoint)  # Load model state
            start_epoch = 60
            print("Successfully loaded Checkpoint")


        train_dataloader = None
        # Set number of epochs to train
        num_epochs = 1000
        # create dataset from file
        dataset = PushTImageDataset(
            dataset_path=dataset_path,
            pred_horizon=pred_horizon,
            obs_horizon=obs_horizon,
            action_horizon=action_horizon
        )
        # save training data statistics (min, max) for each dim
        stats = dataset.stats

        # create dataloader
        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=64,
            num_workers=4,
            shuffle=True,
            # accelerate cpu-gpu transfer
            pin_memory=True,
            # don't kill worker process afte each epoch
            persistent_workers=True
        )

        # configure validation dataset
        # val_dataset = dataset.get_validation_dataset()
        # val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            name='cosine',
            optimizer=self.optimizer,
            num_warmup_steps=500,
            num_training_steps=(
                len(train_dataloader) * num_epochs)
        )

        # configure ema
        ema: EMAModel = None
        self.ema_model = copy.deepcopy(self.model)


        # device transfer
        device = torch.device('cuda')
        self.model.to(device)
        self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        gradient_accumulate_every = 1
        tqdm_interval_sec =1.0

        for local_epoch_idx in range(num_epochs):
            step_log = dict()
            # ========= train for this epoch ==========
            train_losses = list()
            with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                    leave=False, mininterval= tqdm_interval_sec) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    # device transfer
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    if train_sampling_batch is None:
                        train_sampling_batch = batch

                    # compute loss
                    raw_loss = self.model.compute_loss(batch)
                    loss = raw_loss / gradient_accumulate_every
                    loss.backward()

                    # step optimizer
                    if self.global_step % gradient_accumulate_every == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()

                    # update ema
                    ema.step(self.model)

                    # logging
                    raw_loss_cpu = raw_loss.item()
                    tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                    train_losses.append(raw_loss_cpu)
                    step_log = {
                        'train_loss': raw_loss_cpu,
                        'global_step': self.global_step,
                        'epoch': self.epoch,
                        'lr': lr_scheduler.get_last_lr()[0]
                    }

                    is_last_batch = (batch_idx == (len(train_dataloader)-1))
                    if not is_last_batch:
                        self.global_step += 1


            # at the end of each epoch
            # replace train_loss with epoch average
            train_loss = np.mean(train_losses)
            # step_log['train_loss'] = train_loss

            # ========= eval for this epoch ==========
            policy = self.model
            policy = self.ema_model
            policy.eval()

            # run validation
            # if (self.epoch % cfg.training.val_every) == 0:
            #     with torch.no_grad():
            #         val_losses = list()
            #         with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
            #                 leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
            #             for batch_idx, batch in enumerate(tepoch):
            #                 batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
            #                 loss = self.model.compute_loss(batch)
            #                 val_losses.append(loss)
            #                 if (cfg.training.max_val_steps is not None) \
            #                     and batch_idx >= (cfg.training.max_val_steps-1):
            #                     break
            #         if len(val_losses) > 0:
            #             val_loss = torch.mean(torch.tensor(val_losses)).item()
            #             # log epoch average validation loss
            #             step_log['val_loss'] = val_loss
        
            # # run diffusion sampling on a training batch
            # if (self.epoch % cfg.training.sample_every) == 0:
            #     with torch.no_grad():
            #         # sample trajectory from training set, and evaluate difference
            #         batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
            #         obs_dict = {'obs': batch['obs']}
            #         gt_action = batch['action']
                    
            #         result = policy.predict_action(obs_dict)
            #         if cfg.pred_action_steps_only:
            #             pred_action = result['action']
            #             start = cfg.n_obs_steps - 1
            #             end = start + cfg.n_action_steps
            #             gt_action = gt_action[:,start:end]
            #         else:
            #             pred_action = result['action_pred']
            #         mse = torch.nn.functional.mse_loss(pred_action, gt_action)
            #         step_log['train_action_mse_error'] = mse.item()
            #         del batch
            #         del obs_dict
            #         del gt_action
            #         del result
            #         del pred_action
            #         del mse


            ## TODO: Save checkpoint 
            # checkpoint
            # if (self.epoch % cfg.training.checkpoint_every) == 0:
            #     # checkpointing
            #     if cfg.checkpoint.save_last_ckpt:
            #         self.save_checkpoint()
            #     if cfg.checkpoint.save_last_snapshot:
            #         self.save_snapshot()

                # sanitize metric names
                # metric_dict = dict()
                # for key, value in step_log.items():
                #     new_key = key.replace('/', '_')
                #     metric_dict[new_key] = value

            # ========= eval end for this epoch ==========
            policy.train()

            # end of epoch
            # log of last step is combined with validation and rollout

            self.global_step += 1
            self.epoch += 1

def main(cfg):
    workspace = TrainDiffusionTransformerLowdimWorkspace()
    workspace.run()

if __name__ == "__main__":
    main()