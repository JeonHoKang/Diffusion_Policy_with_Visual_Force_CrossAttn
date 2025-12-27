from real_robot_network import DiffusionPolicy_Real
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import json
torch.cuda.empty_cache()
from torch.utils.data import ConcatDataset
import hydra
from omegaconf import DictConfig
from real_robot_dataset import RealRobotDataSet

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
    

@hydra.main(version_base=None, config_path="config", config_name="resnet_delta_with_force_single_view_force_Linear_crossattn_hybrid")
def train_Real_Robot(cfg: DictConfig):
    dataset_path = cfg.model_config.dataset_path
    action_dim  = cfg.model_config.action_dim
    pred_horizon = cfg.model_config.pred_horizon
    obs_horizon  = cfg.model_config.obs_horizon
    action_horizon = cfg.model_config.action_horizon
    continue_training=  cfg.model_config.continue_training
    batch_size = cfg.model_config.batch_size
    start_epoch = cfg.model_config.start_epoch
    end_epoch= cfg.model_config.end_epoch
    encoder:str = cfg.model_config.encoder
    action_def: str = cfg.model_config.action_def
    force_mod: bool = cfg.model_config.force_mod
    single_view: bool = cfg.model_config.single_view
    force_encode = cfg.model_config.force_encode
    force_encoder = cfg.model_config.force_encoder
    cross_attn = cfg.model_config.cross_attn
    hybrid = cfg.model_config.hybrid
    augment = cfg.model_config.augment
    crop = cfg.model_config.crop

    if force_encode:
        cross_attn = False
    if cross_attn:
        force_encode = False

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")   # Apple Silicon (Metal)
    else:
        device = torch.device("cpu")

    print("Using device:", device)


    if augment:
        real_robot_dataset = RealRobotDataSet(
                                            dataset_path,
                                            pred_horizon,
                                            obs_horizon,
                                            action_horizon,
                                            force_mod,
                                            single_view,
                                            augment = False,
                                            crop = crop
                                            )
        
        real_robot_dataset_augmented = RealRobotDataSet(
            dataset_path=dataset_path,
            pred_horizon=pred_horizon,
            obs_horizon=obs_horizon,
            action_horizon=action_horizon,
            force_mod = force_mod,
            single_view=single_view,
            augment = True,
            crop = crop
        )
        dataset = ConcatDataset([real_robot_dataset, real_robot_dataset_augmented])

    else:
        real_robot_dataset = RealRobotDataSet(
                                            dataset_path,
                                            pred_horizon,
                                            obs_horizon,
                                            action_horizon,
                                            force_mod,
                                            single_view,
                                            augment = False,
                                            crop = crop
                                            )
        dataset = real_robot_dataset
        # create dataloader

    stats = real_robot_dataset.stats

    # Save the stats to a file
    with open(f'stats_{encoder}_{action_def}_vn.json', 'w') as f:
        json.dump(stats, f, cls=NumpyEncoder)
        print("stats saved")

    # self.dataloader = data_loader_augmented
    # self.data_loader_augmented = data_loader_augmented
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process afte each epoch
        persistent_workers=True,
    )


    # # for this demo, we use DDPMScheduler with 100 diffusion iterations
    diffusion = DiffusionPolicy_Real(
                                    action_dim  = action_dim,
                                    pred_horizon = pred_horizon,
                                    obs_horizon  = obs_horizon,
                                    action_horizon = action_horizon,
                                    batch_size = batch_size,
                                    encoder= encoder,
                                    action_def = action_def, 
                                    force_mod = force_mod, 
                                    single_view= single_view, 
                                    force_encode=force_encode,
                                    force_encoder=force_encoder,
                                    cross_attn=cross_attn,
                                    hybrid = hybrid,
                                    crop = crop)
    _ = diffusion.nets.to(device)

    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(
        parameters=diffusion.nets.parameters(),
        power=0.75)
    
    checkpoint_dir = "/"
    # To continue t raining load and set the start epoch
    if continue_training:
        start_epoch = 1500
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{start_epoch}.pth')  # Replace with the correct path
        # Load the saved state_dict into the model
        checkpoint = torch.load(checkpoint_path)
        diffusion.nets.load_state_dict(checkpoint)  # Load model state
        print("Successfully loaded Checkpoint")

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
        num_training_steps=len(dataloader) * end_epoch
    )
    # Log loss for epochs
    
    epoch_losses = []

    with tqdm(range(start_epoch, end_epoch), desc='Epoch') as tglobal:
        # epoch
        for epoch_idx in tglobal:
            epoch_loss = list()
            # current_loader = diffusion.dataloader
            # batch loop
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    # data normalized in dataset
                    # device transfer
                    nimage = nbatch['image'][:,:diffusion.obs_horizon].to(device)

                    if not single_view:
                        nimage_second_view = nbatch['image2'][:,:diffusion.obs_horizon].to(device)
                    
                    if force_mod:
                        nforce = nbatch['force'][:,:diffusion.obs_horizon].to(device)
                    else:
                        nforce = None
                    nagent_pos = nbatch['agent_pos'][:,:diffusion.obs_horizon].to(device)
                    naction = nbatch['action'].to(device)
                    
                    # Debug sequential data structure. It shoud be consecutive
                    # import matplotlib.pyplot as plt
                    # imdata1 = nimage[0].cpu()
                    # imdata1 = imdata1.numpy()
                    # print(f"shape of the image data:", imdata1.shape)
                    # # imdata2 = nimage_duplicate[0].cpu()
                    # # imdata2 = imdata2.numpy()
                    # # print(f"shape of the image data:", imdata2.shape)

                    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                    # for j in range(2):
                    #     # Convert the 3x96x96 tensor to a 96x96x3 image (for display purposes)
                    #     img = imdata1[j].transpose(1, 2, 0)

                    # #     # Plot the image on the corresponding subplot
                    #     axes[j].imshow(img)
                    #     axes[j].axis('off')  # Hide the axes
                    #     # Show the plot
                    # plt.show()  
                    # ### For double realsense config only
                    # for j in range(2):
                    #     # Convert the 3x96x96 tensor to a 96x96x3 image (for display purposes)
                    #     img2 = imdata2[j].transpose(1, 2, 0)

                    #     # Plot the image on the corresponding subplot
                    #     axes[j].imshow(img2)
                    #     axes[j].axis('off')  # Hide the axes
                    #     # Show the plot
                    # plt.show()

                    if encoder == "resnet":
                            image_input = nimage.flatten(end_dim=1)
                    elif encoder == "viT":
                            image_input = nimage
                    B = nagent_pos.shape[0]
                    if not cross_attn:
                        # encoder vision features
                        image_features = diffusion.nets['vision_encoder'](
                            image_input)
                        image_features = image_features.reshape(
                            *nimage.shape[:2],-1)
                    # (B,obs_horizon,D)

                    if not single_view:
                        # encoder vision features
                        image_features_second_view = diffusion.nets['vision_encoder2'](
                            nimage_second_view.flatten(end_dim=1))
                        image_features_second_view = image_features_second_view.reshape(
                            *nimage_second_view.shape[:2],-1)
              
                    if force_mod and force_encode:
                        force_feature = diffusion.nets['force_encoder'](nforce)
                        # force_feature = force_feature.reshape(
                        #     *nforce.shape[:2],-1)
                    else:
                        force_feature = nforce
                    
                    if cross_attn:
                        joint_features = diffusion.nets['cross_attn_encoder'](
                            image_input, (nforce))

                    # (B,obs_horizon,D)
                    if force_mod and single_view and not cross_attn:
                        obs_features = torch.cat([image_features, force_feature, nagent_pos], dim=-1)
                    elif force_mod and not single_view and not cross_attn:
                        obs_features = torch.cat([image_features, image_features_second_view, force_feature, nagent_pos], dim=-1)
                    elif not force_mod and single_view:
                        obs_features = torch.cat([image_features, nagent_pos], dim=-1)
                    elif not force_mod and not single_view:
                        obs_features = torch.cat([image_features, image_features_second_view , nagent_pos], dim=-1)
                    elif single_view and cross_attn:
                        # TODO: If hybrid is true, then add force feature on top of it.
                        if hybrid:
                            obs_features = torch.cat([joint_features, force_feature, nagent_pos], dim=-1)
                        else:
                            obs_features = torch.cat([joint_features, nagent_pos], dim=-1)
                    elif not single_view and cross_attn:
                        # TODO: If hybrid is true, then add force feature on top of it.
                        if hybrid:
                            obs_features = torch.cat([joint_features, image_features_second_view, force_feature, nagent_pos], dim=-1)
                        else:
                            obs_features = torch.cat([joint_features, image_features_second_view, nagent_pos], dim=-1)
                    else:
                        print("Check your configuration for training")

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
            epoch_losses.append(avg_loss)
            tglobal.set_postfix(loss=avg_loss)
            
            # Save checkpoint every 10 epochs or at the end of training
            if epoch_idx > 100:
                if (epoch_idx + 1) % 100 == 0 or (epoch_idx + 1) == end_epoch:
                    # Save only the state_dict of the model, including relevant submodules
                    torch.save(diffusion.nets.state_dict(),  os.path.join(checkpoint_dir, f'{cfg.name}_{epoch_idx+1}_bench.pth'))
    # Plot the loss after training is complete
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, end_epoch + 1), epoch_losses, marker='o', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epocshs')
    plt.legend()
    plt.grid(True)
    plt.show()
    # Weights of the EMA model
    # is used for inference
    ema_nets = diffusion.nets
    ema.copy_to(ema_nets.parameters())


if __name__ == "__main__":
    train_Real_Robot()