from real_robot_network import DiffusionPolicy_Real
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt


def train_Real_Robot(continue_training=False, start_epoch = 0):
    # # for this demo, we use DDPMScheduler with 100 diffusion iterations
    diffusion = DiffusionPolicy_Real()
    device = torch.device('cuda')
    _ = diffusion.nets.to(device)

    #@markdown ### **Training**
    #@markdown
    #@markdown Takes about 2.5 hours. If you don't want to wait, skip to the next cell
    #@markdown to load pre-trained weights

    num_epochs = 1400

    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(
        parameters=diffusion.nets.parameters(),
        power=0.75)
    checkpoint_dir = "/home/lm-2023/jeon_team_ws/playback_pose/src/Diffusion_Policy_ICRA/checkpoints"
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
        num_training_steps=len(diffusion.dataloader) * num_epochs
    )
    # Log loss for epochs
    epoch_losses = []

    with tqdm(range(start_epoch, num_epochs), desc='Epoch') as tglobal:
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

                    ### Debug sequential data structure. It shoud be consecutive
                    # import matplotlib.pyplot as plt
                    # imdata1 = nimage[0].cpu()
                    # imdata1 = imdata1.numpy()
                    # imdata2 = nimage_second_view[0].cpu()
                    # imdata2 = imdata2.numpy()
          
                    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                    # for j in range(2):
                    #     # Convert the 3x96x96 tensor to a 96x96x3 image (for display purposes)
                    #     img = imdata2[j].transpose(1, 2, 0)
                        
                    #     # Plot the image on the corresponding subplot
                    #     axes[j].imshow(img)
                    #     axes[j].axis('off')  # Hide the axes

                    #     # Show the plot
                    # plt.show()  


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
            epoch_losses.append(avg_loss)
            tglobal.set_postfix(loss=avg_loss)
            
            # Save checkpoint every 10 epochs or at the end of training
            if (epoch_idx + 1) % 200 == 0 or (epoch_idx + 1) == num_epochs:
                # Save only the state_dict of the model, including relevant submodules
                torch.save(diffusion.nets.state_dict(),  os.path.join(checkpoint_dir, f'checkpoint_{epoch_idx+1}_prying_orange.pth'))
    # Plot the loss after training is complete
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()
    # Weights of the EMA model
    # is used for inference
    ema_nets = diffusion.nets
    ema.copy_to(ema_nets.parameters())



if __name__ == "__main__":
    train_Real_Robot(continue_training=False)