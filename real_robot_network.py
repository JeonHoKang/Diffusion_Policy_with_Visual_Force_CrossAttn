#@markdown ### **Imports**
# diffusion policy import
from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import os
from data_util import RealRobotDataSet
from train_utils import train_utils
import json
from transformer_for_diffusion import BaseLowdimPolicy, TransformerForDiffusion, ModuleAttrMixin
from normalizer import LinearNormalizer
#@markdown ### **Network**
#@markdown
#@markdown Defines a 1D UNet architecture `ConditionalUnet1D`
#@markdown as the noies prediction network
#@markdown
#@markdown Components
#@markdown - `SinusoidalPosEmb` Positional encoding for the diffusion iteration k
#@markdown - `Downsample1d` Strided convolution to reduce temporal resolution
#@markdown - `Upsample1d` Transposed convolution to increase temporal resolution
#@markdown - `Conv1dBlock` Conv1d --> GroupNorm --> Mish
#@markdown - `ConditionalResidualBlock1D` Takes two inputs `x` and `cond`. \
#@markdown `x` is passed through 2 `Conv1dBlock` stacked together with residual connection.
#@markdown `cond` is applied to `x` with [FiLM](https://arxiv.org/abs/1709.07871) conditioning.


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
    
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            cond_dim,
            kernel_size=3,
            n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(
            embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:,0,...]
        bias = embed[:,1,...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(self,
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=5,
        n_groups=8
        ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """

        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        print("number of parameters: {:e}".format(
            sum(p.numel() for p in self.parameters()))
        )

    def forward(self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            global_cond=None):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        # (B,T,C)
        sample = sample.moveaxis(-1,-2)
        # (B,C,T)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1,-2)
        # (B,T,C)
        return x



# download demonstration data from Google Drive
# dataset_path = "pusht_cchi_v7_replay.zarr.zip"
# if not os.path.isfile(dataset_path):
#     id = "1KY1InLurpMvJDRb14L9NlXT_fEsCvVUq&confirm=t"
#     gdown.download(id=id, output=dataset_path, quiet=False)

dataset_path = "/home/jeon/jeon_ws/diffusion_policy/src/diffusion_cam/IU_data_4.zarr.zip"

#@markdown ### **Network Demo**
class DiffusionPolicy_Real:     
    def __init__(self, train=True):

        # construct ResNet18 encoder
        # if you have multiple camera views, use seperate encoder weights for each view.
        # Resnet18 and resnet34 both have same dimension for the output
        vision_encoder = train_utils().get_resnet('resnet18')
        # Define Second vision encoder
        vision_encoder2 = train_utils().get_resnet('resnet18')

        # IMPORTANT!
        # replace all BatchNorm with GroupNorm to work with EMA
        # performance will tank if you forget to do this!
        vision_encoder = train_utils().replace_bn_with_gn(vision_encoder)
        vision_encoder2 = train_utils().replace_bn_with_gn(vision_encoder2)
        # ResNet18 has output dim of 512 X 2 because two views
        vision_feature_dim = 1024
        # agent_pos is six (x,y,z, roll, pitch, yaw) dimensional
        lowdim_obs_dim = 6
        # observation feature has 514 dims in total per step
        obs_dim = vision_feature_dim + lowdim_obs_dim
        # action dimension should also correspond with the state dimension (x,y,z, roll, pitch, yaw)
        action_dim = 6
        # parameters
        pred_horizon = 16
        obs_horizon = 2
        action_horizon = 8
        #|o|o|                             observations: 2
        #| |a|a|a|a|a|a|a|a|               actions executed: 8
        #|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
        if train:
            # create dataset from file
            dataset = RealRobotDataSet(
                dataset_path=dataset_path,
                pred_horizon=pred_horizon,
                obs_horizon=obs_horizon,
                action_horizon=action_horizon
            )
            # save training data statistics (min, max) for each dim
            stats = dataset.stats
           # Save the stats to a file
            with open('stats.json', 'w') as f:
                json.dump(stats, f, cls=NumpyEncoder)
                print("stats saved")
            # create dataloader
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=64,
                num_workers=4,
                shuffle=True,
                # accelerate cpu-gpu transfer
                pin_memory=True,
                # don't kill worker process afte each epoch
                persistent_workers=True
            )

            self.dataloader = dataloader
            self.stats = stats

        #### For debugging purposes uncomment
        # import matplotlib.pyplot as plt
        # imdata = dataset[100]['image']
        # if imdata.dtype == np.float32 or imdata.dtype == np.float64:
        #     imdata = imdata / 255.0
        # img1 = imdata[0]
        # img2 = imdata[1]
        # # Loop through the two different "channels"
        # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        # for i in range(2):
        #     # Convert the 3x96x96 tensor to a 96x96x3 image (for display purposes)
        #     img = np.transpose(imdata[i], (1, 2, 0))
            
        #     # Display the image in the i-th subplot
        #     axes[i].imshow(img)
        #     axes[i].set_title(f'Channel {i + 1}')
        #     axes[i].axis('off')

        # # Show the plot
        # plt.show()  

        # # Check if both images are exactly the same
        # are_equal = np.array_equal(img1, img2)

        # if are_equal:
        #     print("The images are the same.")
        # else:
        #     print("The images are different.")
        ######### End ########

        
            # visualize data in batch
            batch = next(iter(dataloader))
            print("batch['image'].shape:", batch['image'].shape)
            print("batch['agent_pos'].shape:", batch['agent_pos'].shape)
            print("batch['action'].shape", batch['action'].shape)
            self.batch = batch

        # create network object
        noise_pred_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=obs_dim*obs_horizon
        )

        # the final arch has 2 parts
        nets = nn.ModuleDict({
            'vision_encoder': vision_encoder,
            'vision_encoder2': vision_encoder2,
            'noise_pred_net': noise_pred_net
        })
        # diffusion iteration
        num_diffusion_iters = 100

        noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_iters,
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule='squaredcos_cap_v2',
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # our network predicts noise (instead of denoised action)
            prediction_type='epsilon'
        )

        
        self.nets = nets
        self.noise_scheduler = noise_scheduler
        self.num_diffusion_iters = num_diffusion_iters
        self.obs_horizon = obs_horizon
        self.obs_dim = obs_dim
        self.vision_encoder = vision_encoder
        self.vision_encoder2 = vision_encoder2
        self.noise_pred_net = noise_pred_net
        self.action_horizon = action_horizon
        self.pred_horizon = pred_horizon
        self.lowdim_obs_dim = lowdim_obs_dim
        self.action_dim = action_dim
# # demo
# with torch.no_grad():
#     # example inputs
#     image = torch.zeros((1, obs_horizon,3,96,96))
#     agent_pos = torch.zeros((1, obs_horizon, 2))
#     # vision encoder
#     image_features = nets['vision_encoder'](
#         image.flatten(end_dim=1))
#     # (2,512)
#     image_features = image_features.reshape(*image.shape[:2],-1)
#     # (1,2,512)
#     obs = torch.cat([image_features, agent_pos],dim=-1)
#     # (1,2,514)

#     noised_action = torch.randn((1, pred_horizon, action_dim))
#     diffusion_iter = torch.zeros((1,))

#     # the noise prediction network
#     # takes noisy action, diffusion iteration and observation as input
#     # predicts the noise added to action
#     noise = nets['noise_pred_net'](
#         sample=noised_action,
#         timestep=diffusion_iter,
#         global_cond=obs.flatten(start_dim=1))

#     # illustration of removing noise
#     # the actual noise removal is performed by NoiseScheduler
#     # and is dependent on the diffusion noise schedule
#     denoised_action = noised_action - noise



class LowdimMaskGenerator(ModuleAttrMixin):
    def __init__(self,
        action_dim, obs_dim,
        # obs mask setup
        max_n_obs_steps=2, 
        fix_obs_steps=True, 
        # action mask
        action_visible=False
        ):
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.max_n_obs_steps = max_n_obs_steps
        self.fix_obs_steps = fix_obs_steps
        self.action_visible = action_visible

    @torch.no_grad()
    def forward(self, shape, seed=None):
        device = self.device
        B, T, D = shape
        assert D == (self.action_dim + self.obs_dim)

        # create all tensors on this device
        rng = torch.Generator(device=device)
        if seed is not None:
            rng = rng.manual_seed(seed)

        # generate dim mask
        dim_mask = torch.zeros(size=shape, 
            dtype=torch.bool, device=device)
        is_action_dim = dim_mask.clone()
        is_action_dim[...,:self.action_dim] = True
        is_obs_dim = ~is_action_dim

        # generate obs mask
        if self.fix_obs_steps:
            obs_steps = torch.full((B,), 
            fill_value=self.max_n_obs_steps, device=device)
        else:
            obs_steps = torch.randint(
                low=1, high=self.max_n_obs_steps+1, 
                size=(B,), generator=rng, device=device)
            
        steps = torch.arange(0, T, device=device).reshape(1,T).expand(B,T)
        obs_mask = (steps.T < obs_steps).T.reshape(B,T,1).expand(B,T,D)
        obs_mask = obs_mask & is_obs_dim

        # generate action mask
        if self.action_visible:
            action_steps = torch.maximum(
                obs_steps - 1, 
                torch.tensor(0,
                    dtype=obs_steps.dtype, 
                    device=obs_steps.device))
            action_mask = (steps.T < action_steps).T.reshape(B,T,1).expand(B,T,D)
            action_mask = action_mask & is_action_dim

        mask = obs_mask
        if self.action_visible:
            mask = mask | action_mask
        
        return mask

    
class DiffusionTransformerLowdimPolicy(BaseLowdimPolicy):
    def __init__(self, 
            model: TransformerForDiffusion,
            noise_scheduler: DDPMScheduler,
            horizon, 
            obs_dim, 
            action_dim, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_cond=False,
            pred_action_steps_only=False,
            # parameters passed to step
            **kwargs):
        super().__init__()
        if pred_action_steps_only:
            assert obs_as_cond

        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_cond = obs_as_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            cond=None, generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """

        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict # not implemented yet
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        cond = None
        cond_data = None
        cond_mask = None
        if self.obs_as_cond:
            cond = nobs[:,:To]
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs[:,:To]
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            cond=cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        if not self.obs_as_cond:
            nobs_pred = nsample[...,Da:]
            obs_pred = self.normalizer['obs'].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:,start:end]
            result['action_obs_pred'] = action_obs_pred
            result['obs_pred'] = obs_pred
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self, weight_decay: float, learning_rate: float, betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        return self.model.configure_optimizers(
                weight_decay=weight_decay, 
                learning_rate=learning_rate, 
                betas=tuple(betas))

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']
        action = nbatch['action']

        # handle different ways of passing observation
        cond = None
        trajectory = action
        if self.obs_as_cond:
            cond = obs[:,:self.n_obs_steps,:]
            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To - 1
                end = start + self.n_action_steps
                trajectory = action[:,start:end]
        else:
            trajectory = torch.cat([action, obs], dim=-1)
        
        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
