import numpy as np
from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import numpy as np
import math
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import collections
import zarr
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import os
from copy import deepcopy
from data_util import data_utils, center_crop, regular_center_crop

#@markdown ### **Dataset Demo**
class RealRobotDataSet(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str,
                 pred_horizon: int,
                 obs_horizon: int,
                 action_horizon: int,
                 force_mod: bool = False,
                 single_view: bool = False,
                 augment: bool = False,
                 crop: int = 1000):
        
        # read from zarr dataset
        dataset_root = zarr.open(dataset_path, 'r')
        if single_view:
            train_image_data = dataset_root['data']['images_B'][:]
            train_image_data = np.moveaxis(train_image_data, -1,1)

        else:
            # float32, [0,1], (N,96,96,3)
            train_image_data = dataset_root['data']['images_B'][:]
            train_image_data = np.moveaxis(train_image_data, -1,1)
            train_image_data_second_view = dataset_root['data']['images_A'][:]
            train_image_data_second_view = np.moveaxis(train_image_data_second_view, -1,1)

        train_image_data = regular_center_crop(train_image_data, 224, 224)

        if crop == 98:
            train_image_data = center_crop(train_image_data, crop, crop)
        else:
            print("No image change")

        # (N,3,96,96)
        # (N, D)
        train_data = {
            # first seven dims of state vector are agent (i.e. gripper) locations
            # Seven because we will use quaternion 
            'agent_pos': dataset_root['data']['state'][:,:9],
            'action': dataset_root['data']['action']
        }
        episode_ends = dataset_root['meta']['episode_ends'][:]

        # compute start and end of each state-action sequence
        # also handles padding
        indices = data_utils.create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon-1,
            pad_after=action_horizon-1)

        # compute statistics and normalized data to [-1,1]
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            stats[key] = data_utils.get_data_stats(data[:,:3])
            normalized_position = data_utils.normalize_data(data[:,:3], stats[key])
            normalized_orientation = data[:,3:9]
            # normalized_orientation = data_utils.process_quaternion(data[:,3:7])
            normalized_train_data[key] = np.hstack((normalized_position, normalized_orientation))
            ## TODO: Add code that will handle - and + sign for quaternion
        if force_mod:
            train_force_data = dataset_root['data']['state'][:,9:12]
            magnitudes, normalized_force_direction = data_utils.normalize_force_vector(train_force_data)
            stats['force_mag'] = data_utils.get_data_stats(magnitudes)
            normalized_force_mag = data_utils.normalize_force_magnitude(magnitudes, stats['force_mag'])
            normalized_force_data = np.hstack((normalized_force_mag, normalized_force_direction))
        
        # Start adding normalized training data
        normalized_train_data['image'] = train_image_data

        
        # images are already normalized
        if force_mod:      
            normalized_train_data['force'] = normalized_force_data
        if not single_view:
            normalized_train_data['image2'] = train_image_data_second_view   

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        self.force_mod = force_mod
        self.single_view = single_view
        self.augment = augment
        self.crop = crop
        if self.augment:
            if self.crop == 98:
                self.augmentation_transform = transforms.Compose([
                    transforms.RandomResizedCrop(size=(crop, crop), scale=(0.5, 1.5)),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
                ])
            else:
                self.augmentation_transform = transforms.Compose([
                    transforms.RandomResizedCrop(size=(224, 224), scale=(0.5, 1.5)),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.2),
                    transforms.Resize((224, 224)),  # Ensures final output is exactly 224x224

                ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, \
            sample_start_idx, sample_end_idx = self.indices[idx]

        # get nomralized data using these indices
        nsample = data_utils.sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        # Apply augmentation to images only
        if self.augment:
            # Convert image to PIL format for augmentation if needed
            img_tensor = torch.tensor(nsample['image'][:self.obs_horizon, :])
            img_augmented = [self.augmentation_transform(img) for img in img_tensor]
            nsample['image'] = torch.stack(img_augmented)
            nsample['image'] = np.array(nsample['image'])
            nsample['image'] = nsample['image'][:self.obs_horizon,:]

            if not self.single_view:
                img_tensor2 = torch.tensor(nsample['image2'][:self.obs_horizon, :])
                img_augmented2 = [self.augmentation_transform(img) for img in img_tensor2]
                nsample['image2'] = torch.stack(img_augmented2)
                nsample['image2'] = np.array(nsample['image2'])
        else:
            # Convert images to tensor without augmentation
            nsample['image'] = nsample['image'][:self.obs_horizon, :]
            if not self.single_view:
                nsample['image2'] = nsample['image2'][:self.obs_horizon, :]
    
        nsample['agent_pos'] = nsample['agent_pos'][:self.obs_horizon,:]
        if self.force_mod:
            # discard unused observations
            if self.augment:
                noise_std = 0.00005
                force_arr = nsample['force'][:self.obs_horizon, :]
                scaling_factors = np.random.uniform(0.9, 1.1)
                force_augmented = force_arr * scaling_factors + np.random.normal(0, noise_std, size=force_arr.shape)
                nsample['force'] = force_augmented.astype(np.float32)
            else:
                nsample['force'] = nsample['force'][:self.obs_horizon,:]
        # if not self.single_view:
        #     # discard unused observations
        #     nsample['image2'] = nsample['image2'][:self.obs_horizon,:]

        return nsample
