import numpy as np
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
import os
from torchvision import transforms


#@markdown ### **Dataset**
#@markdown
#@markdown Defines `PushTImageDataset` and helper functions
#@markdown
#@markdown The dataset class
#@markdown - Load data ((image, agent_pos), action) from a zarr storage
#@markdown - Normalizes each dimension of agent_pos and action to [-1,1]
#@markdown - Returns
#@markdown  - All possible segments with length `pred_horizon`
#@markdown  - Pads the beginning and the end of each episode with repetition
#@markdown  - key `image`: shape (obs_hoirzon, 3, 96, 96)
#@markdown  - key `agent_pos`: shape (obs_hoirzon, 2)
#@markdown  - key `action`: shape (pred_horizon, 2)

class data_utils:
    def __init__(self):
        pass
    def create_sample_indices(
            episode_ends:np.ndarray, sequence_length:int,
            pad_before: int=0, pad_after: int=0):
        indices = list()
        for i in range(len(episode_ends)):
            start_idx = 0
            if i > 0:
                start_idx = episode_ends[i-1]
            end_idx = episode_ends[i]
            episode_length = end_idx - start_idx

            min_start = -pad_before
            max_start = episode_length - sequence_length + pad_after

            # range stops one idx before end
            for idx in range(min_start, max_start+1):
                buffer_start_idx = max(idx, 0) + start_idx
                buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
                start_offset = buffer_start_idx - (idx+start_idx)
                end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
                sample_start_idx = 0 + start_offset
                sample_end_idx = sequence_length - end_offset
                indices.append([
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx])
        indices = np.array(indices)
        return indices


    def sample_sequence(train_data, sequence_length,
                        buffer_start_idx, buffer_end_idx,
                        sample_start_idx, sample_end_idx):
        result = dict()
        for key, input_arr in train_data.items():
            sample = input_arr[buffer_start_idx:buffer_end_idx]
            data = sample
            if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
                data = np.zeros(
                    shape=(sequence_length,) + input_arr.shape[1:],
                    dtype=input_arr.dtype)
                if sample_start_idx > 0:
                    data[:sample_start_idx] = sample[0]
                if sample_end_idx < sequence_length:
                    data[sample_end_idx:] = sample[-1]
                data[sample_start_idx:sample_end_idx] = sample
            result[key] = data
        return result

    # normalize data
    def get_data_stats(data):
        data = data.reshape(-1,data.shape[-1])
        stats = {
            'min': np.min(data, axis=0),
            'max': np.max(data, axis=0)
        }
        return stats

    def normalize_data(data, stats):
        # nomalize to [0,1]
        ndata = (data - stats['min']) / (stats['max'] - stats['min'])
        # normalize to [-1, 1]
        ndata = ndata * 2 - 1
        return ndata

    def unnormalize_data(ndata, stats):
        ndata = (ndata + 1) / 2
        data = ndata * (stats['max'] - stats['min']) + stats['min']
        return data

    def process_quaternion(quaternion_array):
        negative_real_indices = quaternion_array[:, 3] < 0
        # Negate the entire quaternion for rows where the real component is negative
        quaternion_array[negative_real_indices] *= -1
        
        return quaternion_array

    def normalize_force_vector(force_vector):
        
        # Calculate the magnitude of each force vector along axis 1
        magnitudes = np.linalg.norm(force_vector, axis=1, keepdims=True)
        
        # Avoid division by zero by using a small epsilon value
        magnitudes[magnitudes == 0] = 1e-8
        
        # Normalize each force vector by dividing by its magnitude
        normalized_forces = force_vector / magnitudes
        return magnitudes, normalized_forces
    
    def normalize_force_magnitude(data, stats):
        # nomalize to [0,1]
        ndata = (data - stats['min']) / (stats['max'] - stats['min'])
        return ndata
    
def center_crop(images, crop_height, crop_width):
    # Get original dimensions
    N, C, H, W = images.shape
    assert crop_height <= H and crop_width <= W, "Crop size should be smaller than the original size"
    
    # Calculate the center + 20 only when using 98 and 128 is -20 for start_x only
    start_y = (H - crop_height + 20) // 2
    start_x = (W - crop_width - 20) // 2
    # start_y = (H - crop_height) // 2
    # start_x = (W - crop_width - 20) // 2  
    # Perform cropping
    cropped_images = images[:, :, start_y:start_y + crop_height, start_x:start_x + crop_width]
    
    return cropped_images

# dataset

class PushTImageDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str,
                 pred_horizon: int,
                 obs_horizon: int,
                 action_horizon: int):

        # read from zarr dataset
        dataset_root = zarr.open(dataset_path, 'r')

        # float32, [0,1], (N,96,96,3)
        train_image_data = dataset_root['data']['img'][:]
        train_image_data = np.moveaxis(train_image_data, -1,1)
        # Perform center cropping to 224x224
        # (N,3,96,96)
        # (N, D)
        train_data = {
            # first two dims of state vector are agent (i.e. gripper) locations
            'agent_pos': dataset_root['data']['state'][:,:2],
            'action': dataset_root['data']['action'][:]
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
            stats[key] = data_utils.get_data_stats(data)
            normalized_train_data[key] = data_utils.normalize_data(data, stats[key])

        # images are already normalized
        normalized_train_data['image'] = train_image_data

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

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

        # discard unused observations
        nsample['image'] = nsample['image'][:self.obs_horizon,:]
        nsample['agent_pos'] = nsample['agent_pos'][:self.obs_horizon,:]
        return nsample
    



#@markdown ### **Dataset Demo**
class RealRobotDataSet(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str,
                 pred_horizon: int,
                 obs_horizon: int,
                 action_horizon: int,
                 Transformer: bool = False,
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
        if Transformer:
            print("center crop transformer")
            train_image_data = center_crop(train_image_data, 224, 224)
        elif crop ==  128:
            # If crop parameter 64
            train_image_data = center_crop(train_image_data, crop, crop)
        else:
            ("No Cropping")

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
            if self.crop == 128:
                self.augmentation_transform = transforms.Compose([
                    transforms.RandomResizedCrop(size=(crop, crop), scale=(0.5, 1.5)),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.2),
                ])
            else:
                self.augmentation_transform = transforms.Compose([
                    transforms.RandomResizedCrop(size=(240, 320), scale=(0.5, 1.5)),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.2),
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

        # nsample['image'] = nsample['image'][:self.obs_horizon,:]
        nsample['agent_pos'] = nsample['agent_pos'][:self.obs_horizon,:]

        if self.force_mod:
            # discard unused observations
            if self.augment:
                noise_std = 0.00005
                force_arr = nsample['force'][:self.obs_horizon, :]
                scaling_factors = np.random.uniform(0.9, 1.2)
                force_augmented = force_arr * scaling_factors + np.random.normal(0, noise_std, size=force_arr.shape)
                nsample['force'] = force_augmented.astype(np.float32)
            else:
                nsample['force'] = nsample['force'][:self.obs_horizon,:]
        # if not self.single_view:
        #     # discard unused observations
        #     nsample['image2'] = nsample['image2'][:self.obs_horizon,:]

        return nsample
