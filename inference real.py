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
from real_robot_network import DiffusionPolicy_Real
from data_util import data_utils
import cv2
from train_utils import train_utils
import pyrealsense2 as rs
import os
from scipy.spatial.transform import Rotation as R
import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPositionFK
from moveit_msgs.msg import RobotState, MoveItErrorCodes
from geometry_msgs.msg import Pose
from data_collection.submodules.wait_for_message import wait_for_message

#@markdown ### **Loading Pretrained Checkpoint**
#@markdown Set `load_pretrained = True` to load pretrained weights.

#@markdown ### **Network Demo**
class EndEffectorPoseNode(Node):
    timeout_sec_ = 5.0
    joint_state_topic_ = "joint_states"
    fk_srv_name_ = "compute_fk"
    base_ = "link_0"
    end_effector_ = "link_ee"

    def __init__(self) -> None:
        super().__init__("end_effector_pose_node")

        self.fk_client_ = self.create_client(GetPositionFK, self.fk_srv_name_)
        if not self.fk_client_.wait_for_service(timeout_sec=self.timeout_sec_):
            self.get_logger().error("FK service not available.")
            exit(1)

    def get_fk(self) -> Pose | None:
        current_joint_state_set, current_joint_state = wait_for_message(
            JointState, self, self.joint_state_topic_, time_to_wait=1.0
        )
        if not current_joint_state_set:
            self.get_logger().error("Failed to get current joint state")
            return None

        current_robot_state = RobotState()
        current_robot_state.joint_state = current_joint_state

        request = GetPositionFK.Request()

        request.header.frame_id = self.base_
        request.header.stamp = self.get_clock().now().to_msg()

        request.fk_link_names.append(self.end_effector_)
        request.robot_state = current_robot_state

        future = self.fk_client_.call_async(request)

        rclpy.spin_until_future_complete(self, future)
        if future.result() is None:
            self.get_logger().error("Failed to get FK solution")
            return None
        
        response = future.result()
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(
                f"Failed to get FK solution: {response.error_code.val}"
            )
            return None
        
        return response.pose_stamped[0].pose

class EvaluateRealRobot:
    # construct ResNet18 encoder
    # if you have multiple camera views, use seperate encoder weights for each view.
    def __init__(self, max_steps):
        # TODO: 
        diffusion = DiffusionPolicy_Real()
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
        ############# Reference for environment #######
        # limit enviornment interaction to 200 steps before termination
        ###TODO: Swap this environment with the real data
        #     env = PushTImageEnv()
        #     # use a seed >200 to avoid initial states seen in the training dataset
        #     env.seed(100000)

        #     # get first observation
        #     obs, info = env.reset()
            
        #  # keep a queue of last 2 steps of observations
        #     obs_deque = collections.deque(
        #         [obs] * diffusion.obs_horizon, maxlen=diffusion.obs_horizon)
        # save visualization and rewards
        # imgs = [env.render(mode='rgb_array')]
        
        rewards = list()
        step_idx = 0
        # device transfer
        device = torch.device('cuda')
        _ = diffusion.nets.to(device)
        # Initialize realsense camera
        pipeline_A = rs.pipeline()
        pipeline_B = rs.pipeline()
        camera_context = rs.context()
        camera_devices = camera_context.query_devices()
        self.diffusion = diffusion
        self.vision_feature_dim = vision_feature_dim
        if len(camera_devices) < 2:
            raise RuntimeError("Two cameras are required, but fewer were detected.")

        serial_A = camera_devices[1].get_info(rs.camera_info.serial_number)
        serial_B = camera_devices[0].get_info(rs.camera_info.serial_number)

        # Configure Camera A
        config_A = rs.config()
        config_A.enable_device(serial_A)
        config_A.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config_A.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        # Configure Camera B
        config_B = rs.config()
        config_B.enable_device(serial_B)
        config_B.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config_B.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        # Start pipelines
        pipeline_A.start(config_A)
        pipeline_B.start(config_B)

        align_A = rs.align(rs.stream.color)
        align_B = rs.align(rs.stream.color)

        # self.env = env
        # self.obs = obs
        # self.info = info
        self.rewards = rewards
        self.device = device
        # self.obs_deque = obs_deque
        # self.imgs = imgs
        self.max_steps = max_steps
        self.ema_nets = ema_nets
        self.step_idx = step_idx
        self.pipeline_A = pipeline_A
        self.pipeline_B = pipeline_B
        self.camera_device = camera_devices
        self.align_A = align_A
        self.align_B = align_B

    def get_observation(self):
        ### Get initial observation for the
        #TODO: Image data from two realsense camera
        pipeline_A = self.pipeline_A
        pipeline_B = self.pipeline_B
        align_A = self.align_A
        align_B = self.align_B
        crop_width, crop_height = 480, 480

        # Create directories if they don't exist
        os.makedirs("images_A", exist_ok=True)
        os.makedirs("images_B", exist_ok=True)

        # Camera intrinsics (dummy values, replace with your actual intrinsics)
        camera_intrinsics = {
            'K': np.array([[640, 0, 320], [0, 640, 240], [0, 0, 1]], dtype=np.float32),
            'dist': np.zeros((5, 1))  # No distortion
        }

        # Camera A pose relative to robot base
        camera_pose_robot_base = [0.47202, 0.150503, 1.24777, 0.00156901, 0.999158, -0.0183132, -0.036689]
        camera_translation = np.array(camera_pose_robot_base[:3])
        camera_rotation = R.from_quat(camera_pose_robot_base[3])
        image_A = None
        image_B = None

        #TODO: Get IIWA pose as [x,y,z, roll, pitch, yaw]

        agent_pos = None




        #####
        frames_A = pipeline_A.wait_for_frames()
        aligned_frames_A = align_A.process(frames_A)
        color_frame_A = aligned_frames_A.get_color_frame()

        frames_B = pipeline_B.wait_for_frames()
        aligned_frames_B = align_B.process(frames_B)
        color_frame_B = aligned_frames_B.get_color_frame()
        color_image_A = np.asanyarray(color_frame_A.get_data())
        color_image_A.astype(np.float32)
        color_image_B = np.asanyarray(color_frame_B.get_data())
        color_image_B.astype(np.float32)
        x_start = 0
        y_start = 0
        cropped_image_A = color_image_A[y_start:y_start + crop_height, x_start:x_start + crop_width]

        center_x = color_image_B.shape[1] // 2
        center_y = color_image_B.shape[0] // 2
        x_start_B = center_x - crop_width // 2
        y_start_B = center_y - crop_height // 2
        x_start_B = max(0, min(x_start_B, color_image_B.shape[1] - crop_width))
        y_start_B = max(0, min(y_start_B, color_image_B.shape[0] - crop_height))
        cropped_image_B = color_image_B[y_start_B:y_start_B + crop_height, x_start_B:x_start_B + crop_width]
        image_A = cropped_image_A
        image_B = cropped_image_B

        return image_A, image_B, agent_pos
        # the final arch has 2 parts
    
    def execute_action(self):
        ### Stepping function to execute action with robot
        #TODO: Execute Motion


        obs = self.get_observation()
        return obs
    
        # the final arch has 2 parts
    ###### Load Pretrained 
    def load_pretrained(self, diffusion):

        load_pretrained = True
        if load_pretrained:
            ckpt_path = "/home/lm-2023/jeon_team_ws/playback_pose/src/Diffusion_Policy_ICRA/checkpoints/checkpoint_250.pth"
            #   ckpt_path = "/home/jeon/jeon_ws/diffusion_policy/src/diffusion_cam/checkpoints/pusht_vision_100ep.ckpt"
            #   if not os.path.isfile(ckpt_path):
            #       id = "1XKpfNSlwYMGaF5CncoFaLKCDTWoLAHf1&confirm=t"
            #       gdown.download(id=id, output=ckpt_path, quiet=False)

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
                    # obs, reward, done, _, info = env.step(action[i])
                    # save observations
                    # obs_deque.append(obs)
                    # and reward/vis
                    # rewards.append(reward)
                    imgs.append(env.render(mode='rgb_array'))

                    # update progress bar
                    step_idx += 1
                    pbar.update(1)
                    # pbar.set_postfix(reward=reward)
                    if step_idx > max_steps:
                        done = True
                    if done:
                        break
        # print out the maximum target coverage

        print('Score: ', max(rewards))
        # return imgs


def main():
    # Max steps will dicate how long the inference duration is going to be so it is very important
    # Initialize RealSense pipelines for both cameras

    max_steps = 300
    # Evaluate Real Robot Environment
    eval_real_robot = EvaluateRealRobot(max_steps)
    eval_real_robot.inference()
    ######## This block is for Visualizing if in virtual environment ###### 
    # height, width, layers = imgs[0].shape
    # video = cv2.VideoWriter('/home/jeon/jeon_ws/diffusion_policy/src/diffusion_cam/vis_real_robot.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (width, height))

    # for img in imgs:
    #     video.write(np.uint8(img))

    # video.release()
    ###########
if __name__ == "__main__":
    main()