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
from moveit_msgs.srv import GetPositionFK, GetPositionIK
from moveit_msgs.msg import RobotState, MoveItErrorCodes
from geometry_msgs.msg import Pose
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
# from data_collection.submodules.wait_for_message import wait_for_message
import matplotlib.pyplot as plt
#@markdown ### **Loading Pretrained Checkpoint**
#@markdown Set `load_pretrained = True` to load pretrained weights.
from typing import Union
import json
from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy
from rclpy.qos import QoSProfile
from rclpy.signals import SignalHandlerGuardCondition
from rclpy.utilities import timeout_sec_to_nsec
from kuka_execute import KukaMotionPlanning
import cv2

def wait_for_message(
    msg_type,
    node: 'Node',
    topic: str,
    *,
    qos_profile: Union[QoSProfile, int] = 1,
    time_to_wait=-1
):
    """
    Wait for the next incoming message.

    :param msg_type: message type
    :param node: node to initialize the subscription on
    :param topic: topic name to wait for message
    :param qos_profile: QoS profile to use for the subscription
    :param time_to_wait: seconds to wait before returning
    :returns: (True, msg) if a message was successfully received, (False, None) if message
        could not be obtained or shutdown was triggered asynchronously on the context.
    """
    context = node.context
    wait_set = _rclpy.WaitSet(1, 1, 0, 0, 0, 0, context.handle)
    wait_set.clear_entities()

    sub = node.create_subscription(msg_type, topic, lambda _: None, qos_profile=qos_profile)
    try:
        wait_set.add_subscription(sub.handle)
        sigint_gc = SignalHandlerGuardCondition(context=context)
        wait_set.add_guard_condition(sigint_gc.handle)

        timeout_nsec = timeout_sec_to_nsec(time_to_wait)
        wait_set.wait(timeout_nsec)

        subs_ready = wait_set.get_ready_entities('subscription')
        guards_ready = wait_set.get_ready_entities('guard_condition')

        if guards_ready:
            if sigint_gc.handle.pointer in guards_ready:
                return False, None

        if subs_ready:
            if sub.handle.pointer in subs_ready:
                msg_info = sub.handle.take_message(sub.msg_type, sub.raw)
                if msg_info is not None:
                    return True, msg_info[0]
    finally:
        node.destroy_subscription(sub)

    return False, None

#@markdown ### **Network Demo**
class EndEffectorPoseNode(Node):
    timeout_sec_ = 5.0
    joint_state_topic_ = "lbr/joint_states"
    fk_srv_name_ = "lbr/compute_fk"
    ik_srv_name_ = "lbr/compute_ik"
    base_ = "lbr/link_0"
    end_effector_ = "link_ee"


    def __init__(self, node_id: str) -> None:
        super().__init__(f"end_effector_pose_node_{node_id}")

        self.fk_client_ = self.create_client(GetPositionFK, self.fk_srv_name_)
        if not self.fk_client_.wait_for_service(timeout_sec=self.timeout_sec_):
            self.get_logger().error("FK service not available.")
            exit(1)
        self.ik_client_ = self.create_client(GetPositionIK, self.ik_srv_name_)
        if not self.ik_client_.wait_for_service(timeout_sec=self.timeout_sec_):
            self.get_logger().error("IK service not available.")
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
        
        pose = response.pose_stamped[0].pose
        position = [pose.position.x, pose.position.y, pose.position.z]
        quaternion = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]

        # Convert quaternion to roll, pitch, yaw
        rotation = R.from_quat(quaternion)
        rpy = rotation.as_euler('xyz', degrees=False).tolist()
        return position + rpy
    
    def get_ik(self, target_pose: Pose) -> JointState | None:
        request = GetPositionIK.Request()
    
        request.ik_request.group_name = "arm"
        # tf_prefix = self.get_namespace()[1:]
        request.ik_request.pose_stamped.header.frame_id = f"{self.base_}"
        request.ik_request.pose_stamped.header.stamp = self.get_clock().now().to_msg()
        request.ik_request.pose_stamped.pose = target_pose
        request.ik_request.avoid_collisions = True

        future = self.ik_client_.call_async(request)

        rclpy.spin_until_future_complete(self, future)
        if future.result() is None:
            self.get_logger().error("Failed to get IK solution")
            return None

        response = future.result()
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            return None

        return response.solution.joint_state

class EvaluateRealRobot:
    # construct ResNet18 encoder
    # if you have multiple camera views, use seperate encoder weights for each view.
    def __init__(self, max_steps):

        diffusion = DiffusionPolicy_Real(train=False)
        # num_epochs = 100
        ema_nets = self.load_pretrained(diffusion)
        # ResNet18 has output dim of 512
        vision_feature_dim = 1024
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

        # save visualization and rewards
        # imgs = [env.render(mode='rgb_array')]

        # rewards = list()
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


        align_A = rs.align(rs.stream.color)
        align_B = rs.align(rs.stream.color)


        # self.env = env
        # self.obs = obs
        # self.info = info
        # self.rewards = rewards
        self.device = device
        # self.imgs = imgs
        self.max_steps = max_steps
        self.ema_nets = ema_nets
        self.step_idx = step_idx
        self.pipeline_A = pipeline_A
        self.pipeline_B = pipeline_B
        self.camera_device = camera_devices
        self.align_A = align_A
        self.align_B = align_B


        self.pipeline_A.start(config_A)
        self.pipeline_B.start(config_B)

        obs = self.get_observation()
         # keep a queue of last 2 steps of observations
        obs_deque = collections.deque(
            [obs] * diffusion.obs_horizon, maxlen=diffusion.obs_horizon)

        self.obs_deque = obs_deque

    def get_observation(self):
        ### Get initial observation for the
        EE_Pose_Node = EndEffectorPoseNode("obs")
        obs = {}
        #TODO: Image data from two realsense camera
        pipeline_A = self.pipeline_A
        pipeline_B = self.pipeline_B
        align_A = self.align_A
        align_B = self.align_B

        # Create directories if they don't exist
        os.makedirs("images_A", exist_ok=True)
        os.makedirs("images_B", exist_ok=True)

        # Camera intrinsics (dummy values, replace with your actual intrinsics)
        camera_intrinsics = {
            'K': np.array([[640, 0, 320], [0, 640, 240], [0, 0, 1]], dtype=np.float32),
            'dist': np.zeros((5, 1))  # No distortion
        }
        
        # Camera A pose relative to robot base
        # camera_pose_robot_base = [0.47202, 0.150503, 1.24777, 0.00156901, 0.999158, -0.0183132, -0.036689]
        # camera_translation = np.array(camera_pose_robot_base[:3])
        # camera_rotation = R.from_quat(camera_pose_robot_base[3])
        image_A = None
        image_B = None

        #TODO: Get IIWA pose as [x,y,z, roll, pitch, yaw]
        agent_pos = EE_Pose_Node.get_fk()
        agent_pos = np.array(agent_pos)
        agent_pos.astype(np.float64)


        if agent_pos is None:
            EE_Pose_Node.get_logger().error("Failed to get end effector pose")
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

        image_A = cv2.resize(color_image_A, (360, 240), interpolation=cv2.INTER_AREA)
        image_B = cv2.resize(color_image_B, (360, 240), interpolation=cv2.INTER_AREA)
        # Convert BGR to RGB for Matplotlib visualization
        image_A_rgb = cv2.cvtColor(image_A, cv2.COLOR_BGR2RGB)
        image_B_rgb = cv2.cvtColor(image_B, cv2.COLOR_BGR2RGB)
         ### Visualizing purposes
        # import matplotlib.pyplot as plt
        # plt.imshow(image_A_rgb)
        # plt.show()
        # plt.imshow(image_B_rgb)
        # plt.show()
        print(f'current agent position, {agent_pos}')
        # Reshape to (C, H, W)
        image_A = np.transpose(image_A_rgb, (2, 0, 1))
        image_B = np.transpose(image_B_rgb, (2, 0, 1))
        obs['image_A'] = image_A
        obs['image_B'] = image_B
        obs['agent_pos'] = agent_pos
        EE_Pose_Node.destroy_node()

        return obs
    
    def execute_action(self, end_effector_pose_rpy):
        ### Stepping function to execute action with robot
        #TODO: Execute Motion
        EE_Pose_Node = EndEffectorPoseNode("exec")
        end_effector_pose_rpy = [float(value) for value in end_effector_pose_rpy]
        position = end_effector_pose_rpy[:3]
        rpy = end_effector_pose_rpy[3:]
        rotation = R.from_euler('xyz', rpy, degrees=False)
        quaternion = rotation.as_quat()
        print(f'action command {end_effector_pose_rpy}')
        # Create Pose message for IK
        target_pose = Pose()
        target_pose.position.x = position[0]
        target_pose.position.y = position[1]
        target_pose.position.z = position[2]
        target_pose.orientation.x = quaternion[0]
        target_pose.orientation.y = quaternion[1]
        target_pose.orientation.z = quaternion[2]
        target_pose.orientation.w = quaternion[3]

        # Get IK solution
        joint_state = EE_Pose_Node.get_ik(target_pose)
        if joint_state is None:
            EE_Pose_Node.get_logger().error("Failed to get IK solution")
            return
        
        # # Create a JointTrajectory message
        # goal_msg = FollowJointTrajectory.Goal()
        # trajectory_msg = JointTrajectory()
        kuka_execution = KukaMotionPlanning()
        kuka_execution.send_goal(joint_state)

        # # trajectory_msg.joint_names = kuka_execution.joint_names
        # point = JointTrajectoryPoint()
        # point.positions = joint_state.position
        # point.time_from_start.sec = 1  # Set the duration for the motion
        # trajectory_msg.points.append(point)
        
        # goal_msg.trajectory = trajectory_msg
        # kuka_execution.send_goal(trajectory_msg)

        # # Send the trajectory to the action server
        # kuka_execution._action_client.wait_for_server()
        # kuka_execution._send_goal_future = kuka_execution._action_client.send_goal_async(goal_msg, feedback_callback=kuka_execution.feedback_callback)
        # kuka_execution._send_goal_future.add_done_callback(kuka_execution.goal_response_callback)
        EE_Pose_Node.destroy_node()
        kuka_execution.destroy_node()
        # construct new observation
        obs = self.get_observation()
        return obs
    

        # the final arch has 2 parts
    ###### Load Pretrained 
    def load_pretrained(self, diffusion):

        load_pretrained = True
        if load_pretrained:
            ckpt_path = "/home/jeon/jeon_ws/diffusion_policy/src/diffusion_cam/checkpoints/checkpoint_250.pth"
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
        ema_nets = self.ema_nets
        # rewards = self.rewards
        step_idx = self.step_idx
        done = False


        with open('/home/lm-2023/jeon_team_ws/playback_pose/src/Diffusion_Policy_ICRA/stats.json', 'r') as f:
            stats = json.load(f)
            # Convert stats['agent_pos']['min'] and ['max'] to numpy arrays with float32 type
            stats['agent_pos']['min'] = np.array(stats['agent_pos']['min'], dtype=np.float32)
            stats['agent_pos']['max'] = np.array(stats['agent_pos']['max'], dtype=np.float32)

            # Convert stats['action']['min'] and ['max'] to numpy arrays with float32 type
            stats['action']['min'] = np.array(stats['action']['min'], dtype=np.float32)
            stats['action']['max'] = np.array(stats['action']['max'], dtype=np.float32)

        with tqdm(total=max_steps, desc="Eval Real Robot") as pbar:
            while not done:
                B = 1
                # stack the last obs_horizon number of observations
                images_A = np.stack([x['image_A'] for x in obs_deque])
                images_B = np.stack([x['image_B'] for x in obs_deque])
                agent_poses = np.stack([x['agent_pos'] for x in obs_deque])

                # normalize observation
                nagent_poses = data_utils.normalize_data(agent_poses, stats=stats['agent_pos'])
                # images are already normalized to [0,1]
                nimages = images_A
                nimages_second_view = images_B
                # device transfer
                nimages = torch.from_numpy(nimages).to(device, dtype=torch.float32)
                nimages_second_view = torch.from_numpy(nimages_second_view).to(device, dtype=torch.float32)

                # (2,3,480,480)
                nagent_poses = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32)

                # infer action
                with torch.no_grad():
                    # get image features
                    image_features = ema_nets['vision_encoder'](nimages)
                    # (2,512)
                    image_features_second_view = ema_nets['vision_encoder2'](nimages_second_view)

                    # concat with low-dim observations
                    obs_features = torch.cat([image_features, image_features_second_view, nagent_poses], dim=-1)

                    # reshape observation to (B,obs_horizon*obs_dim)
                    obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

                    # initialize action from Guassian noise
                    noisy_action = torch.randn(
                        (B, diffusion.pred_horizon, diffusion.action_dim), device=device)
                    naction = noisy_action
                    diffusion_inference_iteration = 20
                    # init scheduler
                    diffusion.noise_scheduler.set_timesteps(diffusion_inference_iteration)
                    denoising_steps_inference  = 20

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
                action_pred = data_utils.unnormalize_data(naction, stats=stats['action'])

                # only take action_horizon number of actions5
                start = diffusion.obs_horizon - 1
                end = start + diffusion.action_horizon
                action = action_pred[start:end,:]
            # (action_horizon, action_dim)
    
                # execute action_horizon number of steps
                # without replanning
                for i in range(len(action)):
                    # stepping env
                    obs = self.execute_action(action[i])
                    # save observations
                    obs_deque.append(obs)
                    # and reward/vis
                    # rewards.append(reward)
                    # imgs.append(env.render(mode='rgb_array'))

                    # update progress bar
                    step_idx += 1
                    pbar.update(1)
                    # pbar.set_postfix(reward=reward)
                    if step_idx > max_steps:
                        done = True
                    if done:
                        break
        # print out the maximum target coverage

        # print('Score: ', max(rewards))
        # return imgs


def main():
    # Max steps will dicate how long the inference duration is going to be so it is very important
    # Initialize RealSense pipelines for both cameras
    rclpy.init()
    try:  
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
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Ensure shutdown is called even if an error occurs
        rclpy.shutdown()
if __name__ == "__main__":
    main()