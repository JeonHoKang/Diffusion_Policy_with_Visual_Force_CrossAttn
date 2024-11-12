import sys
import csv
import math
import os
import time
import numpy as np
import pandas as pd
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose
from moveit_msgs.srv import GetPositionFK
from moveit_msgs.msg import MoveItErrorCodes
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped
from submodules.wait_for_message import wait_for_message
from moveit_msgs.srv import GetPositionIK, GetMotionPlan, GetPlanningScene, ApplyPlanningScene
from moveit_msgs.msg import (
    RobotState,
    RobotTrajectory,
    MoveItErrorCodes,
    Constraints,
    JointConstraint,
    PlanningScene,
    CollisionObject
)
import time
import pyrealsense2 as rs
import cv2

def create_directories():
    os.makedirs("/home/lm-2023/jeon_team_ws/playback_pose/src/data_collection/data_collection/images_A", exist_ok=True)
    os.makedirs("/home/lm-2023/jeon_team_ws/playback_pose/src/data_collection/data_collection/images_B", exist_ok=True)

def read_joint_states_from_csv(file_path):
    joint_states = []
    with open(file_path, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            joint_states.append([math.radians(float(row['A1'])), math.radians(float(row['A2'])), math.radians(float(row['A3'])),
                                 math.radians(float(row['A4'])), math.radians(float(row['A5'])), math.radians(float(row['A6'])), math.radians(float(row['A7']))])
    return joint_states


class KukaMotionPlanning(Node):
    timeout_sec_ = 5.0
    move_group_name_ = "arm"
    namespace_ = "lbr"
    joint_state_topic_ = "/lbr/joint_states"
    force_torque_topic_ = "/lbr/force_torque_broadcaster/wrench"
    plan_srv_name_ = "plan_kinematic_path"
    fk_srv_name_ = "lbr/compute_fk"
    execute_action_name_ = "execute_trajectory"
    get_planning_scene_srv_name = "get_planning_scene"

    apply_planning_scene_srv_name = "apply_planning_scene"
    base_ = "link_0"    
    end_effector_ = "link_ee"

    def __init__(self, Transformer = False):
        super().__init__('kuka_motion_planning')
        self.joint_states = None
        self.force_torque_data = None  # Initialize force/torque data container
        self._action_client = ActionClient(self, FollowJointTrajectory, '/lbr/joint_trajectory_controller/follow_joint_trajectory')
        self.joint_names = ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]
        self.joint_trajectories = read_joint_states_from_csv('/home/lm-2023/Downloads/kuka_replay_force_test/2024-10-29_17-53-12.csv')
        self.initialize_robot_pose() 
        create_directories()

        self.fk_client_ = self.create_client(GetPositionFK, self.fk_srv_name_)
        self.force_torque_subscriber = self.create_subscription(WrenchStamped, self.force_torque_topic_, self.force_torque_callback, 10)
        self.send_goal()
        self.feedback = None
        self.Transformer = Transformer

    def force_torque_callback(self, msg):
        self.force_torque_data = msg.wrench

    def joint_state_callback(self, msg):
        with self.lock:
            self.joint_states = msg
            self.joint_positions = msg.position

    def initialize_fk_service(self):
        self.fk_client_ = self.create_client(GetPositionFK, 'compute_fk')
        if not self.fk_client_.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("FK service not available.")
            exit(1)

    def get_fk(self) -> Pose | None:
        current_joint_state = self.get_joint_state()
        if current_joint_state is None:
            self.get_logger().error("Failed to get current joint state!!!")
            return None
        current_robot_state = RobotState()
        current_robot_state.joint_state = current_joint_state
        request = GetPositionFK.Request()
        request.header.frame_id = f"{self.namespace_}/{self.base_}"
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
            self.get_logger().error(f"Failed to get FK solution: {response.error_code.val}")
            return None
        return response.pose_stamped[0].pose

    def convert_joints_to_actions(self, joint_values: list[float]) -> Pose | None:
        # Create a RobotState message and manually assign joint values
        current_robot_state = RobotState()
        current_robot_state.joint_state.name = self.joint_names  # Set joint names appropriately
        current_robot_state.joint_state.position = joint_values  # Set the provided joint values

        # Create the FK request
        request = GetPositionFK.Request()
        request.header.frame_id = f"{self.namespace_}/{self.base_}"
        request.header.stamp = self.get_clock().now().to_msg()
        request.fk_link_names.append(self.end_effector_)  # The end-effector link name
        request.robot_state = current_robot_state  # Assign the robot state with joint values

        # Call the FK service asynchronously
        future = self.fk_client_.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        # Check if the FK service call was successful
        if future.result() is None:
            self.get_logger().error("Failed to get action solution")
            return None
        
        response = future.result()
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(f"Failed to get action solution: {response.error_code.val}")
            return None
        
        # Return the pose of the end-effector
        return response.pose_stamped[0].pose
    
    def get_joint_state(self) -> JointState:        
        current_joint_state_set, current_joint_state = wait_for_message(JointState, self, self.joint_state_topic_)
        if not current_joint_state_set:
            self.get_logger().error("Failed to get current joint state")
            return None
        return current_joint_state

    def record_images(self, image_counter):
        global pipeline_A, pipeline_B, align_A, align_B

        frames_A = pipeline_A.wait_for_frames()
        aligned_frames_A = align_A.process(frames_A)
        color_frame_A = aligned_frames_A.get_color_frame()

        frames_B = pipeline_B.wait_for_frames()
        aligned_frames_B = align_B.process(frames_B)
        color_frame_B = aligned_frames_B.get_color_frame()

        if not color_frame_A or not color_frame_B:
            print("Could not acquire frames from RealSense cameras")
            return

        color_image_A = np.asanyarray(color_frame_A.get_data())
        color_image_B = np.asanyarray(color_frame_B.get_data())
        image_filename_A = f"camera_A_{image_counter}.png"
        color_image_A = cv2.resize(color_image_A, (320, 240), interpolation=cv2.INTER_AREA)
        cv2.imwrite(f"/home/lm-2023/jeon_team_ws/playback_pose/src/data_collection/data_collection/images_A/{image_filename_A}", color_image_A)


        # Get the image dimensions
        height_B, width_B, _ = color_image_B.shape

        # Define the center point
        center_x, center_y = width_B // 2, height_B // 2

        # Define the crop size
        crop_width, crop_height = 320, 240

        # Calculate the top-left corner of the crop box
        x1 = max(center_x - crop_width // 2, 0)
        y1 = max(center_y - crop_height // 2, 0)

        # Calculate the bottom-right corner of the crop box
        x2 = min(center_x + crop_width // 2, width_B)
        y2 = min(center_y + crop_height // 2, height_B)
        cropped_image_B = color_image_B[y1:y2, x1:x2]
        # resized_image_B = cv2.resize(cropped_image_B, (224, 224), interpolation=cv2.INTER_AREA)

        image_filename_B = f"camera_B_{image_counter}.png"
        # color_image_B = cv2.resize(color_image_B, (320, 240), interpolation=cv2.INTER_AREA)
        cv2.imwrite(f"/home/lm-2023/jeon_team_ws/playback_pose/src/data_collection/data_collection/images_B/{image_filename_B}", cropped_image_B)
        complete = os.path.exists(f"/home/lm-2023/jeon_team_ws/playback_pose/src/data_collection/data_collection/images_B/{image_filename_B}")

        while complete == False:
            complete = os.path.exists(f"/home/lm-2023/jeon_team_ws/playback_pose/src/data_collection/data_collection/images_B/{image_filename_B}")
            print(f"Image Saved, {complete}")
        print("Image A and B saved")

    def initialize_robot_pose(self):
        # Create a FollowJointTrajectory goal message
        goal_msg = FollowJointTrajectory.Goal()
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = self.joint_names
        
        self.get_logger().info(f"Processing trajectory point init")

        # Create a JointTrajectoryPoint and assign positions
        point = JointTrajectoryPoint()
        point.positions = self.joint_trajectories[0]
        # point.time_from_start.sec = 1  # Set the time to reach the point (you can modify this)
        point.time_from_start.sec = 8  # Set the seconds part to 0
        # point.time_from_start.nanosec = int(0.5 * 1e9)  # Set the nanoseconds part to 750,000,000

        # Add the point to the trajectory
        trajectory_msg.points.append(point)
        goal_msg.trajectory = trajectory_msg
        self._action_client.wait_for_server()

        # Send the goal asynchronously
        send_goal_future = self._action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)

        rclpy.spin_until_future_complete(self, send_goal_future)

        goal_handle = send_goal_future.result()

        if not goal_handle.accepted:
            self.get_logger().info(f"Goal for init point was rejected")
            return
        
        self.get_logger().info(f"Goal for init point was accepted")
        # Wait for the result to complete before moving to the next trajectory point
        get_result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, get_result_future)
        result = get_result_future.result().result
        self.get_logger().info(f"Result : {result}, Initial position reached")
        input()


    def send_goal(self):
        # Iterate through each joint trajectory point one by one
        for i, joint_values in enumerate(self.joint_trajectories):
            # Create a FollowJointTrajectory goal message
            goal_msg = FollowJointTrajectory.Goal()
            trajectory_msg = JointTrajectory()
            trajectory_msg.joint_names = self.joint_names
            
            self.get_logger().info(f"Processing trajectory point {i}")

            # Create a JointTrajectoryPoint and assign positions
            point = JointTrajectoryPoint()
            point.positions = joint_values
            # point.time_from_start.sec = 1  # Set the time to reach the point (you can modify this)
            point.time_from_start.sec = 0  # Set the seconds part to 0
            point.time_from_start.nanosec = int(0.5 * 1e9)  # Set the nanoseconds part to 750,000,000

            # Add the point to the trajectory
            trajectory_msg.points.append(point)
            goal_msg.trajectory = trajectory_msg
            self._action_client.wait_for_server()

            # Prerecord the states and action pair   
            fk_time = time.time()
            robot_pose = self.get_fk()
            fk_end_time = time.time()
            fk_duration = fk_end_time-fk_time
            if robot_pose is not None:
                robot_state = [robot_pose.position.x, robot_pose.position.y, robot_pose.position.z,
                                robot_pose.orientation.x, robot_pose.orientation.y, robot_pose.orientation.z, robot_pose.orientation.w]
            else:
                robot_pose_data = [None] * 7  # No valid robot pose available
            print(f"fk, {robot_state}")
            action_pose = self.convert_joints_to_actions(joint_values)
            print(action_pose)
            if action_pose is not None:
                robot_action = [action_pose.position.x, action_pose.position.y, action_pose.position.z,
                                action_pose.orientation.x, action_pose.orientation.y, action_pose.orientation.z, action_pose.orientation.w]
            else:
                robot_action = [None] * 7  # No valid robot pose available
            # Get force/torque data
            if self.force_torque_data is not None:
                force_torque_data = [self.force_torque_data.force.x, self.force_torque_data.force.y, self.force_torque_data.force.z,self.force_torque_data.torque.x, self.force_torque_data.torque.y, self.force_torque_data.torque.z]
            else:
                force_torque_data = [None] * 6  # No valid force/torque data available
            print(f"fk_duration, {fk_duration}")
            print(self.force_torque_data)
            self.record_images(i)
            self.write_csv(robot_state, robot_action, force_torque_data)

            # Wait for the action server to be available
            curr_time = time.time()

            # Send the goal asynchronously
            send_goal_future = self._action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
            after_exec = time.time()
            print(f"execution time, {after_exec - curr_time}")

            rclpy.spin_until_future_complete(self, send_goal_future)

            goal_handle = send_goal_future.result()

            if not goal_handle.accepted:
                self.get_logger().info(f"Goal for point {i} was rejected")
                return
            
            self.get_logger().info(f"Goal for point {i} was accepted")
            # Wait for the result to complete before moving to the next trajectory point
            get_result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, get_result_future)
            result = get_result_future.result().result

            self.get_logger().info(f"Result for point {i}: {result}")

            # Introduce a delay if necessary before moving to the next point
            # time.sleep(0.5)  # Optional: Modify the sleep time if required

        self.get_logger().info('All joint trajectories have been processed.')
    
    def write_csv(self, robot_state_data, robot_action_data, force_torque_data):
        timestamp = time.time()
        # Append the new data to the CSV file
        csv_filename = '/home/lm-2023/jeon_team_ws/playback_pose/src/data_collection/data_collection/robot_poses1.csv'
        new_data = [robot_state_data + robot_action_data + force_torque_data + [timestamp]]

        # Check if the file exists to determine if headers are needed
        file_exists = os.path.isfile(csv_filename)

        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                # Write the header if the file doesn't exist
                writer.writerow(['st_robot_x', 'st_robot_y', 'st_robot_z', 'st_robot_qx', 'st_robot_qy', 'st_robot_qz', 'st_robot_qw','ac_robot_x', 'ac_robot_y', 'ac_robot_z', 'ac_robot_qx', 'ac_robot_qy', 'ac_robot_qz', 'ac_robot_qw',
                                 'force_x', 'force_y', 'force_z','torque_x', 'torque_y', 'torque_z', 'timestamp'])
            # Write the data
            writer.writerows(new_data)
    

    # def goal_response_callback(self, future):
    #     goal_handle = future.result()
    #     if not goal_handle.accepted:
    #         self.get_logger().info('Goal rejected :(')
    #         return
    #     self.get_logger().info('Goal accepted :)')
    #     self._get_result_future = goal_handle.get_result_async()
    #     self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        self.feedback = feedback_msg.feedback

    # def get_result_callback(self, future):
    #     result = future.result().result
    #     self.get_logger().info(f'Result: {result}')


pipeline_A = None
pipeline_B = None
align_A = None
align_B = None



def initialize_cameras():
    global pipeline_A, pipeline_B, align_A, align_B

    pipeline_A = rs.pipeline()
    pipeline_B = rs.pipeline()
    
    context = rs.context()
    devices = context.query_devices()
    
    if len(devices) < 2:
        raise RuntimeError("Two cameras are required, but fewer were detected.")
    
    serial_A = devices[1].get_info(rs.camera_info.serial_number)
    serial_B = devices[0].get_info(rs.camera_info.serial_number)
    
    config_A = rs.config()
    config_A.enable_device(serial_A)
    config_A.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    config_B = rs.config()
    config_B.enable_device(serial_B)
    config_B.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    pipeline_A.start(config_A)
    pipeline_B.start(config_B)
    
    align_A = rs.align(rs.stream.color)
    align_B = rs.align(rs.stream.color)
# Initialize once


def main(args=None):
    try:
        rclpy.init(args=args)
    except Exception as e:
        print(f"Failed to initialize rclpy: {e}")
        return

    try:
        initialize_cameras()
        node = KukaMotionPlanning(Transformer = True)
        rclpy.spin(node)
    except Exception as e:
        print(f"Exception during ROS2 node operation: {e}")

if __name__ == '__main__':
    main()
