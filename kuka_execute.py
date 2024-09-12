import sys
import csv
import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import WrenchStamped
import time


class KukaMotionPlanning(Node):
    def __init__(self, current_step):
        super().__init__('kuka_motion_planning')
        self._action_client = ActionClient(self, FollowJointTrajectory, '/lbr/joint_trajectory_controller/follow_joint_trajectory')
        self.force_torque_data = None  # Initialize force/torque data container
        self.current_step = current_step
        self.joint_names = ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]


    def send_goal(self, joint_trajectories):
        goal_msg = FollowJointTrajectory.Goal()
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = list(joint_trajectories.position)
        if self.current_step < 1 :
            point.time_from_start.sec = 2  # 3 seconds for the first point
        else:
            point.time_from_start.sec = 0
            point.time_from_start.nanosec = int(0.25 * 1e9)
        trajectory_msg.points.append(point)
    
        goal_msg.trajectory = trajectory_msg
        
        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        self._send_goal_future.add_done_callback(self.goal_response_callback)
        rclpy.spin_until_future_complete(self, self._send_goal_future)
        time.sleep(0.5)


    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return
        self.get_logger().info('Goal accepted :)')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Feedback: {feedback}')

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: {result}')
        # rclpy.shutdown()