import sqlite3
import time
import subprocess
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

class TrajectoryReplayer(Node):
    def __init__(self, db_path):
        super().__init__('trajectory_replayer')
        self.arm_pub = self.create_publisher(JointTrajectory, '/crane_plus_arm_controller/joint_trajectory', 10)
        self.gripper_pub = self.create_publisher(JointTrajectory, '/crane_plus_gripper_controller/joint_trajectory', 10)
        self.db_path = db_path

    def load_and_publish(self):
        self.get_logger().info(f"Loading trajectory from {self.db_path}...")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id=2 ORDER BY timestamp ASC")
        rows = cursor.fetchall()
        self.get_logger().info(f"Total raw joint messages: {len(rows)}")

        # Resample trajectory at ~10Hz to ensure smooth execution
        sampled_rows = rows[::10]
        
        # Order of joints in JointState:
        # ['crane_plus_joint1', 'crane_plus_joint2', 'crane_plus_joint4', 'crane_plus_joint3', 'crane_plus_joint_hand']
        arm_names = ['crane_plus_joint1', 'crane_plus_joint2', 'crane_plus_joint3', 'crane_plus_joint4']
        gripper_names = ['crane_plus_joint_hand']

        start_time_ns = sampled_rows[0][0]
        self.get_logger().info("Replaying trajectory in real-time...")

        for i, (ts, raw) in enumerate(sampled_rows):
            msg = deserialize_message(raw, JointState)
            # msg.position: [joint1, joint2, joint4, joint3, hand]
            pos = list(msg.position)
            j1 = pos[0]
            j2 = pos[1]
            j4 = pos[2]
            j3 = pos[3]
            hand = pos[4]

            # Publish Arm command
            arm_traj = JointTrajectory()
            arm_traj.joint_names = arm_names
            pt = JointTrajectoryPoint()
            pt.positions = [j1, j2, j3, j4]
            pt.time_from_start = Duration(sec=0, nanosec=100000000) # 0.1s
            arm_traj.points = [pt]
            self.arm_pub.publish(arm_traj)

            # Publish Gripper command
            grip_traj = JointTrajectory()
            grip_traj.joint_names = gripper_names
            gpt = JointTrajectoryPoint()
            gpt.positions = [hand]
            gpt.time_from_start = Duration(sec=0, nanosec=100000000)
            grip_traj.points = [gpt]
            self.gripper_pub.publish(grip_traj)

            time.sleep(0.1) # 10Hz playback

        self.get_logger().info("Trajectory replay finished successfully!")

def main():
    rclpy.init()
    db_path = "/workspace/提出物/data/raw_bags/episode_20261011_142558/episode_20261011_142558_0.db3"
    replayer = TrajectoryReplayer(db_path)
    replayer.load_and_publish()
    replayer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
