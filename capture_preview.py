import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np
import os
import sys

class PreviewCapturer(Node):
    def __init__(self, out_dir="/workspace/提出物/data"):
        super().__init__('preview_capturer')
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.main_msg = None
        self.side_msg = None

        self.create_subscription(Image, '/image_raw', self.cb_main, 10)
        self.create_subscription(Image, '/side_camera/image_raw', self.cb_side, 10)

    def cb_main(self, msg):
        self.main_msg = msg

    def cb_side(self, msg):
        self.side_msg = msg

def main():
    rclpy.init()
    capturer = PreviewCapturer()
    
    # 最大3秒間スピンして両カメラの画像を受信
    for _ in range(30):
        rclpy.spin_once(capturer, timeout_sec=0.1)
        if capturer.main_msg is not None and capturer.side_msg is not None:
            break

    if capturer.main_msg is not None:
        m = capturer.main_msg
        m_img = np.frombuffer(m.data, dtype=np.uint8).reshape((m.height, m.width, 3))
        # Gazebo bridge: rgb8 -> bgr8 for opencv
        cv2.imwrite(os.path.join(capturer.out_dir, "live_preview_main.png"), cv2.cvtColor(m_img, cv2.COLOR_RGB2BGR))

    if capturer.side_msg is not None:
        s = capturer.side_msg
        s_img = np.frombuffer(s.data, dtype=np.uint8).reshape((s.height, s.width, 3))
        cv2.imwrite(os.path.join(capturer.out_dir, "live_preview_side.png"), cv2.cvtColor(s_img, cv2.COLOR_RGB2BGR))

    capturer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
