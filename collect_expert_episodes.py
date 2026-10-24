"""
collect_expert_episodes.py
=====================================================================
Step 6-1: 実Gazebo環境における自律エキスパート実演データ収集エンジン
- 作業空間（通常サイド5配置、逆サイド5配置の計10配置）を順次巡回
- 各配置でペグを自動スポーンし、Stage 5Bの完全自律エキスパート則を実行
- メインカメラ画像(224x224 RGB)、実関節角度(5軸)、14次元アフォーダンス特徴量、
  目標関節指令を10Hzで完全同期記録
- 1エピソード終了ごとにデータをNPZ保存し、ペグを自動デスポーンして次配置へ遷移
- 全エピソード完了後、作業空間多様化分布図と実測関節軌道群グラフを自動生成
=====================================================================
"""

import os
import sys
import time
import subprocess
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

# ROS 2 & OpenCV (optional when running --plot-only on host)
try:
    import cv2
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, JointState
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from builtin_interfaces.msg import Duration
    from visual_servo_peg_in_hole import ObjectCentricDetector, solve_crane_ik
except ImportError:
    cv2 = None
    Node = object

# -------------------------------------------------------------
# 1. 作業空間内の代表ペグ配置 (10エピソード)
# -------------------------------------------------------------
PEG_CONFIGS = [
    # 逆サイド (Y < 0): 直立化ヨー回転 (+5.44°) が必須な検証配置 (近距離 R=0.139m 〜 遠距離 R=0.198m まで広く網羅)
    {"id": 0, "name": "Reverse_Center", "x": 0.150, "y": -0.080, "desc": "標準逆サイド (実証済み・中距離)"},
    {"id": 1, "name": "Reverse_Near",   "x": 0.125, "y": -0.060, "desc": "逆サイド・近距離 (R=0.139m)"},
    {"id": 2, "name": "Reverse_Far",    "x": 0.185, "y": -0.070, "desc": "逆サイド・遠距離 (R=0.198m)"},
    {"id": 3, "name": "Reverse_Left",   "x": 0.135, "y": -0.085, "desc": "逆サイド・左手前深角 (R=0.160m)"},
    {"id": 4, "name": "Reverse_Right",  "x": 0.170, "y": -0.045, "desc": "逆サイド・やや奥浅角 (R=0.176m)"},
    # 通常サイド (Y > 0): 固定爪側支持による対称配置 (近距離 R=0.139m 〜 遠距離 R=0.198m まで広く網羅)
    {"id": 5, "name": "Normal_Center",  "x": 0.150, "y":  0.080, "desc": "標準通常サイド (中距離)"},
    {"id": 6, "name": "Normal_Near",    "x": 0.125, "y":  0.060, "desc": "通常サイド・近距離 (R=0.139m)"},
    {"id": 7, "name": "Normal_Far",     "x": 0.185, "y":  0.070, "desc": "通常サイド・遠距離 (R=0.198m)"},
    {"id": 8, "name": "Normal_Left",    "x": 0.135, "y":  0.085, "desc": "通常サイド・左手前深角 (R=0.160m)"},
    {"id": 9, "name": "Normal_Right",   "x": 0.170, "y":  0.045, "desc": "通常サイド・やや奥浅角 (R=0.176m)"},
]

# -------------------------------------------------------------
# 2. Gazeboエンティティ管理ヘルパー
# -------------------------------------------------------------
def remove_peg_from_gazebo():
    """Ignition Gazebo から既存ペグを削除"""
    cmd = (
        'ign service -s /world/default/remove --reqtype ignition.msgs.Entity '
        '--reptype ignition.msgs.Boolean --timeout 2000 --req \'name: "peg", type: MODEL\''
    )
    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.5)

def spawn_peg_in_gazebo(x, y, z=1.03):
    """Ignition Gazebo に指定座標でペグをスポーン"""
    peg_sdf = "/workspace/提出物/models/peg/model.sdf"
    cmd = (
        f'ros2 run ros_gz_sim create -name peg -file {peg_sdf} -x {x:.3f} -y {y:.3f} -z {z:.3f}'
    )
    res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(2.5)
    return "OK creation" in res.stdout or res.returncode == 0


def reset_arm_to_home():
    """エピソード開始前にアームとグリッパーを初期姿勢へ戻して静定させる"""
    arm_cmd = (
        'ros2 topic pub --once /crane_plus_arm_controller/joint_trajectory '
        'trajectory_msgs/msg/JointTrajectory '
        '"{joint_names: [\'crane_plus_joint1\', \'crane_plus_joint2\', \'crane_plus_joint3\', \'crane_plus_joint4\'], '
        'points: [{positions: [0.0, 0.0, 0.0, 0.0], time_from_start: {sec: 1, nanosec: 0}}]}"'
    )
    grip_cmd = (
        'ros2 topic pub --once /crane_plus_gripper_controller/joint_trajectory '
        'trajectory_msgs/msg/JointTrajectory '
        '"{joint_names: [\'crane_plus_joint_hand\'], '
        'points: [{positions: [-0.15], time_from_start: {sec: 1, nanosec: 0}}]}"'
    )
    subprocess.run(arm_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(grip_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2.0)

# -------------------------------------------------------------
# 3. エキスパート実演データ収集ROS2ノード
# -------------------------------------------------------------
class ExpertRolloutCollectorNode(Node):
    def __init__(self, target_config, max_steps=285):
        super().__init__('expert_rollout_collector')
        self.config = target_config
        self.max_steps = max_steps
        
        self.detector = ObjectCentricDetector(target_peg_color="red")
        
        # 幾何学的オフセット定数 (Stage 5B 実証値)
        self.d_claw_contact = 0.02195  # [m]
        self.d_claw_bottom = 0.0400    # [m]
        
        # ターゲット座標
        self.target_peg_x = target_config["x"]
        self.target_peg_y = target_config["y"]
        self.hole_x = 0.200
        self.hole_y = 0.000
        
        # 極座標と幾何ロック
        r_p_raw = np.sqrt(self.target_peg_x**2 + self.target_peg_y**2)
        th_p_raw = np.arctan2(self.target_peg_y, self.target_peg_x)
        d_th_p = np.arcsin(np.clip(self.d_claw_contact / max(0.05, r_p_raw), -1.0, 1.0))
        self.locked_yaw_peg = th_p_raw - d_th_p
        self.locked_r_peg = np.sqrt(max(0.001, r_p_raw**2 - self.d_claw_contact**2))
        
        r_h_raw = np.sqrt(self.hole_x**2 + self.hole_y**2)
        th_h_raw = np.arctan2(self.hole_y, self.hole_x)
        d_th_h = np.arcsin(np.clip(self.d_claw_bottom / max(0.05, r_h_raw), -1.0, 1.0))
        self.locked_yaw_hole = th_h_raw - d_th_h
        self.locked_r_hole = np.sqrt(max(0.001, r_h_raw**2 - self.d_claw_bottom**2))
        
        # 左右爪非対称（13mm長差・傾斜爪）による把持傾きをY<0 / Y>=0 全域で直立化するヨー回転角 (+5.44°)
        self.is_reverse = (self.target_peg_y < 0.0)
        self.yaw_upright_offset = 0.095
        
        # 姿勢レジスタ (穴ボード上空も安全にクリアする高空ホバリング・リフト)
        self.j_pick = list(solve_crane_ik(self.locked_r_peg, z_target=0.062, pitch_deg=135.0))
        self.j_pick_hov = list(solve_crane_ik(self.locked_r_peg, z_target=0.115, pitch_deg=135.0))
        self.j_pick_lift = list(solve_crane_ik(self.locked_r_peg, z_target=0.130, pitch_deg=135.0))
        
        self.j_bank_hov = [0.365, -0.200, -1.750]
        self.j_bank_touch = [0.540, -0.200, -1.750]
        # 土台接触後の再上昇を1/3 (ΔJ2=0.040rad, 約6.7mm浮上) に抑え、再浮上と穴進入に伴うペグの振幅増大を防止
        self.j_bank_lift = [0.500, -0.200, -1.750]
        self.j_hole_hov = [0.570, -0.055, -1.680]
        self.j_hole_ins = [0.715, -0.110, -1.720]
        
        # 通信設定
        self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.create_subscription(Image, '/side_camera/image_raw', self.side_image_callback, 10)
        self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        
        self.arm_pub = self.create_publisher(JointTrajectory, '/crane_plus_arm_controller/joint_trajectory', 10)
        self.gripper_pub = self.create_publisher(JointTrajectory, '/crane_plus_gripper_controller/joint_trajectory', 10)
        
        self.arm_names = ['crane_plus_joint1', 'crane_plus_joint2', 'crane_plus_joint3', 'crane_plus_joint4']
        self.grip_names = ['crane_plus_joint_hand']
        
        # センサーバッファ
        self.latest_image_bgr = None
        self.latest_side_image_bgr = None
        self.latest_joints = None
        
        # 連続目標レジスタ (アプローチ時のグリッパー開度はペグ直径1.5倍の0.00 rad)
        self.current_cmd_arm = [0.0, 0.0, 0.0, 0.0]
        self.current_cmd_grip = 0.00
        self.is_held = 0.0
        self.current_phase = "INIT"
        self.phase_metric = 0.10
        
        # データ記録バッファ
        self.recorded_images = []
        self.recorded_joints = []
        self.recorded_cmds = []
        self.recorded_features = []
        self.recorded_phases = []
        
        self.step_count = 0
        self.timer_period = 0.12  # Step 5B と完全同期 (約8.3 Hz)
        self.timer = self.create_timer(self.timer_period, self.control_loop)
        self.is_done = False

    def image_callback(self, msg):
        img_rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        self.latest_image_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    def side_image_callback(self, msg):
        img_rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        self.latest_side_image_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    def joint_callback(self, msg):
        name_to_pos = dict(zip(msg.name, msg.position))
        j1 = name_to_pos.get('crane_plus_joint1', 0.0)
        j2 = name_to_pos.get('crane_plus_joint2', 0.0)
        j3 = name_to_pos.get('crane_plus_joint3', 0.0)
        j4 = name_to_pos.get('crane_plus_joint4', 0.0)
        hand = name_to_pos.get('crane_plus_joint_hand', 0.0)
        self.latest_joints = [j1, j2, j3, j4, hand]
        if self.step_count == 0:
            self.current_cmd_arm = [j1, j2, j3, j4]

    def compute_14dim_features(self, progress_tau):
        """14次元アフォーダンス特徴量ベクトルの計算"""
        r_peg = np.sqrt(self.target_peg_x**2 + self.target_peg_y**2)
        th_peg = np.arctan2(self.target_peg_y, self.target_peg_x)
        dx = self.hole_x - self.target_peg_x
        dy = self.hole_y - self.target_peg_y
        dist_peg = r_peg
        dist_hole = np.sqrt(self.hole_x**2 + self.hole_y**2)
        side_lift = 1.0 if self.is_held > 0.5 else 0.0
        
        feat = np.array([
            r_peg, th_peg,
            self.target_peg_x, self.target_peg_y,
            self.hole_x, self.hole_y,
            dx, dy,
            dist_peg, dist_hole,
            progress_tau,
            self.phase_metric,
            self.is_held,
            side_lift
        ], dtype=np.float32)
        return feat

    def control_loop(self):
        if self.latest_image_bgr is None or self.latest_joints is None:
            return
        
        self.step_count += 1
        if self.step_count > self.max_steps:
            self.is_done = True
            self.timer.cancel()
            return

        progress_tau = (self.step_count - 1) / float(self.max_steps - 1)
        
        # -------------------------------------------------------------
        # 5大物理動作フェーズ＆指令生成 (Stage 5B 完全同期)
        # -------------------------------------------------------------
        # パターン 1: 取りに行く (Approach, Hover & Descend) [Step 1〜35]
        # アプローチ開度はペグ直径の1.5倍 (0.00 rad, 幅約45mm) に設定し穴土台との接触を根絶
        if self.step_count <= 10:
            self.current_phase = "1_ALIGN_YAW_HOVER"
            self.phase_metric = 0.10
            target_arm = [self.locked_yaw_peg, self.j_pick_hov[0], self.j_pick_hov[1], self.j_pick_hov[2]]
            target_grip = 0.00
            self.is_held = 0.0
        elif self.step_count <= 35:
            self.current_phase = "2_VERTICAL_DESCENT"
            self.phase_metric = 0.10
            t = min(1.0, (self.step_count - 10) / 25.0)
            j2 = (1 - t) * self.j_pick_hov[0] + t * self.j_pick[0]
            j3 = (1 - t) * self.j_pick_hov[1] + t * self.j_pick[1]
            j4 = (1 - t) * self.j_pick_hov[2] + t * self.j_pick[2]
            target_arm = [self.locked_yaw_peg, j2, j3, j4]
            target_grip = 0.00
            self.is_held = 0.0
            
        # パターン 2: 挟み込む (Stepwise Grasp) [Step 36〜55]
        elif self.step_count <= 55:
            self.current_phase = "3_STEPWISE_GRASP"
            self.phase_metric = 0.30
            target_arm = [self.locked_yaw_peg, self.j_pick[0], self.j_pick[1], self.j_pick[2]]
            t_g = min(1.0, (self.step_count - 35) / 15.0)
            target_grip = 0.00 + t_g * (0.42 - 0.00)
            self.is_held = float(t_g)
            
        # パターン 3: 持ち上げる (Vertical Helicopter Lift) [Step 56〜95]
        elif self.step_count <= 95:
            self.current_phase = "4_HELICOPTER_CLIMB"
            self.phase_metric = 0.50
            t_lift = min(1.0, (self.step_count - 55) / 35.0)
            s_lift = 0.5 * (1.0 - np.cos(np.pi * t_lift))
            j2 = (1 - s_lift) * self.j_pick[0] + s_lift * self.j_pick_lift[0]
            j3 = (1 - s_lift) * self.j_pick[1] + s_lift * self.j_pick_lift[1]
            j4 = (1 - s_lift) * self.j_pick[2] + s_lift * self.j_pick_lift[2]
            target_arm = [self.locked_yaw_peg, j2, j3, j4]
            target_grip = 0.42
            self.is_held = 1.0
            
        # パターン 4: 置きにいく (Cruise, Bank Touchdown & Align) [Step 96〜210]
        elif self.step_count <= 135:
            self.current_phase = "5_CRUISE_TO_BANK"
            self.phase_metric = 0.75
            t_trans = min(1.0, (self.step_count - 95) / 38.0)
            s_trans = 0.5 * (1.0 - np.cos(np.pi * t_trans))
            cur_yaw = (1 - s_trans) * self.locked_yaw_peg + s_trans * self.locked_yaw_hole
            cur_j2 = (1 - s_trans) * self.j_pick_lift[0] + s_trans * self.j_bank_hov[0]
            cur_j3 = (1 - s_trans) * self.j_pick_lift[1] + s_trans * self.j_bank_hov[1]
            cur_j4 = (1 - s_trans) * self.j_pick_lift[2] + s_trans * self.j_bank_hov[2]
            target_arm = [cur_yaw, cur_j2, cur_j3, cur_j4]
            target_grip = 0.42
            self.is_held = 1.0
        elif self.step_count <= 165:
            self.current_phase = "6A_BANK_DAMPING_TOUCH"
            self.phase_metric = 0.75
            t_touch = min(1.0, (self.step_count - 135) / 25.0)
            s_touch = 0.5 * (1.0 - np.cos(np.pi * t_touch))
            cur_j2 = (1 - s_touch) * self.j_bank_hov[0] + s_touch * self.j_bank_touch[0]
            target_arm = [self.locked_yaw_hole, cur_j2, self.j_bank_touch[1], self.j_bank_touch[2]]
            target_grip = 0.42
            self.is_held = 1.0
        elif self.step_count <= 185:
            self.current_phase = "6B_BANK_CLEAR_LIFT"
            self.phase_metric = 0.75
            t_lift = min(1.0, (self.step_count - 165) / 18.0)
            s_lift = 0.5 * (1.0 - np.cos(np.pi * t_lift))
            cur_j2 = (1 - s_lift) * self.j_bank_touch[0] + s_lift * self.j_bank_lift[0]
            target_arm = [self.locked_yaw_hole, cur_j2, self.j_bank_lift[1], self.j_bank_lift[2]]
            target_grip = 0.42
            self.is_held = 1.0
        elif self.step_count <= 210:
            self.current_phase = "6C_HOLE_CREEP_ADVANCE"
            self.phase_metric = 0.75
            t_adv = min(1.0, (self.step_count - 185) / 22.0)
            s_adv = 0.5 * (1.0 - np.cos(np.pi * t_adv))
            cur_j2 = (1 - s_adv) * self.j_bank_lift[0] + s_adv * self.j_hole_hov[0]
            cur_j3 = (1 - s_adv) * self.j_bank_lift[1] + s_adv * self.j_hole_hov[1]
            cur_j4 = (1 - s_adv) * self.j_bank_lift[2] + s_adv * self.j_hole_hov[2]
            target_arm = [self.locked_yaw_hole, cur_j2, cur_j3, cur_j4]
            target_grip = 0.42
            self.is_held = 1.0
            
        # パターン 5: 挿入・直立化・離す [Step 211〜285]
        elif self.step_count <= 235:
            # Phase 7A: 穴直下への鉛直進入降下 (Step 211〜235)
            self.current_phase = "7A_VERTICAL_INSERTION"
            self.phase_metric = 0.90
            t_down = min(1.0, (self.step_count - 210) / 25.0)
            s_down = 0.5 * (1.0 - np.cos(np.pi * t_down))
            j2_ins = (1 - s_down) * self.j_hole_hov[0] + s_down * self.j_hole_ins[0]
            j3_ins = (1 - s_down) * self.j_hole_hov[1] + s_down * self.j_hole_ins[1]
            j4_ins = (1 - s_down) * self.j_hole_hov[2] + s_down * self.j_hole_ins[2]
            target_arm = [self.locked_yaw_hole, j2_ins, j3_ins, j4_ins]
            target_grip = 0.42
            self.is_held = 1.0

        elif self.step_count <= 252:
            # Phase 7B: 【段階的把持圧事前解放 ＆ 同調直立化】 (Step 236〜252)
            # 穴底到達後、手首Yawを locked_yaw_hole に固定したまま把持圧を先に減圧(0.42->0.05rad)。
            # 摩擦ロックが解除された後、手首Yawを0.0radへと滑らかにコサイン旋回させ、
            # 両爪が緩やかなガイド(+2.9° / 幅約35mm)としてペグの傾斜・転倒を支えながら直立化させる。
            # （これにより穴壁拘束によるJoint 1バックトルクおよび追従エラーを完全根絶！）
            self.current_phase = "7B_SOFTEN_UPRIGHT_ALIGN"
            self.phase_metric = 0.95
            t_grip = min(1.0, (self.step_count - 235) / 12.0)
            s_grip = 0.5 * (1.0 - np.cos(np.pi * t_grip))
            target_grip = (1.0 - s_grip) * 0.42 + s_grip * 0.05

            if self.step_count <= 239:
                cur_yaw = self.locked_yaw_hole
            else:
                t_yaw = min(1.0, (self.step_count - 239) / 13.0)
                s_yaw = 0.5 * (1.0 - np.cos(np.pi * t_yaw))
                cur_yaw = (1.0 - s_yaw) * self.locked_yaw_hole + s_yaw * 0.0

            target_arm = [cur_yaw, self.j_hole_ins[0], self.j_hole_ins[1], self.j_hole_ins[2]]
            self.is_held = 0.5

        else:
            # Phase 8: 爪全開 ＆ 垂直退避 ＆ ホーム復帰 (Step 253〜285)
            self.current_phase = "8_RELEASE_RETRACT"
            self.phase_metric = 0.95
            seated_yaw = 0.0
            if self.step_count <= 265:
                # 8A: 爪全開 (Step 253〜265: 13 steps) - 0.05radから-0.35rad(-20°)へ全開
                target_arm = [seated_yaw, self.j_hole_ins[0], self.j_hole_ins[1], self.j_hole_ins[2]]
                t_rel = min(1.0, (self.step_count - 252) / 13.0)
                s_rel = 0.5 * (1.0 - np.cos(np.pi * t_rel))
                target_grip = (1.0 - s_rel) * 0.05 + s_rel * (-0.35)
                self.is_held = 0.0
            elif self.step_count <= 275:
                # 8B: 鉛直上昇退避 (Step 266〜275: 10 steps)
                t_ret = min(1.0, (self.step_count - 265) / 10.0)
                s_ret = 0.5 * (1.0 - np.cos(np.pi * t_ret))
                cur_j2 = (1 - s_ret) * self.j_hole_ins[0] + s_ret * self.j_hole_hov[0]
                cur_j3 = (1 - s_ret) * self.j_hole_ins[1] + s_ret * self.j_hole_hov[1]
                cur_j4 = (1 - s_ret) * self.j_hole_ins[2] + s_ret * self.j_hole_hov[2]
                target_arm = [seated_yaw, cur_j2, cur_j3, cur_j4]
                target_grip = -0.35
                self.is_held = 0.0
            else:
                # 8C: ホーム復帰 (Step 276〜285: 10 steps)
                t_home = min(1.0, (self.step_count - 275) / 10.0)
                s_home = 0.5 * (1.0 - np.cos(np.pi * t_home))
                init_arm = [seated_yaw, self.j_hole_hov[0], self.j_hole_hov[1], self.j_hole_hov[2]]
                target_arm = [(1 - s_home) * init_arm[i] for i in range(4)]
                target_grip = -0.35
                self.is_held = 0.0

        # アナログ連続フィルタ (平滑化)
        smooth_rate = 0.35
        for i in range(4):
            self.current_cmd_arm[i] += smooth_rate * (target_arm[i] - self.current_cmd_arm[i])
        self.current_cmd_grip += smooth_rate * (target_grip - self.current_cmd_grip)

        # 指令パブリッシュ
        traj_arm = JointTrajectory()
        traj_arm.joint_names = self.arm_names
        pt_arm = JointTrajectoryPoint()
        pt_arm.positions = [float(x) for x in self.current_cmd_arm]
        pt_arm.time_from_start = Duration(sec=0, nanosec=int(self.timer_period * 1e9))
        traj_arm.points = [pt_arm]
        self.arm_pub.publish(traj_arm)

        traj_g = JointTrajectory()
        traj_g.joint_names = self.grip_names
        pt_g = JointTrajectoryPoint()
        pt_g.positions = [float(self.current_cmd_grip)]
        pt_g.time_from_start = Duration(sec=0, nanosec=int(self.timer_period * 1e9))
        traj_g.points = [pt_g]
        self.gripper_pub.publish(traj_g)

        # -------------------------------------------------------------
        # 実データ記録 (224x224 RGB画像, 実測関節角, 14次元特徴量, 目標指令)
        # -------------------------------------------------------------
        img_resized = cv2.resize(self.latest_image_bgr, (224, 224))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        feat_14d = self.compute_14dim_features(progress_tau)
        measured_joints = list(self.latest_joints)
        cmd_all = self.current_cmd_arm + [self.current_cmd_grip]

        self.recorded_images.append(img_rgb)
        self.recorded_joints.append(measured_joints)
        self.recorded_cmds.append(cmd_all)
        self.recorded_features.append(feat_14d)
        self.recorded_phases.append(self.phase_metric)

        if self.step_count % 30 == 0 or self.step_count == self.max_steps:
            self.get_logger().info(
                f"[{self.config['name']}] Step {self.step_count:03d}/{self.max_steps} | "
                f"Phase={self.current_phase:24s} | τ={progress_tau:.2f} | "
                f"J1={measured_joints[0]:+.3f} rad | Grip={measured_joints[4]:+.3f} rad"
            )


# -------------------------------------------------------------
# 4. メイン収集ループ実行関数
# -------------------------------------------------------------
def run_collection(num_episodes=10, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(SCRIPT_DIR, "data", "real_expert_dataset")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("🚀 【Step 6-1】実Gazebo環境での自律エキスパート実演データ収集ループ開始")
    print(f"📦 収集予定エピソード数: {num_episodes} (逆サイド5配置 ＋ 通常サイド5配置)")
    print(f"📁 保存先ディレクトリ: {output_dir}")
    print("💡 ブラウザ (http://localhost:6080) でロボットアームの自律動作をリアルタイム観察可能！")
    print("=" * 70)

    rclpy.init()

    summary_records = []
    
    for idx, cfg in enumerate(PEG_CONFIGS[:num_episodes]):
        print(f"\n🎬 >>> [Episode {idx+1}/{num_episodes}] 開始: {cfg['name']} (X={cfg['x']:.3f}m, Y={cfg['y']:.3f}m)")
        print(f"    説明: {cfg['desc']}")

        # 1. 前回の残存ペグを削除
        remove_peg_from_gazebo()
        
        # 2. アームをホーム位置へ復帰して静定
        reset_arm_to_home()

        # 3. 新ペグをスポーン
        spawn_success = spawn_peg_in_gazebo(cfg['x'], cfg['y'], z=1.03)
        if not spawn_success:
            print("    ⚠️ ペグスポーン再試行中...")
            time.sleep(1.0)
            spawn_peg_in_gazebo(cfg['x'], cfg['y'], z=1.03)
        print(f"    ✅ ペグスポーン＆静定完了: ({cfg['x']:.3f}, {cfg['y']:.3f}, 1.030)")

        # 4. ノード起動＆エピソード実行
        node = ExpertRolloutCollectorNode(cfg, max_steps=285)
        start_time = time.time()
        
        while rclpy.ok() and not node.is_done:
            rclpy.spin_once(node, timeout_sec=0.05)

        elapsed = time.time() - start_time
        node.destroy_node()

        # 5. データの検証と保存
        n_steps = len(node.recorded_joints)
        if n_steps >= 200:
            ep_file = os.path.join(output_dir, f"episode_{idx:02d}.npz")
            np.savez_compressed(
                ep_file,
                images=np.array(node.recorded_images, dtype=np.uint8),
                joints=np.array(node.recorded_joints, dtype=np.float32),
                cmd_joints=np.array(node.recorded_cmds, dtype=np.float32),
                features=np.array(node.recorded_features, dtype=np.float32),
                phases=np.array(node.recorded_phases, dtype=np.float32),
                config=np.array([cfg['x'], cfg['y'], float(cfg['id'])], dtype=np.float32)
            )
            print(f"    💾 エピソード保存完了: {ep_file} ({n_steps} ステップ, 所要時間: {elapsed:.1f}s)")
            summary_records.append({
                "id": idx, "name": cfg['name'], "x": cfg['x'], "y": cfg['y'],
                "steps": n_steps, "success": True, "time": elapsed
            })
        else:
            print(f"    ⚠️ ステップ数不足 ({n_steps} < 200)。スキップします。")
            summary_records.append({
                "id": idx, "name": cfg['name'], "x": cfg['x'], "y": cfg['y'],
                "steps": n_steps, "success": False, "time": elapsed
            })

    rclpy.shutdown()

    # 最後のペグを削除してクリーンアップ
    remove_peg_from_gazebo()

    print("\n" + "=" * 70)
    print(f"🎉 【データ収集完了】 全 {len(summary_records)} エピソードの収集が正常終了しました！")
    print(f"   成功エピソード数: {sum(1 for r in summary_records if r['success'])}/{len(summary_records)}")
    total_steps = sum(r['steps'] for r in summary_records if r['success'])
    print(f"   総記録フレーム数: {total_steps} フレーム (画像・関節・特徴量)")
    print("=" * 70)

    # -------------------------------------------------------------
    # 5. 注目点可視化グラフ 1 & 2 の生成
    # -------------------------------------------------------------
    generate_summary_graphs(summary_records, output_dir)


def generate_summary_graphs(summary_records=None, output_dir=None):
    """作業空間多様化分布図 (Graph 1) と 全ジョイント＆グリッパー実測軌道群 (Graph 2: J1〜J4, Hand) を生成"""
    if output_dir is None:
        output_dir = os.path.join(SCRIPT_DIR, "data", "real_expert_dataset")
    if summary_records is None:
        summary_records = [
            {"id": c["id"], "name": c["name"], "x": c["x"], "y": c["y"], "steps": 285, "success": True}
            for c in PEG_CONFIGS
        ]

    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    fig.patch.set_facecolor('#0f172a')

    # ---------------------------------------------------------
    # Panel [0, 0] - Graph 1: 作業空間ペグ配置多様化分布 (Workspace Diversity)
    # ---------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.set_facecolor('#1e293b')

    # ロボット基部・台座穴
    ax1.plot(0.0, 0.0, 's', color='#38bdf8', markersize=12, label='Robot Base (0, 0)')
    ax1.plot(0.20, 0.0, 'o', color='#facc15', markersize=16, label='Target Hole (0.20, 0.00)')

    # 可動限界円弧
    theta = np.linspace(-np.pi/3, np.pi/3, 100)
    ax1.plot(0.12 * np.cos(theta), 0.12 * np.sin(theta), '--', color='#94a3b8', label='Min Reach (R=0.12m)')
    ax1.plot(0.22 * np.cos(theta), 0.22 * np.sin(theta), '-', color='#cbd5e1', label='Max Reach (R=0.22m)')

    # ペグ配置プロット
    for r in summary_records:
        color = '#f43f5e' if r['y'] < 0.0 else '#10b981'
        marker = '^' if r['y'] < 0.0 else 'v'
        ax1.scatter(r['x'], r['y'], color=color, s=130, marker=marker, edgecolors='white', zorder=5)
        ax1.annotate(
            f"Ep{r['id']}:{r['name'].split('_')[1]}\n({r['x']:.2f},{r['y']:.2f})",
            (r['x'], r['y']),
            textcoords="offset points", xytext=(0, 8 if r['y'] >= 0 else -16),
            ha='center', fontsize=7.5, color='#f8fafc', weight='bold'
        )

    ax1.set_xlim(0.05, 0.25)
    ax1.set_ylim(-0.12, 0.12)
    ax1.set_xlabel("X [m] (Forward Reach)", color='white', fontsize=10, fontweight='bold')
    ax1.set_ylabel("Y [m] (Lateral Offset)", color='white', fontsize=10, fontweight='bold')
    ax1.set_title("Graph 1: 10 Reachable Configurations (Reverse & Normal)", color='#38bdf8', fontsize=11.5, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.35, color='#94a3b8')
    ax1.tick_params(colors='white')
    ax1.legend(loc='upper left', facecolor='#0f172a', edgecolor='#475569', labelcolor='white', fontsize=8)

    # ---------------------------------------------------------
    # Panels [0,1], [0,2], [1,0], [1,1], [1,2] - Graph 2: 全ジョイント(J1〜J4)＆グリッパー(Hand)実測軌道群
    # ---------------------------------------------------------
    ep_files = sorted([f for f in os.listdir(output_dir) if f.startswith("episode_") and f.endswith(".npz")])
    colors = plt.cm.turbo(np.linspace(0.08, 0.92, max(1, len(ep_files))))

    joint_specs = [
        (0, axes[0, 1], "Graph 2-A: Joint 1 (Yaw) Trajectories", "Joint 1 (Yaw) [rad]"),
        (1, axes[0, 2], "Graph 2-B: Joint 2 (Shoulder) Trajectories", "Joint 2 (Shoulder) [rad]"),
        (2, axes[1, 0], "Graph 2-C: Joint 3 (Elbow) Trajectories", "Joint 3 (Elbow) [rad]"),
        (3, axes[1, 1], "Graph 2-D: Joint 4 (Wrist) Trajectories", "Joint 4 (Wrist) [rad]"),
        (4, axes[1, 2], "Graph 2-E: Gripper (Hand) Trajectories", "Gripper (Hand) [rad]"),
    ]

    all_ep_data = [np.load(os.path.join(output_dir, ef)) for ef in ep_files]

    for j_idx, ax, title_str, ylabel_str in joint_specs:
        ax.set_facecolor('#1e293b')
        for i, data in enumerate(all_ep_data):
            joints = data['joints']
            time_axis = np.linspace(0, len(joints) * 0.10, len(joints))
            label_text = f"Ep{i} ({'Rev' if data['config'][1] < 0 else 'Norm'} Y={data['config'][1]:+.2f}m)" if i in [0, 4, 5, 9] else None
            ax.plot(time_axis, joints[:, j_idx], color=colors[i], alpha=0.88, linewidth=1.7, label=label_text)

        # 5大フェーズ区切り線
        ax.axvline(3.5, color='#fbbf24', linestyle='--', alpha=0.75, linewidth=1.2, label='P2: Grasp (3.5s)')
        ax.axvline(9.5, color='#34d399', linestyle='--', alpha=0.75, linewidth=1.2, label='P3: Lift (9.5s)')
        ax.axvline(21.0, color='#60a5fa', linestyle='--', alpha=0.75, linewidth=1.2, label='P4: Slide (21.0s)')
        ax.axvline(24.5, color='#c084fc', linestyle='--', alpha=0.75, linewidth=1.2, label='P5: Upright/Release (24.5s)')

        ax.set_xlabel("Time [s]", color='white', fontsize=10, fontweight='bold')
        ax.set_ylabel(ylabel_str, color='white', fontsize=10, fontweight='bold')
        ax.set_title(title_str, color='white', fontsize=11.5, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.35, color='#94a3b8')
        ax.tick_params(colors='white')
        ax.legend(loc='best', facecolor='#0f172a', edgecolor='#475569', labelcolor='white', fontsize=7.5, ncol=2)

    plt.suptitle(
        "Step 6-1: Real Gazebo Expert Demonstrations — Workspace Diversity & All 5-DoF Telemetry (Joints 1–4 & Gripper)",
        color='white', fontsize=14, fontweight='bold', y=0.995
    )
    plt.tight_layout()
    g_path = os.path.join(SCRIPT_DIR, "step6_1_real_telemetry.png")
    plt.savefig(g_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"📊 注目点可視化グラフ (全ジョイントJ1〜J4＆グリッパー対応) を生成・保存しました: {g_path}")


if __name__ == '__main__':
    if "--plot-only" in sys.argv:
        generate_summary_graphs()
    else:
        run_collection(num_episodes=10)

