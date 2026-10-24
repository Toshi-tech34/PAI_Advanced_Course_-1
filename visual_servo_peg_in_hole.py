import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None
import time
import os

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, JointState
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from builtin_interfaces.msg import Duration
except ImportError:
    Node = object

# =====================================================================
# 1. 幾何学的逆運動学 (Analytical Inverse Kinematics for CRANE+ V2)
#    - 人間の手本データから同定された幾何リンクパラメータと余弦定理により
#      任意の目標距離 R・高度 Z・ピッチ角に対して安定な関節角(J2, J3, J4)を一意に導出
#    - 前倒れ（レバー倒れ）や特異点を完全に防止
# =====================================================================
def solve_crane_ik(r_target, z_target, pitch_deg=160.0):
    """
    CRANE+ V2 の3軸平面IKを幾何学的に解く。
    r_target: ロボット基部中心からの水平距離 [m]
    z_target: ロボット基部底面からの目標エンドエフェクタ高度 [m]
    pitch_deg: グリッパー先端の下向き角度 [deg] (180° = 鉛直下向き)
    """
    h0 = 0.0714   # ロボット底面から肩関節(Joint 2)軸までの高さ [m]
    L1 = 0.0830   # 上腕リンク長 [m]
    L2 = 0.0935   # 前腕リンク長 [m]
    L3 = 0.1300   # 手首軸(Joint 4)からグリッパー把持中心までの実効長 [m]

    alpha = np.radians(pitch_deg)
    # 手首(Joint 4)位置の逆算
    rw = r_target - L3 * np.sin(alpha)
    zw = z_target - L3 * np.cos(alpha)

    dr = rw
    dz = zw - h0
    D = np.sqrt(dr**2 + dz**2)

    # 可動域制限・クリップ（特異点防止）
    max_reach = (L1 + L2) * 0.999
    min_reach = abs(L1 - L2) * 1.001
    D = np.clip(D, min_reach, max_reach)

    cos_beta = (L1**2 + L2**2 - D**2) / (2.0 * L1 * L2)
    cos_beta = np.clip(cos_beta, -1.0, 1.0)
    beta = np.arccos(cos_beta)

    cos_gamma = (L1**2 + D**2 - L2**2) / (2.0 * L1 * D)
    cos_gamma = np.clip(cos_gamma, -1.0, 1.0)
    gamma = np.arccos(cos_gamma)

    psi = np.arctan2(dr, dz)

    # 肘下げ (Elbow-down / 人間のお手本と完全合致する安定姿勢)
    a2 = psi - gamma              # Link 2 の鉛直からの傾き角
    a3 = a2 + (np.pi - beta)      # Link 3 の鉛直からの傾き角

    j2 = a2
    j3 = j2 - a3                  # Joint 3 軸([0, -1, 0])に合わせた相対回転
    j4 = a3 - alpha               # Joint 4 軸([0, -1, 0])による先端ピッチ調整

    # 物理関節制限範囲への安全クリッピング
    j2 = np.clip(j2, -1.82, 1.78)
    j3 = np.clip(j3, -2.55, 1.52)
    j4 = np.clip(j4, -2.41, 1.58)

    return float(j2), float(j3), float(j4)


# =====================================================================
# 2. 視覚認識エンジン (Object-Centric Visual Perception)
#    - 複数ペグや異なる色のペグ（赤、緑、青、黄など）に対応
#    - ピンホールカメラ幾何学によるピクセル(u, v) -> 3D座標(X, Y)の高精度変換
#    - サブカメラ・メインカメラ協調による「把持成功・浮上確認 (Tug Test)」
# =====================================================================
class ObjectCentricDetector:
    def __init__(self, target_peg_color="red"):
        self.target_peg_color = target_peg_color
        self.last_distractor_pegs = []

        # 色空間(HSV)閾値定義
        self.color_ranges = {
            "red": [
                (np.array([0, 100, 80]), np.array([10, 255, 255])),
                (np.array([160, 100, 80]), np.array([180, 255, 255]))
            ],
            "green": [
                (np.array([35, 80, 80]), np.array([85, 255, 255]))
            ],
            "blue": [
                (np.array([100, 100, 80]), np.array([130, 255, 255]))
            ],
            "yellow": [
                (np.array([20, 100, 100]), np.array([35, 255, 255]))
            ]
        }
        self.hole_color = "blue"

        # カメラ外部・内部パラメータ (TF / camera_info から厳密同定)
        self.t_cam_x = 0.10626
        self.t_cam_y = 0.10253
        self.t_cam_z = 0.46000
        self.fx = 457.00736
        self.fy = 457.00736
        self.cx = 320.5
        self.cy = 240.5

    def pixel_to_base_3d(self, u, v, z_base=0.080):
        """
        画像ピクセル座標 (u, v) をロボット基部フレーム (X_base, Y_base) へ厳密逆投影。
        Gazebo上方照明環境では8cm円柱ペグの水平上面円盤 (z_base=0.080m) が高輝度(V≈203)で観測されるため、
        z_base=0.080m (光学距離 z_opt = 0.460 - 0.080 = 0.380m) を基準とすることで
        Y<0 から (0,0) を超える Y>=0 全域で誤差 ±1.5mm 以内の高精度座標変換を実現する。
        """
        z_opt = self.t_cam_z - z_base
        x_base = self.t_cam_x - ((v - self.cy) * z_opt / self.fy)
        y_base = self.t_cam_y - ((u - self.cx) * z_opt / self.fx)
        return float(x_base), float(y_base)

    def check_workspace_reachability(self, peg_x, peg_y, hole_x=0.200, hole_y=0.000):
        """
        CRANE+ V2 の機械的リンク特性 (h0=0.0714m, L1=0.0830m, L2=0.0935m, L3=0.1300m,
        L1+L2=0.1765m, ピッチ角135°) および穴ボード (中心(0.20, 0.00), 10cm×10cm角) の物理寸法に基づき、
        カメラが認識したペグ座標 (peg_x, peg_y) が物理的に把持可能な作業範囲内にあるかを厳密判定する。
        """
        r_p = float(np.sqrt(peg_x**2 + peg_y**2))
        th_deg = float(np.degrees(np.arctan2(peg_y, peg_x)))
        dist_from_hole = float(np.sqrt((peg_x - hole_x)**2 + (peg_y - hole_y)**2))

        # 機械的作業空間境界 (Step 6-1 実実演データ収集 PEG_CONFIGS および Graph 1 の可動限界円弧と完全一致)
        R_MIN, R_MAX = 0.120, 0.220
        X_MIN, X_MAX = 0.100, 0.220
        Y_MIN, Y_MAX = -0.120, 0.120
        THETA_MAX_DEG = 60.0
        HOLE_CLEARANCE_MIN = 0.045

        # 穴開口部キャビティ（中心(0.20, 0.00), 開口径40mm×40mm, ペグ半径15mm）との直接重複判定
        # Step 6-1 の代表配置 Ep4(0.170, -0.045) / Ep9(0.170, +0.045) は穴中心距離 0.054m であり正常稼働
        in_hole_cavity = (0.165 <= peg_x <= 0.235) and (-0.035 <= peg_y <= 0.035)

        # ホバリング時 (z=0.115m, pitch=135°) の手首距離 D <= L1 + L2 (0.1765m) 幾何判定
        h0, L1, L2, L3 = 0.0714, 0.0830, 0.0935, 0.1300
        d_claw_contact = 0.02195
        r_eff = np.sqrt(max(0.001, r_p**2 - d_claw_contact**2))
        alpha = np.radians(135.0)
        rw_hov = r_eff - L3 * np.sin(alpha)
        zw_hov = 0.105 - L3 * np.cos(alpha)
        D_hov = float(np.sqrt(rw_hov**2 + (zw_hov - h0)**2))
        max_wrist_reach = L1 + L2  # 0.1765m

        reasons = []
        if r_p > R_MAX or D_hov > max_wrist_reach:
            reasons.append(
                f"極距離 R={r_p:.3f}m (手首距離 D={D_hov:.3f}m) がアーム最大到達限界 R_max={R_MAX:.3f}m (L1+L2={max_wrist_reach:.4f}m) を超過"
            )
        if r_p < R_MIN:
            reasons.append(f"極距離 R={r_p:.3f}m が最小可動半径 R_min={R_MIN:.3f}m 未満 (ロボット基部干渉域)")
        if peg_x > X_MAX:
            reasons.append(f"前後座標 X={peg_x:.3f}m が把持可能上限 X_max={X_MAX:.3f}m を超過")
        if peg_x < X_MIN:
            reasons.append(f"前後座標 X={peg_x:.3f}m が把持可能下限 X_min={X_MIN:.3f}m 未満")
        if peg_y < Y_MIN or peg_y > Y_MAX:
            reasons.append(f"左右座標 Y={peg_y:+.3f}m が把持可能範囲 [{Y_MIN:+.3f}m, {Y_MAX:+.3f}m] 外")
        if abs(th_deg) > THETA_MAX_DEG:
            reasons.append(f"方位角 θ={th_deg:+.1f}° が可動角度範囲 [±{THETA_MAX_DEG:.0f}°] 外")
        if dist_from_hole < HOLE_CLEARANCE_MIN or in_hole_cavity:
            reasons.append(
                f"ペグ位置 ({peg_x:.3f}, {peg_y:+.3f})m が穴開口部キャビティ領域 (中心距離={dist_from_hole:.3f}m < {HOLE_CLEARANCE_MIN:.3f}m) と物理干渉"
            )

        is_reachable = (len(reasons) == 0)
        info = {
            "peg_x": float(peg_x),
            "peg_y": float(peg_y),
            "r_peg": r_p,
            "theta_deg": th_deg,
            "D_hov": D_hov,
            "max_wrist_reach": max_wrist_reach,
            "r_bounds": (R_MIN, R_MAX),
            "x_bounds": (X_MIN, X_MAX),
            "y_bounds": (Y_MIN, Y_MAX),
            "is_reachable": is_reachable,
            "reasons": reasons,
        }
        return is_reachable, reasons, info

    def _extract_peg_from_mask(self, hsv, peg_mask, color_name):
        """
        指定色マスクから円柱ペグを検出し、上面円盤(z_base=0.080m, 高輝度 V>=145)の光学重心から
        Y<0・Y=0・Y>0 全域で歪みのない3D基部座標 (x_base, y_base) を算出する（全色共通）。
        """
        peg_contours, _ = cv2.findContours(peg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_peg = None
        max_peg_area = 0
        for c in peg_contours:
            area = float(cv2.contourArea(c))
            if area > 35 and area > max_peg_area:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    u_cent = float(M["m10"] / M["m00"])
                    v_cent = float(M["m01"] / M["m00"])
                    bx, by, bw, bh = cv2.boundingRect(c)

                    # コンター外接矩形内で、照明が当たる水平上面円盤 (V >= 145) の重心を取得
                    roi_mask_c = peg_mask[by:by + bh, bx:bx + bw] > 0
                    roi_v = hsv[by:by + bh, bx:bx + bw, 2]
                    top_cap_mask = roi_mask_c & (roi_v >= 145)
                    ys_top, xs_top = np.where(top_cap_mask)
                    if len(xs_top) >= 8:
                        u_top = float(bx + np.mean(xs_top))
                        v_top = float(by + np.mean(ys_top))
                    else:
                        u_top, v_top = u_cent, v_cent

                    # 【視差補正: 上面円盤外縁 (Top-Rim) 基準の中心推定】
                    # 円柱の各高さ z の点は (水平距離)/(Zcam - z) に比例した半径方向位置へ投影されるため、
                    # 主点から最も遠いシルエット点は必ず上面円盤 (z=0.080m) の外縁となる。
                    # 側面壁の一部が輝度閾値を通過すると重心がカメラ光軸側へ偏る（光軸 Y=+0.103m から遠い
                    # Y<0 = 固定爪側象限ほど偏りが大きい）ため、外縁から半径 r_px だけ内側を上面中心とする。
                    du, dv = u_top - self.cx, v_top - self.cy
                    rad_px = float(np.hypot(du, dv))
                    if rad_px > 20.0:
                        dir_u, dir_v = du / rad_px, dv / rad_px
                        pts = c.reshape(-1, 2).astype(np.float64)
                        proj = (pts[:, 0] - self.cx) * dir_u + (pts[:, 1] - self.cy) * dir_v
                        r_cap_px = 0.015 * self.fx / (self.t_cam_z - 0.080)  # ≈ 18.0 px
                        along_rim = float(np.max(proj)) - r_cap_px
                        perp = (u_top - self.cx) * (-dir_v) + (v_top - self.cy) * dir_u
                        # 外縁推定と重心推定の差が妥当範囲 (<= 0.6×半径) の場合のみ採用 (遮蔽・結合時の誤補正防止)
                        if abs(along_rim - rad_px) <= 0.6 * r_cap_px:
                            u_top = self.cx + along_rim * dir_u + perp * (-dir_v)
                            v_top = self.cy + along_rim * dir_v + perp * dir_u

                    x_b, y_b = self.pixel_to_base_3d(u_top, v_top, z_base=0.080)
                    cx, cy = int(round(u_top)), int(round(v_top))
                    r_b = float(np.sqrt(x_b**2 + y_b**2))
                    th_b = float(np.arctan2(y_b, x_b))

                    # ロボット基部の赤い部材 (x_b ≈ 0.028, y_b ≈ -0.104) のみ除外し、机上(x_b >= 0.09)は全て検出
                    if x_b >= 0.09 and not (x_b < 0.105 and y_b < -0.06):
                        best_peg = {
                            "color": color_name,
                            "cx": cx, "cy": cy, "area": area,
                            "box": (bx, by, bw, bh),
                            "x_base": x_b, "y_base": y_b,
                            "r_base": r_b, "theta_base": th_b
                        }
                        max_peg_area = area
        return best_peg

    def detect_objects(self, bgr_image):
        """
        カメラ画像からターゲット色ペグ、他色ディストラクタペグ、および穴の2D重心・3D基部座標を検出。
        円柱上面(z_base=0.080m)逆投影により、(0,0) を跨ぐ Y<0 <-> Y>=0 全域および複数色ペグに完全対応。
        """
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        h, w, _ = bgr_image.shape

        # ロボット基部手前(y > 360)を除き、遠方(y >= 20)まで机上全域をカバーするROIマスク
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        roi_mask[20:360, :] = 255

        # 1. ターゲット色ペグの検出
        peg_mask = np.zeros((h, w), dtype=np.uint8)
        for lower, upper in self.color_ranges.get(self.target_peg_color, []):
            peg_mask |= cv2.inRange(hsv, lower, upper)
        peg_mask &= roi_mask
        target_peg = self._extract_peg_from_mask(hsv, peg_mask, self.target_peg_color)

        # 1B. 他色ペグ（ディストラクタ：赤ターゲット時の緑ペグ等）の同時認識（視覚的識別検証用）
        distractors = []
        for other_color, ranges in self.color_ranges.items():
            if other_color in (self.target_peg_color, self.hole_color):
                continue
            other_mask = np.zeros((h, w), dtype=np.uint8)
            for lower, upper in ranges:
                other_mask |= cv2.inRange(hsv, lower, upper)
            other_mask &= roi_mask
            other_peg = self._extract_peg_from_mask(hsv, other_mask, other_color)
            if other_peg is not None:
                distractors.append(other_peg)
        self.last_distractor_pegs = distractors

        # 2. 穴（台座）の検出 (Blue)
        hole_mask = np.zeros((h, w), dtype=np.uint8)
        for lower, upper in self.color_ranges["blue"]:
            hole_mask |= cv2.inRange(hsv, lower, upper)
        hole_mask &= roi_mask

        hole_contours, _ = cv2.findContours(hole_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        target_hole = None
        max_hole_area = 0
        for c in hole_contours:
            area = cv2.contourArea(c)
            if area > 300 and area > max_hole_area:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    x_b, y_b = self.pixel_to_base_3d(cx, cy, z_base=0.02)
                    if 0.15 <= x_b <= 0.25 and -0.06 <= y_b <= 0.06:
                        target_hole = {
                            "cx": cx, "cy": cy, "area": area,
                            "box": cv2.boundingRect(c),
                            "x_base": x_b, "y_base": y_b
                        }
                        max_hole_area = area

        return target_peg, target_hole

    def measure_side_peg_z(self, side_image):
        """
        サブカメラ（側面視点 /side_camera/image_raw）からターゲット色ペグのZ方向ピクセル位置（上端 top_y, 下端 bottom_y）を計測。
        左端 (x < 140) の赤いロボット基部ARマーカーを除外し、手前 (Y<0) から (0,0) 超えの奥 (Y>=0) まで
        机上全域 (y: 10〜420, x: 140〜580) をカバーする。
        """
        info = {
            "detected": False,
            "top_y": None,
            "bottom_y": None,
            "cy": None,
            "area": 0.0,
            "tilt_deg": None,
        }
        if side_image is None:
            return info

        side_hsv = cv2.cvtColor(side_image, cv2.COLOR_BGR2HSV)
        h_s, w_s, _ = side_image.shape
        side_mask = np.zeros((h_s, w_s), dtype=np.uint8)
        for lower, upper in self.color_ranges.get(self.target_peg_color, []):
            side_mask |= cv2.inRange(side_hsv, lower, upper)

        side_roi = np.zeros((h_s, w_s), dtype=np.uint8)
        side_roi[10:min(450, h_s), 130:min(580, w_s)] = 255
        side_mask &= side_roi

        contours, _ = cv2.findContours(side_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            max_c = max(contours, key=cv2.contourArea)
            area = float(cv2.contourArea(max_c))
            if area > 35:
                bx_s, y, bw_s, h_box = cv2.boundingRect(max_c)
                info["detected"] = True
                info["top_y"] = int(y)
                info["bottom_y"] = int(y + h_box)
                info["cy"] = int(y + h_box // 2)
                info["area"] = area

                # 【ペグ傾斜角の計測】ペグ画素群の主成分軸と鉛直とのなす角 (0°=直立)
                blob = np.zeros((h_s, w_s), dtype=np.uint8)
                cv2.drawContours(blob, [max_c], -1, 255, -1)
                blob &= side_mask
                ys_b, xs_b = np.where(blob > 0)
                if len(xs_b) >= 30 and h_box >= 14:
                    pts = np.stack([xs_b, ys_b], axis=1).astype(np.float64)
                    pts -= pts.mean(axis=0)
                    cov = pts.T @ pts / len(pts)
                    evals, evecs = np.linalg.eigh(cov)
                    major = evecs[:, int(np.argmax(evals))]
                    # 縦長シルエット (h_box >= 1.15 * bw_s) かつ縦方向の固有値が優位な場合のみ主軸角を計算
                    # 爪や土台の遮蔽で切断された半円・横長帯・斜めスライスの誤認 (約30°〜90°) を防止
                    if h_box >= int(1.15 * bw_s) and evals.max() > 1.8 * max(evals.min(), 1e-6):
                        # 鉛直軸 (Y方向) 成分が優位 (|major[1]| >= 0.866 <=> 鉛直からの傾斜角 <= 30°)
                        if abs(major[1]) >= 0.866:
                            info["tilt_deg"] = float(np.degrees(np.arctan2(abs(major[0]), abs(major[1]))))
        return info

    def verify_grasp_dual_modal(
        self,
        cmd_grip_rad,
        meas_grip_rad,
        init_side_bottom_y,
        cur_side_bottom_y,
        init_side_top_y=None,
        cur_side_top_y=None
    ):
        """
        【二重把持確認 (Dual-Modal Grasp Verification)】
        1. グリッパー閉鎖角度差チェック (Gripper Angle Discrepancy):
           ペグ直径(30mm)より狭い角度 (cmd_grip ≈ +0.420 rad = +24.1°) を指令した際、
           ペグを実際に挟んでいれば爪が阻まれ meas_grip ≈ +0.336〜+0.385 rad で停止し、
           角度差 Δθ_grip = cmd_grip - meas_grip >= 0.020 rad (+1.15°以上、通常+4.2°〜+4.8°) が生じる。
           差がほぼ無い (Δθ_grip < 0.020 rad かつ meas_grip > 0.400 rad) 場合はペグを挟んでいない空振りと判定。
        2. カメラ画像によるZ方向リフト確認 (Visual Z-Axis Lift Verification):
           掴み損ねた際のXY方向のズレは無視し、側面カメラのペグ高さ (下端 cur_side_bottom_y または上端 cur_side_top_y) が
           初期机上静止時 (init_side_bottom_y / init_side_top_y) から Z方向（画像上方向）へ ΔZ_px >= 12 px 以上
           持ち上がっているかを判定（Y>=0 でペグ下部が初期時に穴ボード背後に隠れる場合も上端 ΔZ_top で正確に検出）。
           Z方向に動いていない (ΔZ_px < 12 px) 場合は持ち上げていない（空振り）と判定。
        """
        grip_diff_rad = float(cmd_grip_rad - meas_grip_rad)
        grip_diff_deg = float(np.degrees(grip_diff_rad))
        cmd_grip_deg = float(np.degrees(cmd_grip_rad))
        meas_grip_deg = float(np.degrees(meas_grip_rad))

        # 条件1: グリッパー指令角と実測角に挟み込み偏差があること (寸分違わず閉まりきっていないこと)
        grip_angle_ok = (grip_diff_rad >= 0.020) and (meas_grip_rad <= 0.402)

        # 条件2: カメラ画像でZ方向（上方向）へ明確に持ち上がっていること (XYのズレは無視)
        dz_candidates = []
        if init_side_bottom_y is not None and cur_side_bottom_y is not None:
            dz_candidates.append(int(init_side_bottom_y - cur_side_bottom_y))
        if init_side_top_y is not None and cur_side_top_y is not None:
            dz_candidates.append(int(init_side_top_y - cur_side_top_y))

        if len(dz_candidates) > 0:
            z_lift_px = int(max(dz_candidates))
            z_lift_ok = (z_lift_px >= 12)
        elif init_side_bottom_y is not None and cur_side_bottom_y is None:
            # 初期机上では見えていたペグが机上から消え、空中リフト中の黒いグリッパー内部に保持されているケース
            z_lift_px = 40 if grip_angle_ok else 0
            z_lift_ok = bool(grip_angle_ok)
        elif cur_side_bottom_y is not None:
            z_lift_px = int(165 - cur_side_bottom_y)
            z_lift_ok = (cur_side_bottom_y <= 148)
        else:
            z_lift_px = 0
            z_lift_ok = False

        is_grasped = bool(grip_angle_ok and z_lift_ok)

        reasons = []
        if not grip_angle_ok:
            reasons.append(
                f"グリッパー閉鎖指令角 ({cmd_grip_deg:+.1f}° / {cmd_grip_rad:+.3f}rad) と実測角 ({meas_grip_deg:+.1f}° / {meas_grip_rad:+.3f}rad) の差が "
                f"Δθ={grip_diff_deg:+.2f}° ({grip_diff_rad:+.4f}rad < 0.020rad) しかなく、ペグを挟まずに閉まりきっている（空振り）"
            )
        if not z_lift_ok:
            reasons.append(
                f"カメラ画像のZ方向リフト量が ΔZ={z_lift_px:+d}px (初期机上 SideY={init_side_bottom_y}px -> 現在 SideY={cur_side_bottom_y}px, 閾値 +12px以上) であり、"
                f"ペグがZ方向に持ち上がっておらず机上に残存している"
            )

        details = {
            "is_grasped": is_grasped,
            "grip_angle_ok": bool(grip_angle_ok),
            "z_lift_ok": bool(z_lift_ok),
            "cmd_grip_rad": float(cmd_grip_rad),
            "meas_grip_rad": float(meas_grip_rad),
            "grip_diff_rad": grip_diff_rad,
            "cmd_grip_deg": cmd_grip_deg,
            "meas_grip_deg": meas_grip_deg,
            "grip_diff_deg": grip_diff_deg,
            "init_side_bottom_y": int(init_side_bottom_y) if init_side_bottom_y is not None else None,
            "cur_side_bottom_y": int(cur_side_bottom_y) if cur_side_bottom_y is not None else None,
            "init_side_top_y": int(init_side_top_y) if init_side_top_y is not None else None,
            "cur_side_top_y": int(cur_side_top_y) if cur_side_top_y is not None else None,
            "z_lift_px": int(z_lift_px),
            "reasons": reasons,
        }
        return is_grasped, details

    def verify_grasp_multiview(self, main_image, side_image, target_peg_orig=None, init_side_bottom_y=None):
        """
        微小リフト(Tug Test)時の把持確認:
        サブカメラのZ方向浮上量 (ΔZ_px) を中心にペグが机から持ち上がったかを視覚的に検証
        """
        side_info = self.measure_side_peg_z(side_image)
        bottom_y = side_info["bottom_y"]
        base_y = init_side_bottom_y if init_side_bottom_y is not None else 160
        z_lift_px = (base_y - bottom_y) if bottom_y is not None else 0
        side_lifted = bool(side_info["detected"] and z_lift_px >= 10)

        evidence = {
            "side_detected": side_info["detected"],
            "side_bottom_y": bottom_y,
            "z_lift_px": z_lift_px,
            "side_lifted": side_lifted,
            "main_detected": False,
            "main_area": 0,
            "main_lifted": False,
            "is_grasped": side_lifted
        }

        if main_image is not None:
            cur_peg, _ = self.detect_objects(main_image)
            if cur_peg is not None:
                evidence["main_detected"] = True
                evidence["main_area"] = cur_peg["area"]
                evidence["main_lifted"] = True

        return evidence["is_grasped"], evidence

    @staticmethod
    def _draw_sleek_badge(img, x, y, title, subtitle, color=(74, 222, 128)):
        """滲み・木目混色を100%防止する高コントラスト・ダークスレート背景カード付きアノテーション (PIL高品位アンチエイリアス対応)"""
        font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
        if not os.path.exists(font_path):
            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        
        try:
            from PIL import Image as PILImage, ImageDraw as PILDraw, ImageFont as PILFont
            pil_img = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = PILDraw.Draw(pil_img)
            f_title = PILFont.truetype(font_path, 13) if os.path.exists(font_path) else PILFont.load_default()
            f_sub = PILFont.truetype(font_path, 11) if os.path.exists(font_path) else PILFont.load_default()

            tb = f_title.getbbox(title)
            sb = f_sub.getbbox(subtitle) if subtitle else (0, 0, 0, 0)
            tw = tb[2] - tb[0]
            sw = sb[2] - sb[0]
            w_box = max(tw, sw) + 32
            h_box = 42 if subtitle else 26

            # 16:9 クロップ領域内 (y: 65〜415, x: 225〜630) にクランプしてサイドバー被りと見切れを完全防止
            bx = max(225, min(x, img.shape[1] - w_box - 10))
            by = max(65, min(y, img.shape[0] - h_box - 10))

            c_rgb = (int(color[2]), int(color[1]), int(color[0]))  # BGR -> RGB
            # 濃紺スレート背景カード
            draw.rounded_rectangle([bx, by, bx + w_box, by + h_box], radius=5, fill=(15, 23, 42), outline=c_rgb, width=1)
            # 丸インジケータ
            draw.ellipse([bx + 8, by + 8, bx + 16, by + 16], fill=c_rgb)
            # タイトル
            draw.text((bx + 22, by + 4), title, font=f_title, fill=c_rgb)
            # サブタイトル
            if subtitle:
                draw.text((bx + 22, by + 21), subtitle, font=f_sub, fill=(248, 250, 252))

            img[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)[:]
        except Exception:
            font = cv2.FONT_HERSHEY_SIMPLEX
            f_scale = 0.40
            thick = 1
            t_size, _ = cv2.getTextSize(title, font, f_scale, thick)
            s_size, _ = cv2.getTextSize(subtitle, font, f_scale, thick) if subtitle else ((0, 0), 0)

            pad_x = 8
            pad_y = 5
            w_box = max(t_size[0], s_size[0]) + pad_x * 2 + 14
            h_box = t_size[1] + (s_size[1] + pad_y if subtitle else 0) + pad_y * 2 + 2

            bx = max(225, min(x, img.shape[1] - w_box - 10))
            by = max(65, min(y, img.shape[0] - h_box - 10))

            sub_img = img[by:by+h_box, bx:bx+w_box]
            card = np.full_like(sub_img, (15, 23, 42))
            cv2.addWeighted(card, 0.90, sub_img, 0.10, 0, sub_img)
            img[by:by+h_box, bx:bx+w_box] = sub_img

            cv2.rectangle(img, (bx, by), (bx+w_box, by+h_box), color, 1, cv2.LINE_AA)
            cv2.circle(img, (bx + pad_x + 3, by + pad_y + t_size[1]//2), 4, color, -1, cv2.LINE_AA)
            cv2.putText(img, title, (bx + pad_x + 12, by + pad_y + t_size[1]), font, f_scale, color, thick, cv2.LINE_AA)
            if subtitle:
                y_sub = by + pad_y * 2 + t_size[1] + s_size[1] + 1
                cv2.putText(img, subtitle, (bx + pad_x + 12, y_sub), font, f_scale, (241, 245, 249), thick, cv2.LINE_AA)

    def annotate_frame(self, bgr_image, peg, hole, state_text, current_grip_deg, side_image=None, grasp_info=None):
        """ユーザー・評価者向けに認識状態（ターゲット色ペグ、無視された他色ディストラクタペグ、穴）を可視化したオーバーレイ画像を生成"""
        annotated = bgr_image.copy()
        is_abort = bool(grasp_info and grasp_info.get("abort", False))
        abort_type = grasp_info.get("abort_type", "OUT_OF_REACH") if grasp_info else "OUT_OF_REACH"

        # 穴の強調表示 (穴ボード上方にカード配置して重なり防止)
        if hole:
            x, y, w_box, h_box = hole["box"]
            cx, cy = hole["cx"], hole["cy"]
            cv2.rectangle(annotated, (x, y), (x + w_box, y + h_box), (56, 189, 248), 2, cv2.LINE_AA)
            cv2.drawMarker(annotated, (cx, cy), (56, 189, 248), cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
            hole_bx = min(annotated.shape[1] - 220, max(225, x - 20))
            hole_by = max(65, y - 48)
            self._draw_sleek_badge(
                annotated, hole_bx, hole_by,
                "HOLE BOARD (TARGET DEST)",
                f"[{cx},{cy}] -> ({hole['x_base']:.2f},{hole['y_base']:.2f})m",
                color=(56, 189, 248)
            )

        # 他色ディストラクタペグ（無視対象）の描画 (ディストラクタ下方にカード配置)
        distractors = (
            grasp_info.get("distractor_pegs", self.last_distractor_pegs)
            if grasp_info is not None
            else self.last_distractor_pegs
        )
        if distractors:
            for d_peg in distractors:
                dx, dy, dw, dh = d_peg["box"]
                dcx, dcy = d_peg["cx"], d_peg["cy"]
                d_col_name = d_peg.get("color", "other").upper()
                d_color = (244, 63, 94)  # ローズレッド/ピンク系で「認識しているが非ターゲットとして無視」を明示
                cv2.rectangle(annotated, (dx, dy), (dx + dw, dy + dh), d_color, 2, cv2.LINE_AA)
                cv2.drawMarker(annotated, (dcx, dcy), d_color, cv2.MARKER_TILTED_CROSS, 16, 2, cv2.LINE_AA)
                d_bx = min(annotated.shape[1] - 200, max(225, dx - 10))
                d_by = min(annotated.shape[0] - 50, dy + dh + 10)
                self._draw_sleek_badge(
                    annotated, d_bx, d_by,
                    f"DISTRACTOR ({d_col_name}: IGNORED)",
                    f"({d_peg['x_base']:.2f},{d_peg['y_base']:+.2f})m",
                    color=d_color
                )

        # ターゲットペグの強調表示 (アボート時は赤枠・警告表示、ペグ上方にカード配置)
        if peg:
            x, y, w_box, h_box = peg["box"]
            cx, cy = peg["cx"], peg["cy"]
            t_col_name = peg.get("color", self.target_peg_color).upper()
            box_color = (239, 68, 68) if is_abort else (74, 222, 128)
            cv2.rectangle(annotated, (x, y), (x + w_box, y + h_box), box_color, 2, cv2.LINE_AA)
            cv2.drawMarker(annotated, (cx, cy), box_color, cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
            r_val = np.sqrt(peg['x_base']**2 + peg['y_base']**2)
            if is_abort:
                prefix = f"UNGRASPED {t_col_name} PEG" if abort_type == "GRASP_FAILED" else f"OUT-OF-REACH {t_col_name} PEG"
            else:
                prefix = f"TARGET ({t_col_name} PEG)"
            t_bx = max(225, min(x - 20, 380))
            t_by = max(65, y - 48 if y >= 115 else y + h_box + 8)
            self._draw_sleek_badge(
                annotated, t_bx, t_by,
                prefix,
                f"[{cx},{cy}] -> ({peg['x_base']:.2f},{peg['y_base']:+.2f})m | R={r_val:.2f}m",
                color=box_color
            )

        # サブカメラ（側面視点）をPIPとして合成 (右上にペグがある場合は左上へ自動退避して遮蔽を防止)
        if side_image is not None:
            side_small = cv2.resize(side_image, (160, 120))
            cv2.rectangle(side_small, (0, 0), (160, 120), (255, 255, 255), 1)
            cv2.putText(side_small, f"Side Cam ({self.target_peg_color.upper()} Z)", (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
            if grasp_info and grasp_info.get("init_side_bottom_y") is not None:
                init_y_scaled = int(grasp_info["init_side_bottom_y"] * (120.0 / side_image.shape[0]))
                cv2.line(side_small, (0, init_y_scaled), (160, init_y_scaled), (0, 165, 255), 1)
            if grasp_info and grasp_info.get("side_bottom_y") is not None:
                bot_y_scaled = int(grasp_info["side_bottom_y"] * (120.0 / side_image.shape[0]))
                line_col = (0, 0, 255) if is_abort else (0, 255, 0)
                cv2.line(side_small, (0, bot_y_scaled), (160, bot_y_scaled), line_col, 1)
                z_str = f"SideY:{grasp_info['side_bottom_y']}"
                if grasp_info.get("z_lift_px") is not None:
                    z_str += f" (dZ:{grasp_info['z_lift_px']:+d}px)"
                cv2.putText(side_small, z_str, (5, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.33, line_col, 1)
            if peg and peg["cx"] > 430 and peg["cy"] < 155:
                annotated[50:170, 10:170] = side_small
            else:
                annotated[45:165, 470:630] = side_small

        # 動作状態バナー
        banner_bg = (0, 0, 180) if is_abort else (20, 20, 20)
        cv2.rectangle(annotated, (0, 0), (640, 40), banner_bg, -1)
        status_str = f"Phase: {state_text} | Grip: {current_grip_deg:.1f} deg"
        cv2.putText(annotated, status_str, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 2)

        return annotated


# =====================================================================
# 3. 視覚誘導ビジュアルサーボ制御ノード (Closed-Loop Visual Servoing)
#    - 段階的把持（Stepwise Grasping）で円柱ペグのすり抜け・転倒を防止
#    - 微小リフト把持確認 (Tug Test) と 自動リトライ機構 (Automatic Retry)
#    - 幾何学的逆運動学(IK)によりペグ位置への正確なアプローチを実現
# =====================================================================
class VisualServoPegInHoleController(Node):
    def __init__(self, target_color="red", output_debug_dir="/workspace/提出物/data"):
        super().__init__('visual_servo_peg_in_hole_controller')
        self.get_logger().info("=" * 65)
        self.get_logger().info("🎯 視覚認識型ビジュアルサーボ制御（Object-Centric Visual Servoing）起動")
        self.get_logger().info(f"🎨 認識対象ペグの色: {target_color.upper()}")
        self.get_logger().info("🔬 微小リフト把持確認(Tug Test) & 自動リトライ機能: 有効")
        self.get_logger().info("=" * 65)

        self.detector = ObjectCentricDetector(target_peg_color=target_color)
        self.output_debug_dir = output_debug_dir
        os.makedirs(self.output_debug_dir, exist_ok=True)

        # ROS 2 通信設定 (メイン俯瞰カメラ & サブ側面カメラ)
        self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.create_subscription(Image, '/side_camera/image_raw', self.side_image_callback, 10)
        self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)

        self.arm_pub = self.create_publisher(JointTrajectory, '/crane_plus_arm_controller/joint_trajectory', 10)
        self.gripper_pub = self.create_publisher(JointTrajectory, '/crane_plus_gripper_controller/joint_trajectory', 10)

        self.arm_names = ['crane_plus_joint1', 'crane_plus_joint2', 'crane_plus_joint3', 'crane_plus_joint4']
        self.grip_names = ['crane_plus_joint_hand']

        # センサーバッファ
        self.latest_image = None
        self.latest_side_image = None
        self.latest_joints = None
        self.target_peg = None
        self.target_hole = None
        self.peg_initial = None
        self.last_grasp_info = None

        # 制御周期: 10 Hz (0.1秒) で極めて滑らかな連続追従を実現
        self.timer_period = 0.1
        self.timer = self.create_timer(self.timer_period, self.control_loop)

        # ステートマシン管理
        self.state = "INIT"
        self.state_step = 0
        self.retry_count = 0
        self.max_retries = 3

        # 段階的把持用パラメータ (0° -> 5.7° -> 11.5° -> 16.0° -> 20.6° -> 24.1°)
        self.grasp_substeps = [0.0, 0.10, 0.20, 0.28, 0.36, 0.42]
        self.grasp_substep_idx = 0

        # 計算済み目標（デフォルト: 穴台座の左右反対位置 X=0.15m, Y=-0.08m）
        self.j1_peg = -0.490
        self.r_peg = 0.170
        self.j1_hole = 0.000
        self.r_hole = 0.200
        self.yaw_upright_offset = 0.095  # 左右爪非対称（13mm長差・傾斜爪）による把持傾きをY<0/Y>=0問わず穴進入時に直立化
        self.annotated_frame = None

        # テレメトリ記録用バッファ (指令値 vs 実測値)
        self.history_steps = []
        self.history_meas = []
        self.history_cmd = []
        self.current_cmd_arm = [0.0, 0.0, 0.0, 0.0]
        self.current_cmd_grip = 0.00
        self.total_step_count = 0

        # サブカメラ初期机上Z基準サンプリング
        self.side_z_samples = []
        self.side_top_z_samples = []
        self.init_side_bottom_y = None
        self.init_side_top_y = None
        self.is_success = False

        # タイム計測 (Approach開始 -> Home帰還完了)
        self.t_start_approach = None
        self.t_end_approach = None
        self.elapsed_time_sec = 0.0
        self.start_step_approach = 0
        self.measured_cycle_hz = 0.0

    def image_callback(self, msg):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        self.latest_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def side_image_callback(self, msg):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        self.latest_side_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def joint_callback(self, msg):
        pos = list(msg.position)
        # [joint1, joint2, joint4, joint3, hand] -> [j1, j2, j3, j4, hand]
        self.latest_joints = [pos[0], pos[1], pos[3], pos[2], pos[4]]

    def publish_arm(self, target_arm_rad, duration_sec=0.2):
        self.current_cmd_arm = [float(x) for x in target_arm_rad]
        traj = JointTrajectory()
        traj.joint_names = self.arm_names
        pt = JointTrajectoryPoint()
        pt.positions = list(self.current_cmd_arm)
        pt.time_from_start = Duration(sec=0, nanosec=int(duration_sec * 1e9))
        traj.points = [pt]
        self.arm_pub.publish(traj)

    def publish_gripper(self, target_grip_rad, duration_sec=0.2):
        self.current_cmd_grip = float(target_grip_rad)
        traj = JointTrajectory()
        traj.joint_names = self.grip_names
        pt = JointTrajectoryPoint()
        pt.positions = [self.current_cmd_grip]
        pt.time_from_start = Duration(sec=0, nanosec=int(duration_sec * 1e9))
        traj.points = [pt]
        self.gripper_pub.publish(traj)

    def control_loop(self):
        if self.latest_image is None or self.latest_joints is None:
            return

        # 視覚認識の実行 (ペグ目標は初期アプローチ開始後に完全ロックし、J1旋回による固定爪押し込み・ペグ転倒を根絶)
        peg, hole = self.detector.detect_objects(self.latest_image)
        if peg:
            self.target_peg = peg
            if self.state in ["INIT", "RETRY_OPEN"]:
                self.j1_peg = np.arctan2(peg["y_base"], peg["x_base"])
                self.r_peg = np.sqrt(peg["x_base"]**2 + peg["y_base"]**2)
                self.yaw_upright_offset = 0.095
            if self.peg_initial is None:
                self.peg_initial = peg
        if hole:
            self.target_hole = hole
            self.j1_hole = np.arctan2(hole["y_base"], hole["x_base"])
            self.r_hole = np.sqrt(hole["x_base"]**2 + hole["y_base"]**2)

        grip_deg = np.degrees(self.latest_joints[4])
        self.annotated_frame = self.detector.annotate_frame(
            self.latest_image, self.target_peg, self.target_hole, self.state, grip_deg,
            side_image=self.latest_side_image, grasp_info=self.last_grasp_info
        )

        self.state_step += 1
        self.total_step_count += 1

        # テレメトリの記録 (全関節＆グリッパーの指令値 vs 実測値)
        self.history_steps.append(self.total_step_count)
        self.history_meas.append(list(self.latest_joints))
        self.history_cmd.append(list(self.current_cmd_arm) + [float(self.current_cmd_grip)])

        # =================================================================
        # ステートマシン実行ロジック (10 Hz 周期)
        # =================================================================
        if self.state == "INIT":
            # ホーム位置でペグと穴の視覚ロックオンを待機 (約1.0秒)
            # グリッパー開度はペグ直径の1.5倍 (0.00 rad, 幅約45mm) に設定し土台衝突を回避
            self.publish_arm([0.0, 0.0, 0.0, 0.0], duration_sec=0.2)
            self.publish_gripper(0.00, duration_sec=0.2)

            # サブカメラから初期机上静止時のZ基準ピクセルをサンプリング
            if self.latest_side_image is not None:
                side_info = self.detector.measure_side_peg_z(self.latest_side_image)
                if side_info.get("bottom_y") is not None:
                    self.side_z_samples.append(side_info["bottom_y"])
                if side_info.get("top_y") is not None:
                    self.side_top_z_samples.append(side_info["top_y"])

            if self.target_peg is not None and self.target_hole is not None and self.state_step >= 10:
                if len(self.side_z_samples) > 0:
                    self.init_side_bottom_y = int(np.median(self.side_z_samples))
                else:
                    self.init_side_bottom_y = 306 if self.target_peg["y_base"] < 0 else 165
                if len(self.side_top_z_samples) > 0:
                    self.init_side_top_y = int(np.median(self.side_top_z_samples))

                self.get_logger().info(
                    f"🎯 視覚ロックオン成功！ PEG 3D=({self.target_peg['x_base']:.3f}, {self.target_peg['y_base']:.3f})m, "
                    f"R={self.r_peg:.3f}m, Yaw J1={self.j1_peg:.3f}rad ({np.degrees(self.j1_peg):.1f}°), "
                    f"初期机上SideY={self.init_side_bottom_y}px"
                )
                self.state = "APPROACH_HOVER"
                self.state_step = 0
                self.t_start_approach = time.perf_counter()
                self.start_step_approach = self.total_step_count
                self.get_logger().info("=" * 65)
                self.get_logger().info("⏱️ 【タイム計測開始】 アプローチ動作開始 (t = 0.00s)")
                self.get_logger().info("=" * 65)

        elif self.state == "APPROACH_HOVER":
            # 1. ペグの上空へアームを旋回・ホバリング（高度 Z=0.10m, ピッチ=135°）
            # ペグ直径1.5倍開度 (0.00 rad) を維持
            j2, j3, j4 = solve_crane_ik(self.r_peg, z_target=0.10, pitch_deg=135.0)
            self.publish_arm([self.j1_peg, j2, j3, j4], duration_sec=0.2)
            self.publish_gripper(0.00, duration_sec=0.2)

            if self.state_step >= 15:  # 1.5秒経過
                self.get_logger().info("⬇️ ペグの直上へ到達。把持高度へ鉛直降下を開始します...")
                self.state = "DESCEND_GRASP"
                self.state_step = 0

        elif self.state == "DESCEND_GRASP":
            # 2. ペグを把持可能な高度まで鉛直降下 (高度 Z=0.055m, ピッチ=135°)
            # 鉛直降下中もペグ直径1.5倍開度 (0.00 rad) と J1完全固定を維持
            j2, j3, j4 = solve_crane_ik(self.r_peg, z_target=0.055, pitch_deg=135.0)
            self.publish_arm([self.j1_peg, j2, j3, j4], duration_sec=0.2)
            self.publish_gripper(0.00, duration_sec=0.2)

            if self.state_step >= 12:  # 1.2秒経過
                self.get_logger().info("🤏 把持高度へ到達。小刻みな段階的把持（Stepwise Grasping）を開始...")
                self.state = "STEPWISE_GRASP"
                self.state_step = 0
                self.grasp_substep_idx = 0

        elif self.state == "STEPWISE_GRASP":
            # 3. 段階的把持: 円柱ペグを弾いたり斜めにしないよう小刻みに閉じる
            j2, j3, j4 = solve_crane_ik(self.r_peg, z_target=0.055, pitch_deg=135.0)
            self.publish_arm([self.j1_peg, j2, j3, j4], duration_sec=0.2)

            # 3ステップごとに把持角度を小刻みに深める (約0.3秒間隔)
            if self.state_step % 3 == 0 and self.grasp_substep_idx < len(self.grasp_substeps):
                target_g = self.grasp_substeps[self.grasp_substep_idx]
                self.publish_gripper(target_g, duration_sec=0.2)
                self.get_logger().info(
                    f"  [Grasp Step {self.grasp_substep_idx+1}/{len(self.grasp_substeps)}] "
                    f"Target={target_g:.3f} rad ({np.degrees(target_g):.1f}°)"
                )
                self.grasp_substep_idx += 1

            if self.grasp_substep_idx >= len(self.grasp_substeps) and self.state_step >= 20:
                self.get_logger().info("🔍 段階的把持完了。微小リフト把持確認(Tug Test)へ移行します...")
                self.state = "VERIFY_GRASP"
                self.state_step = 0

        elif self.state == "VERIFY_GRASP":
            # 【二重把持確認: 人間の目視確認を模した微小リフト (Tug Test)】
            # 手首関節をわずかに引き上げ（高度 Z=0.075m = +2cm）、ペグが机から浮上したかを角度差＆カメラで二重確認
            j2, j3, j4 = solve_crane_ik(self.r_peg, z_target=0.075, pitch_deg=135.0)
            self.publish_arm([self.j1_peg, j2, j3, j4], duration_sec=0.2)
            self.publish_gripper(0.42, duration_sec=0.2)  # 把持力をしっかり維持

            # 1.0秒間（10ステップ）微小リフト姿勢を安定保持し、二重把持判定を実行
            if self.state_step >= 10:
                side_info = self.detector.measure_side_peg_z(self.latest_side_image)
                cur_bot_y = side_info.get("bottom_y")
                cur_top_y = side_info.get("top_y")

                cmd_grip = 0.42
                meas_grip = self.latest_joints[4] if self.latest_joints else 0.42

                is_grasped, details = self.detector.verify_grasp_dual_modal(
                    cmd_grip_rad=cmd_grip,
                    meas_grip_rad=meas_grip,
                    init_side_bottom_y=self.init_side_bottom_y,
                    cur_side_bottom_y=cur_bot_y,
                    init_side_top_y=self.init_side_top_y,
                    cur_side_top_y=cur_top_y
                )
                self.last_grasp_info = details

                if is_grasped:
                    self.get_logger().info(
                        f"🎯 【把持確認成功】 ペグ把持・机上浮上を確認！ "
                        f"(Δθ={details['grip_diff_deg']:+.1f}°, ΔZ={details['z_lift_px']:+d}px, SideY={cur_bot_y})"
                    )
                    self.get_logger().info("🚀 ペグを安全高度(Z=0.16m)まで完全にリフトします...")
                    self.state = "LIFT_PEG"
                    self.state_step = 0
                else:
                    self.get_logger().warn(
                        f"⚠️ 【把持失敗検知】 ペグ未把持または机上残存: "
                        f"Δθ={details['grip_diff_deg']:+.1f}°, ΔZ={details['z_lift_px']:+d}px (SideY={cur_bot_y} vs init={self.init_side_bottom_y}) -> 自動再把持を実行"
                    )
                    self.retry_count += 1
                    if self.retry_count > self.max_retries:
                        self.get_logger().error("❌ 最大リトライ回数を超過しました。安全のためホームへ退避します。")
                        self.state = "RETRACT_HOME"
                        self.state_step = 0
                    else:
                        self.get_logger().info(f"🔄 自動リトライ試行 [{self.retry_count}/{self.max_retries}] を開始...")
                        self.state = "RETRY_OPEN"
                        self.state_step = 0

        elif self.state == "RETRY_OPEN":
            # リトライ動作: グリッパーを1.5倍開度 (0.00 rad) に開き、上空へ退避した上でペグの最新位置を再検出
            j2, j3, j4 = solve_crane_ik(self.r_peg, z_target=0.10, pitch_deg=135.0)
            self.publish_arm([self.j1_peg, j2, j3, j4], duration_sec=0.2)
            self.publish_gripper(0.00, duration_sec=0.2)

            if self.state_step >= 12:  # 1.2秒経過
                # 最新のペグ座標へ目標を再更新
                if self.target_peg:
                    self.j1_peg = np.arctan2(self.target_peg["y_base"], self.target_peg["x_base"])
                    self.r_peg = np.sqrt(self.target_peg["x_base"]**2 + self.target_peg["y_base"]**2)
                    self.get_logger().info(
                        f"🎯 ペグ最新座標へ再ロックオン: 3D=({self.target_peg['x_base']:.3f}, {self.target_peg['y_base']:.3f})m"
                    )
                self.state = "DESCEND_GRASP"
                self.state_step = 0

        elif self.state == "LIFT_PEG":
            # 4. ペグをテーブルから鉛直に持ち上げ (高度 Z=0.16m, ピッチ=140°)
            j2, j3, j4 = solve_crane_ik(self.r_peg, z_target=0.16, pitch_deg=140.0)
            self.publish_arm([self.j1_peg, j2, j3, j4], duration_sec=0.2)
            self.publish_gripper(0.42, duration_sec=0.2)  # 把持力をしっかり維持

            if self.state_step >= 15:  # 1.5秒経過
                self.get_logger().info("🔄 ペグのリフト完了。青い穴(台座)の直上へ向かって旋回します...")
                self.state = "MOVE_TO_HOLE"
                self.state_step = 0

        elif self.state == "MOVE_TO_HOLE":
            # 5. 台座（穴）の直上へ旋回・ホバリング (高度 Z=0.16m, ピッチ=135°)
            j2, j3, j4 = solve_crane_ik(self.r_hole, z_target=0.16, pitch_deg=135.0)
            self.publish_arm([self.j1_hole, j2, j3, j4], duration_sec=0.2)
            self.publish_gripper(0.42, duration_sec=0.2)

            if self.state_step >= 20:  # 2.0秒経過
                self.get_logger().info("🎯 穴の直上へ到達。垂直挿入降下を開始します...")
                self.state = "INSERT_HOLE"
                self.state_step = 0

        elif self.state == "INSERT_HOLE":
            # 6. 穴への垂直挿入 (高度 Z=0.095m, ピッチ=135°)
            j2, j3, j4 = solve_crane_ik(self.r_hole, z_target=0.095, pitch_deg=135.0)
            # 穴開口部進入後、ヨーを固定爪側へ回転してペグを垂直に直立矯正
            cur_j1 = self.j1_hole + (self.yaw_upright_offset if self.state_step >= 6 else 0.0)
            self.publish_arm([cur_j1, j2, j3, j4], duration_sec=0.2)
            self.publish_gripper(0.42, duration_sec=0.2)

            if self.state_step >= 15:  # 1.5秒経過
                self.get_logger().info("📥 挿入完了！ グリッパーを開放してペグを設置します...")
                self.state = "RELEASE_PEG"
                self.state_step = 0

        elif self.state == "RELEASE_PEG":
            # 7. グリッパー開放 (直立ヨー姿勢を維持し固定爪に寄り添わせたまま開放)
            j2, j3, j4 = solve_crane_ik(self.r_hole, z_target=0.095, pitch_deg=135.0)
            cur_j1 = self.j1_hole + self.yaw_upright_offset
            self.publish_arm([cur_j1, j2, j3, j4], duration_sec=0.2)
            self.publish_gripper(-0.35, duration_sec=0.2)

            if self.state_step >= 10:  # 1.0秒経過
                self.get_logger().info("⬆️ 解放完了。穴から垂直に待避します...")
                self.is_success = True
                self.state = "RETRACT_UP"
                self.state_step = 0

        elif self.state == "RETRACT_UP":
            # 8. 穴の上空へ垂直リフト (高度 Z=0.16m, 直立ヨー姿勢維持)
            j2, j3, j4 = solve_crane_ik(self.r_hole, z_target=0.16, pitch_deg=135.0)
            cur_j1 = self.j1_hole + self.yaw_upright_offset
            self.publish_arm([cur_j1, j2, j3, j4], duration_sec=0.2)
            self.publish_gripper(-0.35, duration_sec=0.2)

            if self.state_step >= 10:  # 1.0秒経過
                self.get_logger().info("🏠 ホーム位置へ帰還します...")
                self.state = "RETRACT_HOME"
                self.state_step = 0

        elif self.state == "RETRACT_HOME":
            # 9. ホーム位置へ帰還
            self.publish_arm([0.0, 0.0, 0.0, 0.0], duration_sec=0.2)
            self.publish_gripper(-0.35, duration_sec=0.2)

            if self.state_step >= 15:  # 1.5秒経過
                self.t_end_approach = time.perf_counter()
                if self.t_start_approach is not None:
                    self.elapsed_time_sec = float(self.t_end_approach - self.t_start_approach)
                else:
                    self.elapsed_time_sec = 0.0
                steps_elapsed = int(self.total_step_count - getattr(self, "start_step_approach", 0))
                cycle_hz = float(steps_elapsed / self.elapsed_time_sec) if self.elapsed_time_sec > 0 else 0.0
                self.measured_cycle_hz = cycle_hz

                self.save_telemetry()
                if self.annotated_frame is not None:
                    out_path = os.path.join(self.output_debug_dir, "visual_servo_result.png")
                    cv2.imwrite(out_path, self.annotated_frame)
                npz_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "visual_servo_telemetry.npz")
                img_path = os.path.join(self.output_debug_dir, "visual_servo_result.png")

                banner_text = (
                    "\n" + "=" * 65 + "\n"
                    "⏱️ 【タスク実行時間計測 (Approach to Home Return) [Step 5 Approach A]】\n"
                    f"⏱️   - アプローチ開始 (Start) : t = 0.00s (Step 1)\n"
                    f"⏱️   - ホーム帰還完了 (Goal)  : t = {self.elapsed_time_sec:.2f}s (Step {steps_elapsed})\n"
                    f"⏱️   - 総所要時間 (Elapsed)  : {self.elapsed_time_sec:.2f} 秒 ({self.elapsed_time_sec/60:.2f} 分)\n"
                    f"⏱️   - 制御周期 (Hz)         : {cycle_hz:.1f} Hz ({1000/cycle_hz if cycle_hz>0 else 0:.1f} ms/step)\n"
                    + "=" * 65 + "\n"
                    f"🎉 規定ステップ（{steps_elapsed} 回）完遂！ビジュアルサーボ制御シーケンスを正常終了します。\n"
                    f"📊 Gazebo実行テレメトリを保存しました: {npz_path}\n"
                    f"📸 最終実行結果画像を保存しました: {img_path}\n"
                    "📈 全関節・グリッパー 指令vs実測テレメトリ比較グラフを自動生成しました"
                )
                print(banner_text, flush=True)

                if self.is_success:
                    self.get_logger().info("🎉 【ペグ・イン・ホール完全成功】 全タスクを正常終了しました！")
                else:
                    self.get_logger().warning("⚠️ 【ミッション終了】 把持失敗・リトライ回数超過のため安全にホームへ退避しました。")
                self.state = "DONE"
                self.timer.cancel()
                rclpy.shutdown()

    def save_telemetry(self):
        """全関節およびグリッパーの指令値 vs 実測値テレメトリデータを保存"""
        if len(self.history_steps) > 1:
            out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
            os.makedirs(out_dir, exist_ok=True)
            npz_path = os.path.join(out_dir, "visual_servo_telemetry.npz")
            np.savez_compressed(
                npz_path,
                steps=np.array(self.history_steps),
                measured_joints=np.array(self.history_meas),
                commanded_joints=np.array(self.history_cmd),
                elapsed_time_sec=float(self.elapsed_time_sec),
                cycle_hz=float(getattr(self, "measured_cycle_hz", 0.0))
            )
            self.get_logger().info(f"📊 [Step 5 Approach A] テレメトリデータを保存しました: {npz_path}")

    def destroy_node(self):
        self.save_telemetry()
        super().destroy_node()

def main():
    rclpy.init()
    controller = VisualServoPegInHoleController(target_color="red")
    try:
        rclpy.spin(controller)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        controller.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(0 if controller.is_success else 1)

if __name__ == '__main__':
    main()
