"""
feature_utils.py
=====================================================================
14次元アフォーダンス特徴量ベクトルの統一計算ユーティリティ
- データ収集 (collect_expert_episodes.py)
- データセット前処理 (prepare_real_dataset.py)
- 学習データローダー (train_real_distillation.py)
- 実環境推論ロールアウト (autonomous_act_distilled.py)
上記すべてのモジュールで 100% 同一の次元並び順・数理定義を保証
=====================================================================
"""

import numpy as np

def compute_14dim_features(
    peg_x, peg_y,
    hole_x=0.200, hole_y=0.000,
    progress_tau=0.0,
    phase_metric=0.10,
    is_held=0.0
):
    """
    14次元アフォーダンス特徴量ベクトルを厳密に計算
    [0]: r_peg       (ペグ原点極距離 [m])
    [1]: th_peg      (ペグ原点偏角 [rad])
    [2]: peg_x       (ペグ3D X座標 [m])
    [3]: peg_y       (ペグ3D Y座標 [m])
    [4]: hole_x      (台座穴3D X座標 [m])
    [5]: hole_y      (台座穴3D Y座標 [m])
    [6]: dx          (穴-ペグ相対X変位 [m])
    [7]: dy          (穴-ペグ相対Y変位 [m])
    [8]: dist_peg    (原点-ペグ水平距離 [m])
    [9]: dist_hole   (原点-穴水平距離 [m])
    [10]: tau        (タスク連続進行度 0.0〜1.0)
    [11]: phase      (5大行動パターン指標 0.10, 0.30, 0.50, 0.75, 0.95)
    [12]: is_held    (把持確信度 0.0〜1.0)
    [13]: side_lift  (浮上確認フラグ 0.0 or 1.0)
    """
    r_p = float(np.sqrt(peg_x**2 + peg_y**2))
    th_p = float(np.arctan2(peg_y, peg_x))
    dx = float(hole_x - peg_x)
    dy = float(hole_y - peg_y)
    dist_p = r_p
    dist_h = float(np.sqrt(hole_x**2 + hole_y**2))
    side_lift = 1.0 if is_held > 0.5 else 0.0

    feat = np.array([
        r_p, th_p,
        float(peg_x), float(peg_y),
        float(hole_x), float(hole_y),
        dx, dy,
        dist_p, dist_h,
        float(progress_tau),
        float(phase_metric),
        float(is_held),
        float(side_lift)
    ], dtype=np.float32)

    return feat
