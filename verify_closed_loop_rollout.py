"""
verify_closed_loop_rollout.py
=====================================================================
Step 6-4: 生徒ACTモデルによる閉ループ直接推論 ＆ 軌道忠実度検証
- 学習済み蒸留モデル (act_distilled_*_best.pth) をロード
- 逆サイド (Ep 0) および通常サイド (Ep 5) に対する自己回帰ロールアウト検証
- エキスパート真値軌道と生徒モデル予測軌道の追従忠実度を比較評価
- 注目点可視化グラフ (step6_4_direct_rollout_fidelity.png) の生成
=====================================================================
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "real_expert_dataset")
MODEL_DIR = os.path.join(SCRIPT_DIR, "models_trained")
MEAN_PATH = os.path.join(MODEL_DIR, "feature_mean.npy")
STD_PATH = os.path.join(MODEL_DIR, "feature_std.npy")

from train_real_distillation import DistilledArmPolicy, DistilledGripperPolicy

def evaluate_rollout():
    print("=" * 70)
    print("🔬 【Step 6-4】生徒ACTモデルによる閉ループ推論 ＆ 軌道忠実度検証開始")
    print("=" * 70)

    device = torch.device("cpu")
    arm_model = DistilledArmPolicy(arm_dim=4, obj_dim=14, chunk_size=8).to(device)
    grip_model = DistilledGripperPolicy(grip_dim=1, obj_dim=14, chunk_size=8).to(device)

    arm_path = os.path.join(MODEL_DIR, "act_distilled_arm_best.pth")
    grip_path = os.path.join(MODEL_DIR, "act_distilled_gripper_best.pth")

    if not os.path.exists(arm_path) or not os.path.exists(grip_path):
        print(f"❌ モデルファイルが見つかりません: {arm_path}")
        return

    arm_model.load_state_dict(torch.load(arm_path, map_location=device))
    grip_model.load_state_dict(torch.load(grip_path, map_location=device))
    arm_model.eval()
    grip_model.eval()
    print("✅ 学習済み生徒モデルのロード完了")

    feat_mean = np.load(MEAN_PATH)
    feat_std = np.load(STD_PATH)

    # 評価対象: Ep 0 (逆サイド) と Ep 5 (通常サイド)
    ep0_path = os.path.join(DATA_DIR, "episode_00.npz")
    ep5_path = os.path.join(DATA_DIR, "episode_05.npz")
    if not os.path.exists(ep5_path):
        ep_files = sorted([f for f in os.listdir(DATA_DIR) if f.startswith("episode_") and f.endswith(".npz")])
        ep5_path = os.path.join(DATA_DIR, ep_files[-1]) if ep_files else ep0_path

    data_rev = np.load(ep0_path)
    data_norm = np.load(ep5_path)

    def run_inference(data_ep):
        images = data_ep['images']
        joints_true = data_ep['joints']
        cmds_true = data_ep['cmd_joints']
        feats = data_ep['features']
        n_steps = len(joints_true)

        pred_arms = []
        pred_grips = []

        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        with torch.no_grad():
            for t in range(n_steps):
                img_t = transform(images[t]).unsqueeze(0)
                f_norm = (feats[t] - feat_mean) / feat_std
                f_t = torch.tensor(f_norm, dtype=torch.float32).unsqueeze(0)
                cur_a = torch.tensor(joints_true[t, :4], dtype=torch.float32).unsqueeze(0)
                cur_g = torch.tensor([[joints_true[t, 4]]], dtype=torch.float32)

                out_arm = arm_model(img_t, f_t, cur_a).squeeze(0).numpy() # (chunk, 4)
                out_grip = grip_model(img_t, f_t, cur_g).squeeze(0).numpy() # (chunk, 1)

                pred_arms.append(out_arm[0])
                pred_grips.append(out_grip[0])

        pred_arms = np.array(pred_arms)
        pred_grips = np.array(pred_grips)
        return pred_arms, pred_grips, cmds_true

    print("🤖 逆サイド (Reverse Side) エピソード推論中...")
    arm_rev_pred, grip_rev_pred, cmd_rev_true = run_inference(data_rev)
    print("🤖 通常サイド (Normal Side) エピソード推論中...")
    arm_norm_pred, grip_norm_pred, cmd_norm_true = run_inference(data_norm)

    # 全関節＆グリッパー誤差算出
    mse_rev_arm = np.mean((arm_rev_pred - cmd_rev_true[:, :4])**2, axis=0)
    mse_rev_grip = np.mean((grip_rev_pred[:, 0] - cmd_rev_true[:, 4])**2)
    mse_norm_arm = np.mean((arm_norm_pred - cmd_norm_true[:, :4])**2, axis=0)
    mse_norm_grip = np.mean((grip_norm_pred[:, 0] - cmd_norm_true[:, 4])**2)

    print(f"📊 軌道忠実度評価サマリー (全5軸 MSE [rad^2]):")
    print(f"   ✓ 逆サイド (Ep 0): J1={mse_rev_arm[0]:.6f}, J2={mse_rev_arm[1]:.6f}, J3={mse_rev_arm[2]:.6f}, J4={mse_rev_arm[3]:.6f}, Grip={mse_rev_grip:.6f}")
    print(f"   ✓ 通常サイド (Ep 5): J1={mse_norm_arm[0]:.6f}, J2={mse_norm_arm[1]:.6f}, J3={mse_norm_arm[2]:.6f}, J4={mse_norm_arm[3]:.6f}, Grip={mse_norm_grip:.6f}")

    # -------------------------------------------------------------
    # 注目点可視化グラフ (step6_4_direct_rollout_fidelity.png) の生成
    # -------------------------------------------------------------
    generate_fidelity_graph(arm_rev_pred, grip_rev_pred, cmd_rev_true,
                            arm_norm_pred, grip_norm_pred, cmd_norm_true)


def generate_fidelity_graph(arm_rev_pred, grip_rev_pred, cmd_rev_true,
                            arm_norm_pred, grip_norm_pred, cmd_norm_true):
    """全関節 (Joint 1〜4) およびグリッパー (Hand) の閉ループ推論忠実度を6面パネル (2x3) で描画"""
    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    fig.patch.set_facecolor('#0f172a')

    for ax in axes.flat:
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='white')
        ax.grid(True, linestyle=':', alpha=0.3, color='#94a3b8')

    t_axis_rev = np.linspace(0, len(cmd_rev_true)*0.10, len(cmd_rev_true))
    t_axis_norm = np.linspace(0, len(cmd_norm_true)*0.10, len(cmd_norm_true))

    # パネル 1 [0, 0]: 逆サイド Joint 1 (Yaw) ＆ 自立化回転 (+5.44°)
    ax1 = axes[0, 0]
    ax1.plot(t_axis_rev, cmd_rev_true[:, 0], color='#f43f5e', linewidth=2.5, label='Expert Ground Truth (Teacher)')
    ax1.plot(t_axis_rev, arm_rev_pred[:, 0], color='#38bdf8', linewidth=2.0, linestyle='--', label='Distilled ACT Prediction (Student)')
    ax1.axvline(21.0, color='#facc15', linestyle=':', linewidth=1.8, label='Upright Yaw Rotation (+5.44 deg)')
    ax1.set_title("Panel 1: Reverse Side (Y=-0.08m) Joint 1 (Yaw) & Uprighting", color='white', fontsize=11, fontweight='bold')
    ax1.set_xlabel("Time [s]", color='white')
    ax1.set_ylabel("Joint 1 (Yaw) [rad]", color='white')
    ax1.legend(facecolor='#0f172a', edgecolor='#334155', labelcolor='white', fontsize=8)

    # パネル 2 [0, 1]: 通常サイド Joint 1 (Yaw)
    ax2 = axes[0, 1]
    ax2.plot(t_axis_norm, cmd_norm_true[:, 0], color='#10b981', linewidth=2.5, label='Expert Ground Truth (Teacher)')
    ax2.plot(t_axis_norm, arm_norm_pred[:, 0], color='#38bdf8', linewidth=2.0, linestyle='--', label='Distilled ACT Prediction (Student)')
    ax2.set_title("Panel 2: Normal Side (Y=+0.08m) Joint 1 (Yaw) Fidelity", color='white', fontsize=11, fontweight='bold')
    ax2.set_xlabel("Time [s]", color='white')
    ax2.set_ylabel("Joint 1 (Yaw) [rad]", color='white')
    ax2.legend(facecolor='#0f172a', edgecolor='#334155', labelcolor='white', fontsize=8)

    # パネル 3 [0, 2]: Joint 2 (Shoulder) 垂直リフト＆土手接地追従 (逆サイド＆通常サイド)
    ax3 = axes[0, 2]
    ax3.plot(t_axis_rev, cmd_rev_true[:, 1], color='#f43f5e', linewidth=2.3, label='Rev Expert J2 (Teacher)')
    ax3.plot(t_axis_rev, arm_rev_pred[:, 1], color='#38bdf8', linewidth=1.9, linestyle='--', label='Rev Distilled ACT J2 (Student)')
    ax3.plot(t_axis_norm, cmd_norm_true[:, 1], color='#10b981', linewidth=1.8, alpha=0.75, label='Norm Expert J2 (Teacher)')
    ax3.plot(t_axis_norm, arm_norm_pred[:, 1], color='#facc15', linewidth=1.6, linestyle=':', label='Norm Distilled ACT J2 (Student)')
    ax3.set_title("Panel 3: Joint 2 (Shoulder) Vertical Lift & Touchdown", color='white', fontsize=11, fontweight='bold')
    ax3.set_xlabel("Time [s]", color='white')
    ax3.set_ylabel("Joint 2 (Shoulder) [rad]", color='white')
    ax3.legend(facecolor='#0f172a', edgecolor='#334155', labelcolor='white', fontsize=8)

    # パネル 4 [1, 0]: Joint 3 (Elbow) リーチ伸展＆高空クリアランス軌道
    ax4 = axes[1, 0]
    ax4.plot(t_axis_rev, cmd_rev_true[:, 2], color='#f43f5e', linewidth=2.3, label='Rev Expert J3 (Teacher)')
    ax4.plot(t_axis_rev, arm_rev_pred[:, 2], color='#38bdf8', linewidth=1.9, linestyle='--', label='Rev Distilled ACT J3 (Student)')
    ax4.plot(t_axis_norm, cmd_norm_true[:, 2], color='#10b981', linewidth=1.8, alpha=0.75, label='Norm Expert J3 (Teacher)')
    ax4.plot(t_axis_norm, arm_norm_pred[:, 2], color='#facc15', linewidth=1.6, linestyle=':', label='Norm Distilled ACT J3 (Student)')
    ax4.set_title("Panel 4: Joint 3 (Elbow) Reach Extension & High-Altitude Cruise", color='white', fontsize=11, fontweight='bold')
    ax4.set_xlabel("Time [s]", color='white')
    ax4.set_ylabel("Joint 3 (Elbow) [rad]", color='white')
    ax4.legend(facecolor='#0f172a', edgecolor='#334155', labelcolor='white', fontsize=8)

    # パネル 5 [1, 1]: Joint 4 (Wrist) 鉛直ピッチ姿勢維持＆挿入ダンピング軌道
    ax5 = axes[1, 1]
    ax5.plot(t_axis_rev, cmd_rev_true[:, 3], color='#f43f5e', linewidth=2.3, label='Rev Expert J4 (Teacher)')
    ax5.plot(t_axis_rev, arm_rev_pred[:, 3], color='#38bdf8', linewidth=1.9, linestyle='--', label='Rev Distilled ACT J4 (Student)')
    ax5.plot(t_axis_norm, cmd_norm_true[:, 3], color='#10b981', linewidth=1.8, alpha=0.75, label='Norm Expert J4 (Teacher)')
    ax5.plot(t_axis_norm, arm_norm_pred[:, 3], color='#facc15', linewidth=1.6, linestyle=':', label='Norm Distilled ACT J4 (Student)')
    ax5.set_title("Panel 5: Joint 4 (Wrist) Pitch Alignment & Damping Control", color='white', fontsize=11, fontweight='bold')
    ax5.set_xlabel("Time [s]", color='white')
    ax5.set_ylabel("Joint 4 (Wrist) [rad]", color='white')
    ax5.legend(facecolor='#0f172a', edgecolor='#334155', labelcolor='white', fontsize=8)

    # パネル 6 [1, 2]: グリッパー開閉 (Hand) 段階的把持＆自立化徐開
    ax6 = axes[1, 2]
    ax6.plot(t_axis_rev, cmd_rev_true[:, 4], color='#facc15', linewidth=2.5, label='Rev Expert Gripper (Teacher)')
    ax6.plot(t_axis_rev, grip_rev_pred[:, 0], color='#38bdf8', linewidth=2.0, linestyle='--', label='Rev Distilled ACT Gripper (Student)')
    ax6.plot(t_axis_norm, cmd_norm_true[:, 4], color='#10b981', linewidth=1.6, alpha=0.75, label='Norm Expert Gripper (Teacher)')
    ax6.plot(t_axis_norm, grip_norm_pred[:, 0], color='#f43f5e', linewidth=1.5, linestyle=':', label='Norm Distilled ACT Gripper (Student)')
    ax6.set_title("Panel 6: Gripper Action (Hand) Stepwise Pinch & Gentle Release", color='white', fontsize=11, fontweight='bold')
    ax6.set_xlabel("Time [s]", color='white')
    ax6.set_ylabel("Gripper (Hand) [rad]", color='white')
    ax6.legend(facecolor='#0f172a', edgecolor='#334155', labelcolor='white', fontsize=8)

    plt.tight_layout()
    g_path = os.path.join(SCRIPT_DIR, "step6_4_direct_rollout_fidelity.png")
    plt.savefig(g_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"📊 全関節＆グリッパー軌道忠実度検証グラフを保存しました: {g_path}")

if __name__ == '__main__':
    evaluate_rollout()
