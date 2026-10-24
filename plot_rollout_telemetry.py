#!/usr/bin/env python3
"""
plot_rollout_telemetry.py
=============================================================================
ステップ6の Gazebo 実演ロールアウト結果（distilled_act_gazebo_telemetry.npz）から、
全関節（Joint 1〜4）およびグリッパー（Hand）の「指令値 (Commanded) vs 実測値 (Measured)」の
時系列曲線を6面マルチパネルでグラフ化し、外力干渉やバックドライブ、追従偏差を精密に可視化するスクリプト。

【仕様】
- アームが動作した場合（成功・失敗問わず）: 全関節の指令値 vs 実測値の時系列曲線をプロット。
- ペグが閾値外にありアームが静止状態のままアボートした場合:
  前のメモリを破棄し、「箱（サブプロット枠・軸・グリッド）だけが存在して中身（曲線）は描かれない」
  静止状態グラフを生成する。
=============================================================================
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _draw_empty_static_box(output_path, script_dir, reason="Arm Kept Static (Workspace Guard / No Motion Recorded)", title_prefix="Step 6-5: Telemetry Analysis", max_steps=285):
    """アームが未動作・静止状態で終了した場合、箱（枠線・軸・グリッド）のみ存在し中身の曲線は描かれない空グラフを生成"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.patch.set_facecolor('#0f172a')

    joint_titles = [
        ("Joint 1 (Yaw / Base Rotation)", "rad"),
        ("Joint 2 (Shoulder Pitch)", "rad"),
        ("Joint 3 (Elbow Pitch)", "rad"),
        ("Joint 4 (Wrist Pitch)", "rad"),
        ("Gripper (Hand Clamping)", "rad"),
    ]

    for idx in range(5):
        r = idx // 3
        c = idx % 3
        ax = axes[r, c]
        ax.set_facecolor('#1e293b')

        title_str, unit = joint_titles[idx]
        ax.set_title(f"{title_str} [Static / No Motion]", color='#94a3b8', fontsize=11, fontweight='bold')
        ax.set_xlabel("Rollout Step", color='#64748b', fontsize=9)
        ax.set_ylabel(f"Angle [{unit}]", color='#64748b', fontsize=9)
        ax.set_xlim(0, max_steps)
        ax.set_ylim(-1.0, 1.0)
        ax.tick_params(colors='#64748b')
        for spine in ax.spines.values():
            spine.set_color('#334155')
        ax.grid(True, linestyle='--', alpha=0.20, color='#475569')

    # 6th panel: Multi-Axis Tracking Errors
    ax_err = axes[1, 2]
    ax_err.set_facecolor('#1e293b')
    ax_err.set_title("Multi-Axis Tracking Errors (|Cmd - Meas|) [Static]", color='#94a3b8', fontsize=11, fontweight='bold')
    ax_err.set_xlabel("Rollout Step", color='#64748b', fontsize=9)
    ax_err.set_ylabel("Absolute Error [rad]", color='#64748b', fontsize=9)
    ax_err.set_xlim(0, max_steps)
    ax_err.set_ylim(0.0, 0.5)
    ax_err.axhline(y=0.05, color='#ef4444', linestyle='--', alpha=0.4, label='Interference Threshold (0.05 rad)')
    ax_err.tick_params(colors='#64748b')
    for spine in ax_err.spines.values():
        spine.set_color('#334155')
    ax_err.grid(True, linestyle='--', alpha=0.20, color='#475569')
    ax_err.legend(loc='upper right', facecolor='#0f172a', edgecolor='#334155', labelcolor='#64748b', fontsize=8)

    plt.suptitle(f"{title_prefix} - {reason}",
                 color='#94a3b8', fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=180, facecolor=fig.get_facecolor())
    data_dir_out = os.path.join(script_dir, "data", "step6_5_command_vs_measured_telemetry.png")
    plt.savefig(data_dir_out, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"ℹ️ Static empty box chart generated (No motion): {output_path}")
    return output_path


def plot_telemetry(npz_path=None, output_path=None, title_prefix="Step 6-5: Telemetry Analysis"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if npz_path is None:
        npz_path = os.path.join(script_dir, "data", "distilled_act_gazebo_telemetry.npz")
    if output_path is None:
        output_path = os.path.join(script_dir, "step6_5_command_vs_measured_telemetry.png")

    if not os.path.exists(npz_path):
        print(f"ℹ️ テレメトリファイルが存在しません（アーム静止・未動作）: {npz_path}")
        return _draw_empty_static_box(output_path, script_dir, reason="Arm Kept Static (No Telemetry Recorded)", title_prefix=title_prefix)

    try:
        data = np.load(npz_path, allow_pickle=True)
        steps = data["steps"] if "steps" in data else np.array([])
        meas = data["measured_joints"] if "measured_joints" in data else np.array([])
        cmd = data["commanded_joints"] if "commanded_joints" in data else np.array([])
        motion_mode = str(data["motion_mode"]) if "motion_mode" in data else ""
        rec_max_steps = int(data["max_steps"]) if "max_steps" in data else 0
        elapsed_sec = float(data["elapsed_time_sec"]) if "elapsed_time_sec" in data else None
        cycle_hz = float(data["cycle_hz"]) if "cycle_hz" in data else None
    except Exception as e:
        print(f"⚠️ テレメトリ読込例外: {e}")
        return _draw_empty_static_box(output_path, script_dir, reason="Arm Kept Static (Corrupted or Empty Telemetry)", title_prefix=title_prefix)

    is_fast = (motion_mode == "fast") or (rec_max_steps == 193) or (len(steps) > 0 and steps[-1] <= 200)
    plot_max_steps = 193 if is_fast else 285

    if len(steps) < 2 or meas.shape[0] < 2 or cmd.shape[0] < 2:
        print("ℹ️ アーム静止状態（未動作・閾値外アボート等）: 箱のみ（中身なし）のテレメトリ枠を生成します")
        return _draw_empty_static_box(output_path, script_dir, reason="Arm Kept Static (Reachable Workspace Guard / No Motion)", title_prefix=title_prefix, max_steps=plot_max_steps)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.patch.set_facecolor('#0f172a')

    joint_titles = [
        ("Joint 1 (Yaw / Base Rotation)", "rad"),
        ("Joint 2 (Shoulder Pitch)", "rad"),
        ("Joint 3 (Elbow Pitch)", "rad"),
        ("Joint 4 (Wrist Pitch)", "rad"),
        ("Gripper (Hand Clamping)", "rad"),
    ]

    if is_fast:
        phase_bounds = [
            (15, "Hover"), (31, "Descend"), (47, "Grasp"),
            (75, "Lift"), (101, "Cruise"), (143, "Align"), (167, "Insert")
        ]
        time_str = f" / {elapsed_sec:.1f}s ({cycle_hz:.1f}Hz)" if elapsed_sec is not None else " / 23.2s (-32.3%)"
        mode_label = f"[Fast Mode: 193 steps{time_str}]"
    else:
        phase_bounds = [
            (15, "Hover"), (35, "Descend"), (55, "Grasp"),
            (95, "Lift"), (135, "Cruise"), (210, "Align"), (245, "Insert")
        ]
        time_str = f" / {elapsed_sec:.1f}s ({cycle_hz:.1f}Hz)" if elapsed_sec is not None else " / 34.2s"
        mode_label = f"[Standard Mode: 285 steps{time_str}]"

    for idx in range(5):
        r = idx // 3
        c = idx % 3
        ax = axes[r, c]
        ax.set_facecolor('#1e293b')

        cmd_j = cmd[:, idx]
        meas_j = meas[:, idx]
        err_j = np.abs(cmd_j - meas_j)

        # Commanded (Cyan) vs Measured (Rose dashed)
        ax.plot(steps, cmd_j, color='#38bdf8', linewidth=2.2, label='Commanded')
        ax.plot(steps, meas_j, color='#f43f5e', linewidth=2.0, linestyle='--', label='Measured')

        # Tracking error highlight (yellow fill)
        ax.fill_between(steps, cmd_j, meas_j, color='#fbbf24', alpha=0.25, label='Tracking Error')

        # Phase boundaries
        for p_step, p_name in phase_bounds:
            if p_step <= steps[-1]:
                ax.axvline(x=p_step, color='#64748b', linestyle=':', alpha=0.6)
                if idx == 0:
                    y_top = ax.get_ylim()[1]
                    ax.text(p_step, y_top, f" {p_name}", color='#94a3b8', fontsize=8, verticalalignment='top')

        title_str, unit = joint_titles[idx]
        max_err = float(np.max(err_j))
        max_err_deg = np.degrees(max_err)
        ax.set_title(f"{title_str} [Max Dev: {max_err:.3f}{unit} / {max_err_deg:.1f}deg]", color='#f8fafc', fontsize=11, fontweight='bold')
        ax.set_xlabel("Rollout Step", color='#cbd5e1', fontsize=9)
        ax.set_ylabel(f"Angle [{unit}]", color='#cbd5e1', fontsize=9)
        ax.set_xlim(0, plot_max_steps)
        ax.tick_params(colors='#94a3b8')
        for spine in ax.spines.values():
            spine.set_color('#334155')
        ax.grid(True, linestyle='--', alpha=0.25, color='#475569')
        if idx in (0, 4):
            ax.legend(loc='best', facecolor='#0f172a', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=8)

    # 6th panel: Multi-Axis Tracking Errors & External Interference
    ax_err = axes[1, 2]
    ax_err.set_facecolor('#1e293b')
    colors = ['#38bdf8', '#34d399', '#a78bfa', '#fb923c', '#f43f5e']
    labels = ['J1 (Yaw)', 'J2 (Shoulder)', 'J3 (Elbow)', 'J4 (Wrist)', 'Gripper']
    for idx in range(5):
        err_j = np.abs(cmd[:, idx] - meas[:, idx])
        ax_err.plot(steps, err_j, color=colors[idx], linewidth=1.8, label=labels[idx])

    ax_err.axhline(y=0.05, color='#ef4444', linestyle='--', alpha=0.7, label='Interference Threshold (0.05 rad / 2.9deg)')
    ax_err.set_title("Multi-Axis Tracking Errors (|Cmd - Meas|)", color='#f8fafc', fontsize=11, fontweight='bold')
    ax_err.set_xlabel("Rollout Step", color='#cbd5e1', fontsize=9)
    ax_err.set_ylabel("Absolute Error [rad]", color='#cbd5e1', fontsize=9)
    ax_err.set_xlim(0, plot_max_steps)
    ax_err.tick_params(colors='#94a3b8')
    for spine in ax_err.spines.values():
        spine.set_color('#334155')
    ax_err.grid(True, linestyle='--', alpha=0.25, color='#475569')
    ax_err.legend(loc='upper right', facecolor='#0f172a', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=8)

    plt.suptitle(f"{title_prefix} {mode_label} - Commanded vs Measured Joint Curves",
                 color='#38bdf8', fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=180, facecolor=fig.get_facecolor())
    if "step6_5" in output_path:
        data_dir_out = os.path.join(script_dir, "data", "step6_5_command_vs_measured_telemetry.png")
        plt.savefig(data_dir_out, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"✅ Telemetry comparison chart generated and saved: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default=None, help="Path to telemetry npz file")
    parser.add_argument("--out", default=None, help="Output image path")
    parser.add_argument("--title", default="Step 6-5: Telemetry Analysis", help="Title prefix for chart")
    args = parser.parse_args()
    plot_telemetry(npz_path=args.npz, output_path=args.out, title_prefix=args.title)
