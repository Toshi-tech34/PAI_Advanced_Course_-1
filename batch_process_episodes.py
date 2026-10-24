import os
import glob
import cv2
import numpy as np
import sqlite3
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image, JointState

def process_all_episodes(raw_bags_dir="/workspace/提出物/data/raw_bags", output_dir="/workspace/提出物/data/processed_dataset", window_size=5):
    print("="*60)
    print("🔄 全エピソード一括処理・同期・マスキングパイプライン")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    bag_dirs = sorted(glob.glob(os.path.join(raw_bags_dir, "*")))
    valid_episodes = []
    
    for b in bag_dirs:
        db_files = glob.glob(os.path.join(b, "*.db3"))
        if not db_files:
            continue
        db_file = db_files[0]
        try:
            conn = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
            c = conn.cursor()
            c.execute("SELECT topics.name, count(messages.id) FROM messages JOIN topics ON messages.topic_id = topics.id GROUP BY topics.name")
            counts = dict(c.fetchall())
            conn.close()
            if counts.get("/image_raw", 0) > 10 and counts.get("/joint_states", 0) > 10:
                valid_episodes.append((b, db_file, counts.get("/image_raw", 0), counts.get("/joint_states", 0)))
            else:
                print(f"⚠️ スキップ ({os.path.basename(b)}): 画像または関節データが不足しています {counts}")
        except Exception as e:
            print(f"⚠️ スキップ ({os.path.basename(b)}): DB読み込みエラー {e}")

    print(f"\n🎯 処理対象となる有効エピソード数: {len(valid_episodes)}")
    for ep, _, img_cnt, j_cnt in valid_episodes:
        print(f"  - {os.path.basename(ep)} (画像: {img_cnt}枚, 関節: {j_cnt}件)")

    if len(valid_episodes) == 0:
        print("❌ 処理可能なエピソード（画像＋関節データ）が見つかりません。")
        return

    all_episode_data = []

    for ep_idx, (b, db_file, _, _) in enumerate(valid_episodes):
        ep_name = os.path.basename(b)
        print(f"\n📦 [{ep_idx+1}/{len(valid_episodes)}] 処理中: {ep_name}")

        try:
            # metadata.yaml の有無に左右されないよう db3 直接指定でオープン
            storage_options = StorageOptions(uri=db_file, storage_id="sqlite3")
            converter_options = ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
            reader = SequentialReader()
            reader.open(storage_options, converter_options)
        except Exception as err:
            print(f"  ❌ DBオープン失敗 ({ep_name}): {err}")
            continue

        raw_images = []
        raw_joints = []
        step = 0

        while reader.has_next():
            topic, data, ts = reader.read_next()
            if topic == "/image_raw":
                step += 1
                if step % 2 == 0: # 10Hz -> 5Hz
                    msg = deserialize_message(data, Image)
                    img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
                    raw_images.append((ts, img))
            elif topic == "/joint_states":
                msg = deserialize_message(data, JointState)
                pos = list(msg.position)
                joints = [pos[0], pos[1], pos[3], pos[2], pos[4]] # [j1, j2, j3, j4, hand]
                raw_joints.append((ts, joints))

        if len(raw_images) == 0 or len(raw_joints) == 0:
            print(f"  ⚠️ スキップ: メッセージが抽出できませんでした")
            continue

        joint_timestamps = np.array([j[0] for j in raw_joints])
        joint_values = np.array([j[1] for j in raw_joints])

        # 移動平均フィルタ
        filtered_joints = np.zeros_like(joint_values)
        for i in range(5):
            filtered_joints[:, i] = np.convolve(joint_values[:, i], np.ones(window_size)/window_size, mode='same')

        # 最近傍マッチング
        synced_images = []
        synced_raw_joints = []
        synced_filtered_joints = []

        for idx, (img_ts, img) in enumerate(raw_images):
            closest_idx = np.argmin(np.abs(joint_timestamps - img_ts))
            img_filename = f"{ep_name}_frame_{idx:05d}.png"
            img_path = os.path.join(images_dir, img_filename)
            cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            synced_images.append(img_filename)
            synced_raw_joints.append(joint_values[closest_idx])
            synced_filtered_joints.append(filtered_joints[closest_idx])

        synced_images = np.array(synced_images)
        synced_raw_joints = np.array(synced_raw_joints)
        synced_filtered_joints = np.array(synced_filtered_joints)

        # 駆動信号マスキング（静止ノイズ除去：アーム移動とグリッパー微調整を独立判定）
        arm_diffs = np.linalg.norm(np.diff(synced_filtered_joints[:, :4], axis=0), axis=1)
        grip_diffs = np.abs(np.diff(synced_filtered_joints[:, 4]))
        motion_mask = np.zeros(len(synced_filtered_joints), dtype=bool)
        for i in range(len(arm_diffs)):
            # アームの移動(>0.003 rad) または グリッパーの微小調整(>0.001 rad = 約0.06度)があれば保持
            if arm_diffs[i] > 0.003 or grip_diffs[i] > 0.001:
                start = max(0, i - 2)
                end = min(len(synced_filtered_joints), i + 3)
                motion_mask[start:end] = True

        active_indices = np.where(motion_mask)[0]
        if len(active_indices) < 10:
            print(f"  ⚠️ 有効動作フレームが少なすぎるためスキップ: {len(active_indices)} フレーム")
            continue

        print(f"  - 全フレーム数: {len(synced_filtered_joints)} -> 駆動抽出: {len(active_indices)} ({len(active_indices)/len(synced_filtered_joints)*100:.1f}%)")

        all_episode_data.append({
            'episode_name': ep_name,
            'images': synced_images,
            'raw_joints': synced_raw_joints,
            'filtered_joints': synced_filtered_joints,
            'active_indices': active_indices,
            'masked_images': synced_images[active_indices],
            'masked_joints': synced_filtered_joints[active_indices]
        })

    if len(all_episode_data) == 0:
        print("❌ 有効なエピソードデータが抽出できませんでした。")
        return

    # 全エピソード統合のデータセット保存
    all_masked_images = np.concatenate([ep['masked_images'] for ep in all_episode_data])
    all_masked_joints = np.concatenate([ep['masked_joints'] for ep in all_episode_data])
    episode_lengths = [len(ep['masked_joints']) for ep in all_episode_data]
    episode_names = [ep['episode_name'] for ep in all_episode_data]

    # 保存1: 全エピソード詳細データ（可視化用）
    np.save(os.path.join(output_dir, "episodes_metadata.npy"), all_episode_data, allow_pickle=True)

    # 保存2: Step 4学習用統合データセット
    multi_dataset_path = os.path.join(output_dir, "dataset_action_masked_multi.npz")
    np.savez_compressed(
        multi_dataset_path,
        images=all_masked_images,
        filtered_joints=all_masked_joints,
        episode_lengths=episode_lengths,
        episode_names=episode_names
    )

    # 従来の単一ファイル互換パスにも保存
    np.savez_compressed(
        os.path.join(output_dir, "dataset_action_masked.npz"),
        images=all_masked_images,
        filtered_joints=all_masked_joints
    )

    print("\n" + "="*60)
    print("🎉 全エピソード前処理＆統合完了！")
    print(f"   統合エピソード数: {len(all_episode_data)}")
    for name, length in zip(episode_names, episode_lengths):
        print(f"     * {name}: {length} フレーム (約{length*0.2:.1f}秒)")
    print(f"   統合純粋動作フレーム総数: {len(all_masked_images)} フレーム")
    print(f"   学習用データセット保存先: {multi_dataset_path}")
    print("="*60)

if __name__ == '__main__':
    process_all_episodes()
