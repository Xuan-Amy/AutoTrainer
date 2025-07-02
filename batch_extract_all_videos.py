import cv2
import mediapipe as mp
import csv
import os
from tqdm import tqdm

# ========= 配置部分 =========
video_dir = "videos"                 # 视频主目录（可含子目录）
csv_path = "video_labels.csv"       # CSV 输出路径
skip_rate = 1                        # 每 N 帧采一次
# ===========================

# 初始化 MediaPipe（最多检测两只手）
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)

# ==== 判断 CSV 是否首次写入 ====
is_first_write = not os.path.exists(csv_path)
write_mode = "w" if is_first_write else "a"

# 收集所有符合条件的视频文件路径
video_files = []
for root, dirs, files in os.walk(video_dir):
    for filename in files:
        if filename.endswith((".mp4", ".mov")):
            video_files.append((os.path.join(root, filename), os.path.basename(root)))

# 打开 CSV 文件
with open(csv_path, mode=write_mode, newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)

    # 写入表头
    if is_first_write:
        header = ["frame_filename"] + \
                 [f"hand{i}_{pt}_{axis}" for i in range(2) for pt in range(21) for axis in ['x', 'y', 'z']] + \
                 ["is_two_hands", "label"]
        writer.writerow(header)

    # 遍历每个视频（外层进度条）
    for video_path, label in tqdm(video_files, desc="📦 正在处理视频", unit="video"):
        filename = os.path.basename(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_index = 0

        # 每帧进度条
        with tqdm(total=total_frames, desc=f"{filename}", unit="frame", leave=False) as frame_pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_index % skip_rate != 0:
                    frame_index += 1
                    frame_pbar.update(1)
                    continue

                frame_filename = f"{label}_{os.path.splitext(filename)[0]}_frame_{frame_index:04d}.jpg"
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                # 仅保存检测到手的帧
                if results.multi_hand_landmarks:
                    hand_num = len(results.multi_hand_landmarks)
                    landmarks_all = []

                    for i in range(min(hand_num, 2)):
                        lm = results.multi_hand_landmarks[i].landmark
                        landmarks_all.extend([coord for pt in lm for coord in (pt.x, pt.y, pt.z)])

                    if hand_num == 1:
                        landmarks_all.extend([-1] * (21 * 3))  # 另一只手补齐

                    row = [frame_filename]
                    row.extend(landmarks_all)
                    row.append(1 if hand_num == 2 else 0)
                    row.append(label)
                    writer.writerow(row)

                frame_index += 1
                frame_pbar.update(1)

        cap.release()

hands.close()
print("\n✅ 所有视频处理完毕，数据已写入：", csv_path)
