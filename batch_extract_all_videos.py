import cv2
import mediapipe as mp
import csv
import os

# ========= 配置部分 =========
video_dir = "videos"                 # 所有视频所在文件夹（支持中文名）
output_dir = "video_frames"         # 所有帧图像统一保存目录
csv_path = "video_labels.csv"       # 总 CSV 输出路径
skip_rate = 1                       # 每 N 帧采一次
# ===========================

# 初始化输出文件夹
os.makedirs(output_dir, exist_ok=True)

# 初始化 MediaPipe（最多检测两只手）
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)

# ==== 自动判断 CSV 是否首次写入 ====
is_first_write = not os.path.exists(csv_path)
write_mode = "w" if is_first_write else "a"

# 打开 CSV 文件
with open(csv_path, mode=write_mode, newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)

    # 写入表头
    if is_first_write:
        header = ["frame_filename"] + \
                 [f"hand{i}_{pt}_{axis}" for i in range(2) for pt in range(21) for axis in ['x', 'y', 'z']] + \
                 ["is_two_hands", "label"]
        writer.writerow(header)

    # 遍历所有 mp4 视频
    for filename in os.listdir(video_dir):
        if not filename.endswith(".mp4"):
            continue

        video_path = os.path.join(video_dir, filename)
        action_label = os.path.splitext(filename)[0]  # 中文文件名作为标签（如 房子.mp4 → 房子）
        print(f"\n🚀 开始处理视频：{filename}，标签：{action_label}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ 无法打开视频：{video_path}")
            continue

        frame_index = 0
        valid_frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % skip_rate != 0:
                frame_index += 1
                continue

            frame_filename = f"{action_label}_frame_{frame_index:04d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            success = cv2.imwrite(frame_path, frame)
            if not success:
                print(f"⚠️ 无法保存帧 {frame_index} 到 {frame_path}")
                frame_index += 1
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            row = [frame_filename]

            if results.multi_hand_landmarks:
                hand_num = len(results.multi_hand_landmarks)
                landmarks_all = []

                for i in range(min(hand_num, 2)):
                    lm = results.multi_hand_landmarks[i].landmark
                    landmarks_all.extend([coord for pt in lm for coord in (pt.x, pt.y, pt.z)])

                if hand_num == 1:
                    landmarks_all.extend([-1] * (21 * 3))  # 补齐另一只手为 -1

                row.extend(landmarks_all)
                row.append(1 if hand_num == 2 else 0)  # 是否双手
                valid_frame_count += 1
                print(f"✅ {action_label} 第 {frame_index:04d} 帧：检测到 {hand_num} 手")
            else:
                # 没检测到手，填 -1
                row.extend([-1] * (21 * 3 * 2))
                row.append(0)
                print(f"⛔ {action_label} 第 {frame_index:04d} 帧：未检测到手")

            row.append(action_label)
            writer.writerow(row)
            frame_index += 1

        cap.release()
        print(f"📊 完成：{filename}，总帧：{frame_index}，有效帧：{valid_frame_count}")

hands.close()
print("\n✅ 所有视频处理完毕，数据写入：", csv_path)
