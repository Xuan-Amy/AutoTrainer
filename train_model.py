import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
from tqdm import tqdm
import time

# === 配置 ===
csv_path = "video_labels.csv"                  # 数据路径（含双手数据和 is_two_hands）
model_dir = "models"                           # 模型保存目录
model_output_path = os.path.join(model_dir, "svm_model.joblib")

# 初始化 tqdm 阶段进度条
steps = [
    "加载数据", "提取特征与标签", "过滤无手帧",
    "标签编码", "特征标准化", "拆分数据集",
    "训练模型", "评估模型", "保存模型组件"
]
pbar = tqdm(steps, desc="🚀 正在训练 SVM 模型", ncols=100)

# === 1. 加载数据 ===
try:
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
except FileNotFoundError:
    print(f"❌ 找不到 CSV 文件：{csv_path}")
    exit(1)
pbar.update()

# === 2. 提取特征与标签（保留 is_two_hands）===
try:
    X = df.drop(columns=["frame_filename", "label"]).values
    y = df["label"].values
except KeyError:
    print("❌ CSV 中缺少必要列名：frame_filename 或 label")
    exit(1)
pbar.update()

# === 3. 过滤掉无手帧 ===
mask = ~np.all(X[:, :-1] == -1, axis=1)
X = X[mask]
y = y[mask]
if len(X) == 0:
    print("❌ 所有帧均无手势关键点，无法训练")
    exit(1)
pbar.update()

# === 4. 标签编码 ===
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

os.makedirs(model_dir, exist_ok=True)
label_map_path = os.path.join(model_dir, "label_map.txt")
with open(label_map_path, "w", encoding="utf-8") as f:
    for label, idx in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)):
        f.write(f"{idx},{label}\n")
pbar.update()

# === 5. 特征标准化 ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pbar.update()

# === 6. 拆分数据集 ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
pbar.update()

# === 7. 训练模型 ===
clf = SVC(kernel='rbf', probability=True)
clf.fit(X_train, y_train)
pbar.update()

# === 8. 模型评估 ===
y_pred = clf.predict(X_test)
print("\n📊 分类报告:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print("🧩 混淆矩阵:")
print(confusion_matrix(y_test, y_pred))
pbar.update()

# === 9. 保存模型组件 ===
joblib.dump(clf, os.path.join(model_dir, "svm_model.joblib"))
joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))
joblib.dump(label_encoder, os.path.join(model_dir, "label_encoder.joblib"))

pbar.update()
pbar.close()

print(f"\n✅ 模型训练完成，保存至：{model_output_path}")
print(f"📄 标签映射表已写入：{label_map_path}")
