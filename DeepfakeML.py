import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt


#Настройка на dataset

REAL_DIR = "D:/STM/data/real"
FAKE_DIR = "D:/STM/data/fake"
FRAMES_PER_VIDEO = 3


def load_images(folder, label, frames_per_video=3):
    images = []
    labels = []
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
        frames = [f for f in os.listdir(subfolder_path) if f.endswith(".jpg") or f.endswith(".png")]
        frames.sort()
        for frame_file in frames[:frames_per_video]:
            frame_path = os.path.join(subfolder_path, frame_file)
            img = cv2.imread(frame_path)
            if img is not None:
                img_resized = cv2.resize(img, (64, 64))
                images.append(img_resized.flatten())
                labels.append(label)
    return images, labels


real_data, real_labels = load_images(REAL_DIR, "REAL", FRAMES_PER_VIDEO)
fake_data, fake_labels = load_images(FAKE_DIR, "FAKE", FRAMES_PER_VIDEO)

X = np.array(real_data + fake_data)
y = np.array(real_labels + fake_labels)
print(f"Loaded {len(X)} images: {len(real_data)} real, {len(fake_data)} fake")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"\nBase Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


INPUT_DIRS = [REAL_DIR, FAKE_DIR]
OUTPUT_ROOT = "D:/STM/data_compressed"
QUALITIES = [90, 70, 50]

os.makedirs(OUTPUT_ROOT, exist_ok=True)

for input_dir in INPUT_DIRS:
    category = os.path.basename(input_dir)
    for quality in QUALITIES:
        output_dir = os.path.join(OUTPUT_ROOT, f"{category}_q{quality}")
        os.makedirs(output_dir, exist_ok=True)
        for subfolder in os.listdir(input_dir):
            sub_path = os.path.join(input_dir, subfolder)
            if not os.path.isdir(sub_path):
                continue
            out_subfolder = os.path.join(output_dir, subfolder)
            os.makedirs(out_subfolder, exist_ok=True)
            for frame_file in os.listdir(sub_path):
                if frame_file.endswith(".jpg") or frame_file.endswith(".png"):
                    frame_path = os.path.join(sub_path, frame_file)
                    img = cv2.imread(frame_path)
                    if img is not None:
                        out_path = os.path.join(out_subfolder, frame_file)
                        cv2.imwrite(out_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

print("Compression Done! JPEG qualities 90,70,50 created.")


results = []

for q in QUALITIES:
    real_dir_q = os.path.join(OUTPUT_ROOT, f"real_q{q}")
    fake_dir_q = os.path.join(OUTPUT_ROOT, f"fake_q{q}")

    real_data_q, real_labels_q = load_images(real_dir_q, 0, FRAMES_PER_VIDEO)
    fake_data_q, fake_labels_q = load_images(fake_dir_q, 1, FRAMES_PER_VIDEO)

    X_q = np.array(real_data_q + fake_data_q)
    y_q = np.array(real_labels_q + fake_labels_q)

    X_train_q, X_test_q, y_train_q, y_test_q = train_test_split(X_q, y_q, test_size=0.2, random_state=42)
    model_q = LogisticRegression(max_iter=1000)
    model_q.fit(X_train_q, y_train_q)
    y_pred_q = model_q.predict(X_test_q)
    acc = accuracy_score(y_test_q, y_pred_q)
    results.append((q, acc))
    print(f"JPEG quality {q} → Accuracy: {acc:.2f}")


qualities, accuracies = zip(*results)
plt.plot(qualities, accuracies, marker='o')
plt.xlabel("JPEG Quality")
plt.ylabel("Accuracy")
plt.title("Влияние на компресията върху точността")
plt.gca().invert_xaxis()  
plt.show()

import shutil


#изтриване на file 
delete = True 

if delete:
    if os.path.exists(OUTPUT_ROOT):
        shutil.rmtree(OUTPUT_ROOT)
        print(f"Deleted compressed folder: {OUTPUT_ROOT}")