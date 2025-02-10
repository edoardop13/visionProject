# EXTRACT HAND KEYPOINTS FROM HaGRID DATASET USING MEDIAPIPE

import cv2
import os
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import json

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.2)

dataset_path = "hagrid-sample/hagrid-sample-500k-384p/split/test"
annotations_path = "hagrid-sample/hagrid-sample-500k-384p/ann_train_val"

X_data = []
y_data = []
paths_data = []

def load_annotations(annotations_path):
    annotations = {}
    for json_file in os.listdir(annotations_path):
        if json_file.endswith(".json"):
            with open(os.path.join(annotations_path, json_file), "r") as f:
                ann_data = json.load(f)
                annotations.update(ann_data)
    print(f"[INFO] Annotations loaded: {len(annotations)}")
    return annotations

def get_class_from_path(root):
    """Get class by folder name, considering 'no_gesture'."""
    class_name = os.path.basename(root)
    if class_name.startswith("train_val_"):
        class_name = class_name.replace("train_val_", "")
    return class_name

def process_images(dataset_path, ground_truths):
    total_images = 0
    images_with_annotations = 0
    skipped_no_annotations = 0
    skipped_mismatched_class = 0
    processed_images = 0

    for root, dirs, files in os.walk(dataset_path):
        class_name = get_class_from_path(root)
        for file in tqdm(files, desc="Processing Images"):
            if file.endswith(".jpg"):
                total_images += 1
                image_path = os.path.join(root, file)
                image_id = os.path.splitext(file)[0]

                if image_id not in ground_truths:
                    skipped_no_annotations += 1
                    print(f"[WARNING] No annotation for image with ID {image_id}. Skipping.")
                    continue

                images_with_annotations += 1

                annotation_label = ground_truths[image_id]["labels"][0] if ground_truths[image_id]["labels"] else "no_gesture"
                if annotation_label != class_name and annotation_label != "no_gesture":
                    print(f"[WARNING] Class {class_name} does not match the annotation for {image_id}. Annotation: {annotation_label}. Skipping.")
                    continue

                image = cv2.imread(image_path)
                if image is None:
                    print(f"[ERROR] Image not uploaded {image_path}. Skipping.")
                    continue

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Landmarks extraction
                results = hands.process(image_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        feature_vector = []
                        for keypoint in mp_hands.HandLandmark:
                            landmark = hand_landmarks.landmark[keypoint]
                            feature_vector.extend([landmark.x, landmark.y, landmark.z])

                        label = ground_truths[image_id]["labels"][0]

                        X_data.append(feature_vector)
                        y_data.append(label)
                        paths_data.append(image_path)
                        processed_images += 1
                else:
                    print(f"[WARNING] No hand found in the image {image_id}. Skipping.")

    print("\n[INFO] Process completed:")
    print(f"- Total images in dataset: {total_images}")
    print(f"- Images with annotations: {images_with_annotations}")
    print(f"- Skipped images (no annotation): {skipped_no_annotations}")
    print(f"- Skipped images (class mismatch): {skipped_mismatched_class}")
    print(f"- Successfully processed images: {processed_images}")

ground_truths = load_annotations(annotations_path)

process_images(dataset_path, ground_truths)


X_data = np.array(X_data, dtype=np.float32)
y_data = np.array(y_data)

np.save("X_hagrid_test.npy", X_data)
np.save("y_hagrid_test.npy", y_data)
np.save("paths_hagrid_test.npy", paths_data)

print("Feature extraction completed. Saved.")