import cv2
import os
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import json

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.2)

dataset_base_path = "hand gestures.v2-release.coco"
annotations_paths = {
    "train": os.path.join(dataset_base_path, "train", "_annotations.coco.json"),
    "valid": os.path.join(dataset_base_path, "valid", "_annotations.coco.json"),
    "test": os.path.join(dataset_base_path, "test", "_annotations.coco.json")
}

output_dir = "roboflow"
os.makedirs(output_dir, exist_ok=True)

def load_annotations(annotations_file):
    with open(annotations_file, "r") as f:
        data = json.load(f)

    images = {img["id"]: img["file_name"] for img in data["images"]}
    annotations = {img_id: [] for img_id in images.keys()}

    for ann in data["annotations"]:
        image_id = ann["image_id"]
        category_id = ann["category_id"]
        if image_id in annotations:
            annotations[image_id].append(category_id)

    print(f"[INFO] Loaded {len(images)} images and {len(data['annotations'])} annotations.")
    return images, annotations

def process_images(split, images, ground_truths):
    dataset_path = os.path.join(dataset_base_path, split)

    X_data, y_data, paths_data = [], [], set()
    total_images, processed_images = 0, 0

    for image_id, file_name in tqdm(images.items(), desc=f"Processing {split} images"):
        image_path = os.path.join(dataset_path, file_name)
        total_images += 1

        if image_path in paths_data:
            print(f"[WARNING] Duplicate processing detected for image {image_id}: {image_path}")
            continue

        if not os.path.exists(image_path):
            print(f"[WARNING] Image not found: {image_path}. Skipping.")
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] Could not load image: {image_path}. Skipping.")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        labels = ground_truths.get(image_id, None)
        if labels is None or len(labels) == 0:
            print(f"[WARNING] No labels found for image {image_id}. Skipping.")
            continue  

        label = labels[0]  

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                feature_vector = []
                for keypoint in mp_hands.HandLandmark:
                    landmark = hand_landmarks.landmark[keypoint]
                    feature_vector.extend([landmark.x, landmark.y, landmark.z])

                X_data.append(feature_vector)
                y_data.append(label)
                paths_data.add(image_path)
                processed_images += 1
        else:
            print(f"[WARNING] No hand found in image {file_name}. Skipping.")

    print(f"[INFO] Processed {len(paths_data)}/{total_images} unique images from {split} set.")
    return np.array(X_data, dtype=np.float32), np.array(y_data), list(paths_data)

for split in ["train", "valid", "test"]:
    images, ground_truths = load_annotations(annotations_paths[split])
    X_data, y_data, paths_data = process_images(split, images, ground_truths)

    np.save(os.path.join(output_dir, f"X_{split}.npy"), X_data)
    np.save(os.path.join(output_dir, f"y_{split}.npy"), y_data)
    np.save(os.path.join(output_dir, f"paths_{split}.npy"), paths_data)

    print(f"[INFO] Saved extracted features for {split} set in {output_dir} directory.")