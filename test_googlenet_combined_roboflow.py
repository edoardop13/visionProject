import os
import cv2
import json
import numpy as np
import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def _load_annotations_from_coco(annotation_file):
    with open(annotation_file, "r") as f:
        data = json.load(f)

    annotations = {}
    for ann in data["annotations"]:
        image_id = ann["image_id"]
        category_id = ann["category_id"]
        if image_id not in annotations:
            annotations[image_id] = category_id

    images = {img["id"]: img["file_name"] for img in data["images"]}

    print(f"[INFO] Caricate {len(images)} immagini e {len(annotations)} annotazioni.")
    print(f"[DEBUG] Esempio immagini: {list(images.items())[:5]}")
    print(f"[DEBUG] Esempio annotazioni: {list(annotations.items())[:5]}")

    return images, annotations

class ImageLandmarkDataset(Dataset):
    def __init__(self, root_dir, split, landmark_file, annotation_file, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.landmarks = np.load(landmark_file)  
        self.images, self.annotations = _load_annotations_from_coco(annotation_file)

        self.image_paths = []
        self.labels = []

        split_dir = os.path.join(self.root_dir, self.split)
        for image_id, file_name in self.images.items():
            image_path = os.path.join(split_dir, file_name)
            if not os.path.exists(image_path):
                print(f"[WARNING] Immagine non trovata: {image_path}")
            if image_id in self.annotations:
                self.image_paths.append(image_path)
                self.labels.append(self.annotations[image_id])

        print(f"[DEBUG] {split.upper()} - Campioni caricati: {len(self.image_paths)}")
        print(f"[DEBUG] {split.upper()} - Esempio di immagini nel dataset: {self.image_paths[:5]}")
        print(f"[DEBUG] {split.upper()} - Esempio di etichette nel dataset: {self.labels[:5]}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        landmark = self.landmarks[idx]  
        label = self.labels[idx]        

        try:
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                raise FileNotFoundError(f"Image not found: {image_path}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            if self.transform:
                from PIL import Image
                img_rgb = Image.fromarray(img_rgb)
                img_tensor = self.transform(img_rgb)  
            else:
                img_tensor = torch.tensor(img_rgb).permute(2, 0, 1).float() / 255.0

            landmark_tensor = torch.tensor(landmark, dtype=torch.float32)
            label_tensor = torch.tensor(label, dtype=torch.long)

            return img_tensor, landmark_tensor, label_tensor

        except Exception as e:
            print(f"[ERROR] Errore caricando immagine {image_path}: {e}")
            raise

class LandmarkBranch(nn.Module):
    def __init__(self, landmark_dim=63, hidden_dim=128, out_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(landmark_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class CombinedModel(nn.Module):
    def __init__(self, cnn_model, landmark_branch, cnn_out_dim, land_out_dim, num_classes):
        super().__init__()
        self.cnn_model = cnn_model
        self.landmark_branch = landmark_branch
        self.final_fc = nn.Linear(cnn_out_dim + land_out_dim, num_classes)

    def forward(self, images, landmarks):
        img_feat = self.cnn_model(images)
        land_feat = self.landmark_branch(landmarks)
        return self.final_fc(torch.cat([img_feat, land_feat], dim=1))

def main():
    root_dir = "hand gestures.v2-release.coco"
    test_split = "test"
    test_landmark_file = "roboflow/X_test.npy"
    annotation_folder = "hand gestures.v2-release.coco"
    model_path = "googlenet_combined_roboflow.pth"
    output_file = "googlenet_combined_roboflow_test.txt"

    test_annotation_file = os.path.join(annotation_folder, "test", "_annotations.coco.json")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset   = ImageLandmarkDataset(root_dir, "test", test_landmark_file, test_annotation_file, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    max_test_label = max(test_dataset.labels)
    print(f"[DEBUG] Valore massimo nelle etichette di training: {max_test_label}")

    num_classes = max_test_label + 1  
    print(f"[DEBUG] Numero totale di classi impostato: {num_classes}")

    cnn_model = models.googlenet(pretrained=False, aux_logits=False)
    cnn_out_dim = cnn_model.fc.in_features
    cnn_model.fc = nn.Identity()

    landmark_branch = LandmarkBranch(landmark_dim=63, hidden_dim=128, out_dim=128)
    combined_model = CombinedModel(cnn_model, landmark_branch, cnn_out_dim, land_out_dim=128, num_classes=num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    combined_model.to(device)
    combined_model.load_state_dict(torch.load(model_path, map_location=device))
    combined_model.eval()

    print(f"Model loaded from: {model_path}\nStarting inference on test set...")

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, lands, labels in tqdm.tqdm(test_loader, desc="Inferencing"):
            imgs, lands, labels = imgs.to(device), lands.to(device), labels.to(device)
            outputs = combined_model(imgs, lands)
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1-score: {f1:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")

    with open(output_file, "w") as f:
        f.write(f"Test Accuracy: {accuracy:.4f}\n")
        f.write(f"Test F1-score: {f1:.4f}\n")
        f.write(f"Test Precision: {precision:.4f}\n")
        f.write(f"Test Recall: {recall:.4f}\n")

    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
