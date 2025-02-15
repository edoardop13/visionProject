import os
import cv2
import copy
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

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
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x  

class CombinedModel(nn.Module):
    def __init__(self, 
                 cnn_model,          
                 landmark_branch,    
                 cnn_out_dim,        
                 land_out_dim,       
                 num_classes):
        super().__init__()
        self.cnn_model = cnn_model            
        self.landmark_branch = landmark_branch  

        self.final_fc = nn.Linear(cnn_out_dim + land_out_dim, num_classes)

    def forward(self, images, landmarks):

        img_feat = self.cnn_model(images)           

        land_feat = self.landmark_branch(landmarks) 

        combined = torch.cat([img_feat, land_feat], dim=1) 

        logits = self.final_fc(combined)
        return logits  

def main():
    root_dir = "hand gestures.v2-release.coco"
    train_landmark_file = "roboflow/X_train.npy"
    val_landmark_file = "roboflow/X_valid.npy"
    annotation_folder = "hand gestures.v2-release.coco"

    train_annotation_file = os.path.join(annotation_folder, "train", "_annotations.coco.json")
    val_annotation_file = os.path.join(annotation_folder, "valid", "_annotations.coco.json")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageLandmarkDataset(root_dir, "train", train_landmark_file, train_annotation_file, transform=transform)
    val_dataset   = ImageLandmarkDataset(root_dir, "valid", val_landmark_file, val_annotation_file, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    max_train_label = max(train_dataset.labels)
    max_val_label = max(val_dataset.labels)

    num_classes = max(max_train_label, max_val_label) + 1  

    cnn_model = models.googlenet(weights="IMAGENET1K_V1")  
    cnn_out_dim = cnn_model.fc.in_features
    cnn_model.fc = nn.Identity()

    land_branch = LandmarkBranch(landmark_dim=63, hidden_dim=128, out_dim=128)
    land_out_dim = 128

    combined_model = CombinedModel(cnn_model, land_branch, cnn_out_dim, land_out_dim, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    combined_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(combined_model.parameters(), lr=1e-4)

    num_epochs = 10
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)
        combined_model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0

        for batch_idx, (imgs, lands, labels) in enumerate(train_loader):

            if labels.max().item() >= num_classes:
                raise ValueError(f"[ERROR] Target {labels.max().item()} è fuori dal range! num_classes = {num_classes}")

            imgs, lands, labels = imgs.to(device), lands.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = combined_model(imgs, lands)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, dim=1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

        epoch_loss = running_loss / running_total
        epoch_acc = running_correct / running_total
        print(f"Train - Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

        combined_model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, lands, labels in val_loader:
                imgs, lands, labels = imgs.to(device), lands.to(device), labels.to(device)
                outputs = combined_model(imgs, lands)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * imgs.size(0)
                _, preds = torch.max(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss / val_total
        val_acc  = val_correct / val_total
        print(f"Val   - Loss: {val_loss:.4f}  Acc: {val_acc:.4f}\n")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(combined_model.state_dict())

    torch.save(combined_model.state_dict(), "googlenet_combined_roboflow.pth")
    print(f"Model saved with best validation accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()
