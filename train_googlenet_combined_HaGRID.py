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

def _load_annotations_from_folder(annotation_folder):
    
    annotation_dict = {}
    
    ann_files = [f for f in os.listdir(annotation_folder) if f.endswith(".json")]
    
    for ann_file in ann_files:
        path_file = os.path.join(annotation_folder, ann_file)
        with open(path_file, 'r') as f:
            data = json.load(f)

        
        for img_id, content in data.items():
            if img_id not in annotation_dict:
                annotation_dict[img_id] = {
                    "labels": [],
                    "bboxes": []
                }
            annotation_dict[img_id]["labels"].extend(content["labels"])
            annotation_dict[img_id]["bboxes"].extend(content["bboxes"])
    
    return annotation_dict

class ImageLandmarkDataset(Dataset):
    def __init__(self, 
                 root_dir, 
                 split, 
                 landmark_file, 
                 annotation_folder,  # <-- folder with annotations JSON files
                 transform=None):
        
        self.root_dir = root_dir
        self.split = split  # "train", "val", "test"
        self.transform = transform
        
        self.landmarks = np.load(landmark_file)  # shape [N, 63]
        
        # Upload and merge all JSON files found in the folder
        self.annotations = _load_annotations_from_folder(annotation_folder)

        # We create a mapping "img_id -> label_string"
        # If there is one label != no_gesture, keep the first one found (between 2 hands for example)
        self.img_to_label = self._build_img_to_label_dict()

        # List of samples and the mapping label->indice
        self.df_data, self.label_to_idx = self._load_data()


    def _build_img_to_label_dict(self):

        img_to_label = {}
        for img_id, ann in self.annotations.items():
            labels = ann["labels"]
            
            # If there are multiple hands and they are no_gesture, keep one
            chosen_label = None
            for lab in labels:
                if lab != "no_gesture":
                    chosen_label = lab
                    break
            
            if chosen_label is None:
                chosen_label = "no_gesture"
            
            img_to_label[img_id] = chosen_label
        return img_to_label


    def _load_data(self):
        """
        Create sample image list (image_path, landmark, label_idx).
        """
        # To build the mapping label_str -> indice
        all_labels = set(self.img_to_label.values())
        label_to_idx = {lab: i for i, lab in enumerate(sorted(all_labels))}
        
        split_dir = os.path.join(self.root_dir, self.split)
        image_files = []

        for root, _, files in os.walk(split_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    image_files.append(os.path.join(root, f))
        
        data = []
        for idx, img_file in enumerate(image_files):
            # Take name without extension
            img_id = os.path.splitext(os.path.basename(img_file))[0]
            
            if img_id in self.img_to_label:
                label_str = self.img_to_label[img_id]
                label_idx = label_to_idx[label_str]

                if idx < len(self.landmarks):
                    data.append({
                        "image_path": img_file,
                        "landmark": self.landmarks[idx],
                        "label": label_idx
                    })
        
        print(f"[DEBUG] _load_data generated {len(data)} samples for the split: {self.split}")
        return data, label_to_idx


    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, idx):
        sample = self.df_data[idx]
        image_path = sample["image_path"]
        landmark = sample["landmark"]  # shape (63,)
        label = sample["label"]        # int

        try:
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                raise FileNotFoundError(f"Image not found: {image_path}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            if self.transform:
                from PIL import Image
                img_rgb = Image.fromarray(img_rgb)
                img_tensor = self.transform(img_rgb)  # shape [3,H,W]
            else:
                img_tensor = torch.tensor(img_rgb).permute(2, 0, 1).float() / 255.0

            landmark_tensor = torch.tensor(landmark, dtype=torch.float32)
            label_tensor = torch.tensor(label, dtype=torch.long)

            return img_tensor, landmark_tensor, label_tensor

        except Exception as e:
            print(f"[ERROR] Errore loading image {image_path}: {e}")
            raise

# Simple MLP to map a vector of dimention 'landmark_dim' to a vector of dimention 'out_dim'
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
        return x  # shape (batch, out_dim)

class CombinedModel(nn.Module):
    def __init__(self, 
                 cnn_model,          # pretrained model (es. ResNet con fc = Identity)
                 landmark_branch,    # MLP for landmark
                 cnn_out_dim,        # feature dimention for CNN
                 land_out_dim,       # feature dimention for landmarks' branch
                 num_classes):
        super().__init__()
        self.cnn_model = cnn_model            # images feature extractor
        self.landmark_branch = landmark_branch  # landmarks feature extractor
        # Final fully connected
        self.final_fc = nn.Linear(cnn_out_dim + land_out_dim, num_classes)

    def forward(self, images, landmarks):
        # Get features from image
        img_feat = self.cnn_model(images)           # shape (batch, cnn_out_dim)
        # Get features from landmarks
        land_feat = self.landmark_branch(landmarks) # shape (batch, land_out_dim)
        # Concatenate
        combined = torch.cat([img_feat, land_feat], dim=1) 
        # Classification
        logits = self.final_fc(combined)
        return logits  # shape (batch, num_classes)

# TRAINING LOOP
def main():
    root_dir = "hagrid-sample/hagrid-sample-500k-384p/split/"
    train_landmark_file = "X_hagrid_train.npy"
    val_landmark_file = "X_hagrid_val.npy"
    annotation_folder = "hagrid-sample/hagrid-sample-500k-384p/ann_train_val/"
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = ImageLandmarkDataset(root_dir,
                                         "train",
                                         train_landmark_file,
                                         annotation_folder,
                                         transform=transform)
    
    val_dataset   = ImageLandmarkDataset(root_dir,
                                         "val",
                                         val_landmark_file,
                                         annotation_folder,
                                         transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    num_classes = len(set([sample["label"] for sample in train_dataset.df_data]))
    cnn_model = models.googlenet(pretrained=True)
    cnn_out_dim = cnn_model.fc.in_features
    cnn_model.fc = nn.Identity()
    
    land_branch = LandmarkBranch(landmark_dim=63, hidden_dim=128, out_dim=128)
    land_out_dim = 128
    
    combined_model = CombinedModel(cnn_model, land_branch, cnn_out_dim, land_out_dim, num_classes)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    combined_model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(combined_model.parameters(), lr=1e-4)
    
    num_epochs = 5
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-"*10)
        combined_model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0
        
        for batch_idx, (imgs, lands, labels) in enumerate(train_loader):
            print(f"[DEBUG] Batch {batch_idx + 1}/{len(train_loader)}")
            print(f"[DEBUG] Images shape: {imgs.shape}, Landmark shape: {lands.shape}, Label shape: {labels.shape}")
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

            if batch_idx == 0:
                print(f"[DEBUG] Output example for the first batch (epoch {epoch+1}):")
                print(f" - outputs.shape: {outputs.shape}") 
                print(f" - outputs[0]: {outputs[0].detach().cpu().numpy()}")
                print(f" - label attesa[0]: {labels[0].item()}")
        
        epoch_loss = running_loss / running_total
        epoch_acc = running_correct / running_total
        print(f"Train - Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

        ###VALIDATION PHASE
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
        
        # Save best weights based on accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(combined_model.state_dict())
        
    print("Training completed.")
    # Search for best weights and save
    combined_model.load_state_dict(best_weights)
    torch.save(combined_model.state_dict(), "googlenet_combined.pth")
    print(f"Model saved with best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()