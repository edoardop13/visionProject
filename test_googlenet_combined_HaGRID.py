import os
import cv2
import json
import numpy as np
import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import accuracy_score, f1_score

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
                 annotation_folder,  
                 transform=None):
        
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        self.landmarks = np.load(landmark_file)
        
        self.annotations = _load_annotations_from_folder(annotation_folder)

        self.img_to_label = self._build_img_to_label_dict()

        self.df_data, self.label_to_idx = self._load_data()


    def _build_img_to_label_dict(self):
        img_to_label = {}
        for img_id, ann in self.annotations.items():
            labels = ann["labels"]
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
        
        print(f"[DEBUG] _load_data generated {len(data)} samples fot the split: {self.split}")
        return data, label_to_idx


    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, idx):
        sample = self.df_data[idx]
        image_path = sample["image_path"]
        landmark = sample["landmark"]
        label = sample["label"]

        try:
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                raise FileNotFoundError(f"Immagine not found: {image_path}")
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
            print(f"[ERROR] Error during image upload {image_path}: {e}")
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
    root_dir          = "hagrid-sample/hagrid-sample-500k-384p/split/"
    test_split        = "test"
    test_landmark_file= "X_hagrid_test.npy"
    annotation_folder = "hagrid-sample/hagrid-sample-500k-384p/ann_train_val/"
    

    model_path = "googlenet_combined.pth"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    test_dataset = ImageLandmarkDataset(
        root_dir,
        test_split,
        test_landmark_file,
        annotation_folder,
        transform=transform
    )
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    num_classes = len(set([sample["label"] for sample in test_dataset.df_data]))

    print(f"==> Number of classes found: {num_classes}")

    cnn_model = models.googlenet(pretrained=False, aux_logits=False)
    cnn_out_dim = cnn_model.fc.in_features
    cnn_model.fc = nn.Identity()

    landmark_branch = LandmarkBranch(
        landmark_dim=63, 
        hidden_dim=128, 
        out_dim=128
    )
    
    combined_model = CombinedModel(
        cnn_model, 
        landmark_branch, 
        cnn_out_dim, 
        land_out_dim=128, 
        num_classes=num_classes
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    combined_model.to(device)
    combined_model.load_state_dict(torch.load(model_path, map_location=device))
    combined_model.eval()
    
    print(f"Model uploaded from: {model_path}")
    print("\n==> Starting inference on test set...")

    all_preds = []
    all_labels = []

    with torch.no_grad():
        batch_count = 0
        total_batches = len(test_loader)

        progress_bar = tqdm.tqdm(enumerate(test_loader), total=total_batches, desc="Inferenza in corso")

        for batch_idx, (imgs, lands, labels) in progress_bar:
            imgs, lands, labels = imgs.to(device), lands.to(device), labels.to(device)

            outputs = combined_model(imgs, lands)
            _, preds = torch.max(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Debug: Show first 5 predictions of each batch
            if batch_count % 10 == 0:  # Ogni 10 batch
                print(f"[DEBUG] Batch {batch_count}: first 5 predictions -> {preds[:5].cpu().numpy()}")


    from sklearn.metrics import accuracy_score, f1_score
    test_acc = accuracy_score(all_labels, all_preds)
    test_f1  = f1_score(all_labels, all_preds, average='macro')

    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1-score (macro): {test_f1:.4f}")

if __name__ == "__main__":
    main()
