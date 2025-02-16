import os
import torch
import json
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from efficientnet_pytorch import EfficientNet

class CocoDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        self.image_info = {img["id"]: img for img in data["images"]}
        self.annotations = {img_id: [] for img_id in self.image_info.keys()}
        for ann in data["annotations"]:
            self.annotations[ann["image_id"]].append(ann["category_id"])
        
        self.classes = {}
        for cat in data["categories"]:
            if cat["id"] > 0:  # skip the supercategory with id=0
                self.classes[cat["id"]] = cat["name"]


    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):
        img_id = list(self.image_info.keys())[idx]
        img_data = self.image_info[img_id]
        img_path = os.path.join(self.image_dir, img_data["file_name"])
        image = Image.open(img_path).convert("RGB")
        label = self.annotations[img_id][0] if self.annotations[img_id] else 0
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Paths
path = os.path.realpath(os.path.dirname(__file__)) # Path to the current directory
test_dir = path + "/handgestures.v2-release.coco/test"
test_ann = path + "/handgestures.v2-release.coco/test/_annotations.coco.json"

# -----------------------------------------------------------------------------
# 1. Create Data Loaders for Test
# -----------------------------------------------------------------------------
batch_size = 32

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Create dataset
test_dataset = CocoDataset(test_dir, test_ann, transform=transform)
 
# Create data loader
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

print(f"1 - Loaded {len(test_dataset)} test images")

# -----------------------------------------------------------------------------
# 2. Load Model
# -----------------------------------------------------------------------------
# Load a pre-trained EfficientNet-B0 model
model = EfficientNet.from_pretrained('efficientnet-b0')

# Replace the final fully connected layer to match the number of gesture classes
num_features = model._fc.in_features
num_gesture_classes = len(test_dataset.classes)
model._fc = nn.Linear(num_features, num_gesture_classes)

# Move the model to the GPU (if available)
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print("2 - Model loaded.")

# -----------------------------------------------------------------------------
# 3. Load Weights
# -----------------------------------------------------------------------------
# Load previous trained weights
checkpoint_path = "EfficientNet_roboflow_gestures_best.pt"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

print("3 - Previous trained weights loaded.")

# -----------------------------------------------------------------------------
# 4. Inference Loop on Test Data
# -----------------------------------------------------------------------------
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

# -----------------------------------------------------------------------------
# 5. Metrics (Accuracy, Classification Report, etc.)
# -----------------------------------------------------------------------------
# 5a) Simple accuracy
correct_count = sum([1 for p, t in zip(all_preds, all_labels) if p == t])
accuracy = correct_count / len(all_labels)
print(f"\n6 - Test Accuracy: {accuracy:.4f}")

# 5b) Classification report (requires scikit-learn)
print("\n6 - Classification Report:")
unique_labels = sorted(set(all_labels) | set(all_preds))  # or known 0..14
target_names = [v for _, v in sorted(test_dataset.classes.items(), key=lambda x: x[0])]
print(classification_report(all_labels, all_preds, labels=unique_labels, target_names=target_names))

# 5c) Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
print("6 - Confusion Matrix:")
print(cm)