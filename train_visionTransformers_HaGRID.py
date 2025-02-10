import torch
import os
import json
from tqdm import tqdm
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.models import vit_b_16, ViT_B_16_Weights

class HaGRID_Dataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# A helper function to extract the gesture label (removing the "train_val_" prefix)
def extract_label(directory_path):
    directory_path = Path(directory_path)
    directory_names = [dir.name for dir in directory_path.iterdir() if dir.is_dir()]
    return [name.replace("train_val_", "") for name in directory_names]

# A helper function to get image paths and labels
def get_image_paths_and_labels(root_dir, classes):
    image_paths = []
    labels = []

    for idx, class_name in enumerate(classes):
        subfolder = f"train_val_{class_name}"
        class_dir = os.path.join(root_dir, subfolder)

        if not os.path.isdir(class_dir):
            print(f"Warning: directory {class_dir} does not exist.")
            continue

        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(class_dir, img_name)
                image_paths.append(img_path)
                labels.append(idx)

    return image_paths, labels


# Paths
path = os.path.realpath(os.path.dirname(__file__))  # Path to the current directory
train_images_dir = path + "/hagrid-sample/hagrid-sample-500k-384p/split/train"  # Path to train images (with subfolders)
val_images_dir   = path + "/hagrid-sample/hagrid-sample-500k-384p/split/val"       # Path to validation images (with subfolders)
annotations_dir  = path + "/hagrid-sample/hagrid-sample-500k-384p/ann_train_val"  # Path to JSON annotations

# -----------------------------------------------------------------------------
# 1. Extract Gesture Labels
# -----------------------------------------------------------------------------
classes = extract_label(train_images_dir)
print(f"1 - Detected classes: {classes}")

train_image_paths, train_labels = get_image_paths_and_labels(train_images_dir, classes)
val_image_paths, val_labels     = get_image_paths_and_labels(val_images_dir, classes)
print("1 - Number of train images:", len(train_image_paths), "and labels:", len(train_labels))
print("1 - Number of val images:", len(val_image_paths), "and labels:", len(val_labels))

# -----------------------------------------------------------------------------
# 2. Load Annotations
# -----------------------------------------------------------------------------
# Load all annotations from the JSON files
def load_annotations(annotations_dir):
    ground_truths = {}
    for json_file in os.listdir(annotations_dir):
        if json_file.endswith(".json"):
            with open(os.path.join(annotations_dir, json_file), "r") as f:
                annotations = json.load(f)
                for image_id, annotation in annotations.items():
                    if image_id not in ground_truths:
                        ground_truths[image_id] = {"bboxes": [], "labels": []}
                    # Convert normalized bounding boxes to absolute format
                    bboxes = []
                    labels = []
                    for bbox, label in zip(annotation["bboxes"], annotation["labels"]):
                        if len(bbox) == 4:
                            x_min = bbox[0] * 384
                            y_min = bbox[1] * 512
                            width = bbox[2] * 384
                            height = bbox[3] * 512
                            x_max = x_min + width
                            y_max = y_min + height
                            bboxes.append([x_min, y_min, x_max, y_max])
                            labels.append(label)

                    # Only add annotations with matching lengths
                    if len(bboxes) == len(labels):
                        ground_truths[image_id]["bboxes"].extend(bboxes)
                        ground_truths[image_id]["labels"].extend(labels)
                    else:
                        print(f"Warning: Inconsistent annotation for {image_id} in file {json_file}")
    return ground_truths

# Load ground truth annotations
ground_truth_annotations = load_annotations(annotations_dir)
print("2 - Loaded annotations for", len(ground_truth_annotations), "images.")

# -----------------------------------------------------------------------------
# 3. Create Data Loaders for Training and Validation
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

# Create the datasets
train_dataset = HaGRID_Dataset(train_image_paths, train_labels, transform=transform)
val_dataset   = HaGRID_Dataset(val_image_paths,   val_labels,   transform=transform)

# Create the data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

print("3 - Loaded", len(train_dataset), "training images and", len(val_dataset), "validation images.")

# -----------------------------------------------------------------------------
# 4. Load and Modify the Vision Transformer Model for Gesture Classification
# -----------------------------------------------------------------------------
# Load vit_b_16 from torchvision, with pretrained weights
weights = ViT_B_16_Weights.DEFAULT
model = vit_b_16(weights=weights)

# Replace the final classification head to match the number of gesture classes
num_features = model.heads.head.in_features
num_gesture_classes = len(classes)
model.heads.head = nn.Linear(num_features, num_gesture_classes)

# Move the model to the GPU (if available)
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("4 - Vision Transformer (ViT) model loaded and ready for training.")

# -----------------------------------------------------------------------------
# 5. Fine-tune the Model on the HaGRID Dataset
# -----------------------------------------------------------------------------
print("5 - Training the model...")
num_epochs = 5
for epoch in range(num_epochs):
    # ------------------------
    # TRAINING PHASE
    # ------------------------
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] [TRAIN]")
    for images, labels in train_pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Stats
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        train_pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{(correct/total):.4f}"
        })
    
    train_loss = running_loss / len(train_loader.dataset)
    train_acc = correct / total
    print(f"\nEpoch {epoch+1}/{num_epochs} Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}\n")

    # ------------------------
    # VALIDATION PHASE
    # ------------------------
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    val_pbar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] [VAL]")
    with torch.no_grad():
        for images, labels in val_pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            val_pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{(correct/total):.4f}"
            })
    
    val_loss /= len(val_loader.dataset)
    val_acc = correct / total

    print(f"\nEpoch {epoch+1}/{num_epochs} Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}\n")

# -----------------------------------------------------------------------------
# 6. Save the Trained Model Checkpoint
# -----------------------------------------------------------------------------
torch.save(model.state_dict(), "ViT_hagrid_gestures.pt")
print("6 - Model checkpoint saved as 'ViT_hagrid_gestures.pt'")
