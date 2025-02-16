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
train_dir = path + "/handgestures.v2-release.coco/train"
val_dir = path + "/handgestures.v2-release.coco/valid"
train_ann = path + "/handgestures.v2-release.coco/train/_annotations.coco.json"
val_ann = path + "/handgestures.v2-release.coco/valid/_annotations.coco.json"

# -----------------------------------------------------------------------------
# 1. Create Data Loaders for Training and Validation
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

# Create datasets
train_dataset = CocoDataset(train_dir, train_ann, transform=transform)
val_dataset = CocoDataset(val_dir, val_ann, transform=transform)
 
# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"1 - Loaded {len(train_dataset)} training images and {len(val_dataset)} validation images")

# -----------------------------------------------------------------------------
# 2. Load and Modify the EfficientNet Model for Gesture Classification
# -----------------------------------------------------------------------------
# Load a pre-trained EfficientNet-B0 model
model = EfficientNet.from_pretrained('efficientnet-b0')

# Replace the final fully connected layer to match the number of gesture classes
num_features = model._fc.in_features
num_classes = len(train_dataset.classes)
model._fc = nn.Linear(num_features, num_classes)

# Move the model to the GPU (if available)
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("2 - Model loaded and ready for training.")

# -----------------------------------------------------------------------------
# 3. Fine-tune the Model on the Roboflow Dataset
# -----------------------------------------------------------------------------
print("3 - Training the model...")
num_epochs = 15
# Early Stopping Parameters
patience = 2  # how many epochs to wait before stopping
best_val_loss = float('inf')
epochs_no_improve = 0

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
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update tqdm postfix (shown to the right of the progress bar)
        train_pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{(correct/total):.4f}"
        })
    
    train_loss = running_loss / len(train_loader.dataset)
    train_acc = correct / total
    print(f"\nEpoch {epoch+1}/{num_epochs} Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f})\n")

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

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            val_pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{(correct/total):.4f}"
            })
    
    val_loss /= len(val_loader.dataset)
    val_acc = correct / total
    print(f"\nEpoch {epoch+1}/{num_epochs} Val Loss: {val_loss:.4f},   Val Acc: {val_acc:.4f}\n")

    # Check for Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        ## Save the best model
        torch.save(model.state_dict(), "EfficientNet_roboflow_gestures_best.pt")
        print(f"Validation loss improved. Saving model to EfficientNet_hagrid_gestures_best.pt")
    else:
        epochs_no_improve += 1
        print(f"Validation loss did not improve. (Epochs no improve: {epochs_no_improve})")

        if epochs_no_improve >= patience:
            print("Early stopping triggered!")
            break

# -----------------------------------------------------------------------------
# 4. Save the Trained Model Checkpoint
# -----------------------------------------------------------------------------
torch.save(model.state_dict(), "EfficientNet_roboflow_gestures.pt")
print("4 - Model checkpoint saved as 'EfficientNet_roboflow_gestures.pt'")
