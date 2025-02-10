import os
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from efficientnet_pytorch import EfficientNet

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
    
# A helper function to extract the gesture label (removing the "test_val_" prefix)
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
            print(f"[Warning] Directory {class_dir} not found.")
            continue
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(class_dir, img_name)
                image_paths.append(img_path)
                labels.append(idx)
    return image_paths, labels

# Paths
path = os.path.realpath(os.path.dirname(__file__)) # Path to the current directory
test_images_dir = path + "/hagrid-sample/hagrid-sample-500k-384p/split/test"     # Path to test images (with subfolders)

# -----------------------------------------------------------------------------
# 1. Extract Gesture Labels
# -----------------------------------------------------------------------------
classes = extract_label(test_images_dir)
print(f"1 - Detected classes: {classes}")

test_image_paths, test_labels = get_image_paths_and_labels(test_images_dir, classes)
print("1 - Number of test images:", len(test_image_paths), "and labels:", len(test_labels))

# -----------------------------------------------------------------------------
# 2. Create Data Loaders for Test
# -----------------------------------------------------------------------------
batch_size = 32

# Define the image transformations
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_dataset = HaGRID_Dataset(test_image_paths, test_labels, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("2 - Loaded", len(test_dataset), "test images.")

# -----------------------------------------------------------------------------
# 4. Load Model
# -----------------------------------------------------------------------------
# Load a pre-trained EfficientNet-B0 model
model = EfficientNet.from_pretrained('efficientnet-b0')

# Replace the final fully connected layer to match the number of gesture classes
num_features = model._fc.in_features
num_gesture_classes = len(classes)
model._fc = nn.Linear(num_features, num_gesture_classes)

# Move the model to the GPU (if available)
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print("3 - Model loaded.")

# -----------------------------------------------------------------------------
# 4. Load Weights
# -----------------------------------------------------------------------------
# Load previous trained weights
checkpoint_path = "EfficientNet_hagrid_gestures.pt"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

print("4 - Previous trained weights loaded.")

# -----------------------------------------------------------------------------
# 5. Inference Loop on Test Data
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
# 6. Metrics (Accuracy, Classification Report, etc.)
# -----------------------------------------------------------------------------
# 6a) Simple accuracy
correct_count = sum([1 for p, t in zip(all_preds, all_labels) if p == t])
accuracy = correct_count / len(all_labels)
print(f"\n6 - Test Accuracy: {accuracy:.4f}")

# 6b) Classification report (requires scikit-learn)
print("\n6 - Classification Report:")
print(classification_report(all_labels, all_preds, target_names=classes))

# 6c) Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
print("6 - Confusion Matrix:")
print(cm)
