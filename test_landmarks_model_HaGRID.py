import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

label_classes = np.load("label_classes.npy", allow_pickle=True)
num_classes = len(label_classes)

# We need to define the architecture used for the training
class GestureClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(GestureClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)

        logits = self.fc3(x)  # Returns logits
        return logits

model_path = "gesture_model_pytorch.pth"
input_size = 63  # keypoints
model = GestureClassifier(input_size, num_classes)

state_dict = torch.load(model_path, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

print("[INFO] Model uploaded!")

X_test = np.load("X_hagrid_test.npy")  # shape (num_samples_test, input_size)
y_test = np.load("y_hagrid_test.npy")  # shape (num_samples_test, )

print(f"[DEBUG] X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

label_encoder = LabelEncoder()
label_encoder.classes_ = label_classes
y_test_encoded = label_encoder.transform(y_test)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

all_preds = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        # Forward pass
        logits = model(X_batch)  # shape (batch_size, num_classes)

        _, predicted = torch.max(logits, dim=1)

        # Store results
        all_preds.append(predicted.numpy())
        all_labels.append(y_batch.numpy())

all_preds  = np.concatenate(all_preds, axis=0)   # shape (n_test_samples, )
all_labels = np.concatenate(all_labels, axis=0)  # shape (n_test_samples, )


accuracy = accuracy_score(all_labels, all_preds)

# F1-score (macro, micro o weighted)
f1_macro = f1_score(all_labels, all_preds, average="macro")
f1_micro = f1_score(all_labels, all_preds, average="micro")
f1_weighted = f1_score(all_labels, all_preds, average="weighted")

print("\n===== TEST SET RESULTS =====")
print(f"Accuracy:             {accuracy:.4f}")
print(f"F1-score (macro):     {f1_macro:.4f}")
print(f"F1-score (micro):     {f1_micro:.4f}")
print(f"F1-score (weighted):  {f1_weighted:.4f}")
