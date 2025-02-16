import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

label_classes = np.load("roboflow_label_classes.npy", allow_pickle=True)
num_classes = len(label_classes)

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

        logits = self.fc3(x)  
        return logits

model_path = "ROBOFLOWgesture_model_pytorch.pth"
input_size = 63  
model = GestureClassifier(input_size, num_classes)

state_dict = torch.load(model_path, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

print("[INFO] Model uploaded!")

X_test = np.load("roboflow/X_test.npy")  
y_test = np.load("roboflow/y_test.npy")  

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
        logits = model(X_batch)  
        _, predicted = torch.max(logits, dim=1)

        all_preds.append(predicted.numpy())
        all_labels.append(y_batch.numpy())

all_preds  = np.concatenate(all_preds, axis=0)   
all_labels = np.concatenate(all_labels, axis=0)  

accuracy = accuracy_score(all_labels, all_preds)
precision_macro = precision_score(all_labels, all_preds, average="macro", zero_division=0)
recall_macro = recall_score(all_labels, all_preds, average="macro", zero_division=0)
f1_macro = f1_score(all_labels, all_preds, average="macro")

print("\n===== TEST SET RESULTS =====")
print(f"Accuracy:         {accuracy:.4f}")
print(f"Precision (macro): {precision_macro:.4f}")
print(f"Recall (macro):    {recall_macro:.4f}")
print(f"F1-score (macro):  {f1_macro:.4f}")

output_file = "landmarks_model_HaGRID_test.txt"
with open(output_file, "w") as f:
    f.write("===== TEST SET RESULTS =====\n")
    f.write(f"Accuracy:         {accuracy:.4f}\n")
    f.write(f"Precision (macro): {precision_macro:.4f}\n")
    f.write(f"Recall (macro):    {recall_macro:.4f}\n")
    f.write(f"F1-score (macro):  {f1_macro:.4f}\n")

print(f"\n[INFO] Results saved to '{output_file}'")
