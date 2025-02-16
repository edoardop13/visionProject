import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from collections import Counter

X_train_path = "roboflow/X_train.npy"
y_train_path = "roboflow/y_train.npy"
X_val_path   = "roboflow/X_valid.npy"
y_val_path   = "roboflow/y_valid.npy"

X_train = np.load(X_train_path)  # shape (num_samples_train, num_features)
y_train = np.load(y_train_path)  # shape (num_samples_train, )
X_val   = np.load(X_val_path)    # shape (num_samples_val, num_features)
y_val   = np.load(y_val_path)    # shape (num_samples_val, )

print(f"[DEBUG] X_train shape: {X_train.shape}")
print(f"[DEBUG] y_train shape: {y_train.shape}")
print(f"[DEBUG] X_val shape:   {X_val.shape}")
print(f"[DEBUG] y_val shape:   {y_val.shape}")

# Conta il numero di campioni per classe
train_class_distribution = Counter(y_train)
val_class_distribution = Counter(y_val)

print("[DEBUG] Distribuzione classi nel training set:")
for label, count in sorted(train_class_distribution.items()):
    print(f"Classe {label}: {count} campioni")

print("\n[DEBUG] Distribuzione classi nel validation set:")
for label, count in sorted(val_class_distribution.items()):
    print(f"Classe {label}: {count} campioni")


label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded   = label_encoder.transform(y_val)

# We save labels' mapping for future inference
np.save("roboflow_label_classes.npy", label_encoder.classes_)

print(f"[INFO] Classi (label_encoder): {label_encoder.classes_}")

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)

X_val_tensor   = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor   = torch.tensor(y_val_encoded, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset   = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)

print("[INFO] Dataloaders correctly created!")

# Model DEFINITION
class GestureClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(GestureClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(64, num_classes)
        # We don't add Softmax because we use CrossEntropyLoss (which includes log-softmax)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)

        logits = self.fc3(x)
        # return raw logits (no softmax)
        return logits

input_size  = X_train.shape[1]
num_classes = len(label_encoder.classes_)

model = GestureClassifier(input_size, num_classes)
print("[DEBUG] Model:")
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# TRAINING LOOP
num_epochs = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"[INFO] Training on device: {device}")
model.train()

for epoch in range(num_epochs):

    model.train()

    total_loss = 0.0
    correct_train = 0
    total_train = 0

    # training loop
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # reset gradient
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X_batch)  # logits
        loss = criterion(outputs, y_batch)

        # Backprop
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        # accuracy computation on batch
        _, predicted = torch.max(outputs, dim=1)
        correct_train += (predicted == y_batch).sum().item()
        total_train   += y_batch.size(0)

    avg_loss = total_loss / len(train_loader)
    train_acc = correct_train / total_train

    # End of epoch - validation
    model.eval()
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch = X_val_batch.to(device)
            y_val_batch = y_val_batch.to(device)
            val_outputs = model(X_val_batch)

            _, val_predicted = torch.max(val_outputs, dim=1)
            correct_val += (val_predicted == y_val_batch).sum().item()
            total_val += y_val_batch.size(0)

    val_acc = correct_val / total_val

    print(f"Epoch [{epoch+1}/{num_epochs}] - "
          f"Loss: {avg_loss:.4f}, "
          f"Train Acc: {train_acc:.4f}, "
          f"Val Acc: {val_acc:.4f}")


model.eval()
correct = 0
total = 0
with torch.no_grad():
    for X_val_batch, y_val_batch in val_loader:
        X_val_batch = X_val_batch.to(device)
        y_val_batch = y_val_batch.to(device)

        outputs = model(X_val_batch)
        _, predicted = torch.max(outputs, 1)
        
        correct += (predicted == y_val_batch).sum().item()
        total += y_val_batch.size(0)

final_val_acc = correct / total
print(f"[RESULT] Final accuracy on validation set: {final_val_acc:.4f}")


model_path = "ROBOFLOWgesture_model_pytorch.pth"
torch.save(model.state_dict(), model_path)
print(f"[INFO] MOdel saved at: {model_path}")

# Inference
loaded_model = GestureClassifier(input_size, num_classes)
loaded_model.load_state_dict(torch.load(model_path))
loaded_model.to(device)
loaded_model.eval()

# classes load for decode
label_classes = np.load("roboflow_label_classes.npy", allow_pickle=True)

# random input
new_input = np.random.rand(1, input_size).astype(np.float32)
new_input_tensor = torch.tensor(new_input, dtype=torch.float32).to(device)

with torch.no_grad():
    logits = loaded_model(new_input_tensor)
    probs = torch.softmax(logits, dim=1)
    predicted_class_idx = torch.argmax(probs, dim=1).item()

print("Probs:", probs.cpu().numpy())
print(f"Gesture inferenced: {label_classes[predicted_class_idx]}")
