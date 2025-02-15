import os
import json
import yaml
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from ultralytics import YOLO

def _load_annotations_from_folder(annotation_folder, output_label_dir, split_dir):
    annotation_dict = {}
    ann_files = [f for f in os.listdir(annotation_folder) if f.endswith(".json")]
    os.makedirs(output_label_dir, exist_ok=True)

    class_names = [
    "call", "dislike", "fist", "four", "like", "mute", "no_gesture", "ok", "one", "palm",
    "peace", "peace_inverted", "rock", "stop", "stop_inverted", "three", "three2", "two_up", "two_up_inverted"
    ]
    label_map = {name: i for i, name in enumerate(class_names)}  

    for ann_file in ann_files:
        with open(os.path.join(annotation_folder, ann_file), 'r') as f:
            data = json.load(f)

        for img_id, content in data.items():
            if img_id not in annotation_dict:
                annotation_dict[img_id] = {"labels": [], "bboxes": []}

            annotation_dict[img_id]["labels"].extend(content["labels"])
            annotation_dict[img_id]["bboxes"].extend(content["bboxes"])

            label_file_path = os.path.join(output_label_dir, split_dir, "labels", f"{img_id}.txt")
            os.makedirs(os.path.dirname(label_file_path), exist_ok=True)
            with open(label_file_path, "w") as label_file:
                for label, bbox in zip(content["labels"], content["bboxes"]):
                    if label not in label_map:
                        print(f"WARNING: Label '{label}' not found in label_map!")
                        continue

                    class_id = label_map[label]  
                    cx, cy, w, h = bbox  
                    label_file.write(f"{class_id} {cx} {cy} {w} {h}\n")

    return annotation_dict

def create_hagrid_yaml(annotation_folder, root_dir, output_file="hagrid.yaml"):
    train_label_dir = os.path.join(root_dir, "train", "labels")
    val_label_dir = os.path.join(root_dir, "val", "labels")
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)
    annotations = _load_annotations_from_folder(annotation_folder, root_dir, "train")
    _load_annotations_from_folder(annotation_folder, root_dir, "val")
    class_names = sorted(set(label for ann in annotations.values() for label in ann["labels"]))

    yaml_data = {
        "train": os.path.abspath(os.path.join(root_dir, "train")),
        "val": os.path.abspath(os.path.join(root_dir, "val")),
        "test": os.path.abspath(os.path.join(root_dir, "test")),
        "nc": len(class_names),
        "names": class_names
    }

    with open(output_file, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
    print(f"[INFO] YAML file saved at {output_file}")

class YOLODataset(Dataset):
    def __init__(self, root_dir, split, annotation_folder, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.label_dir = os.path.join(root_dir, self.split, "labels")
        self.annotations = _load_annotations_from_folder(annotation_folder, root_dir, self.split)

        all_labels = set()
        for ann in self.annotations.values():
            all_labels.update(ann["labels"])
        self.label_to_idx = {lab: i for i, lab in enumerate(sorted(all_labels))}

        split_dir = os.path.join(self.root_dir, self.split)
        self.image_files = [os.path.join(root, f) for root, _, files in os.walk(split_dir) for f in files if f.lower().endswith(('jpg', 'png', 'jpeg'))]

        self.df_data = [
            {'image_path': img_path, 'img_id': os.path.splitext(os.path.basename(img_path))[0]}
            for img_path in self.image_files if os.path.splitext(os.path.basename(img_path))[0] in self.annotations
        ]
        print(f"[DEBUG] YOLODataset '{self.split}' has {len(self.df_data)} samples.")
        print(f"[DEBUG] Checking first 5 annotations:")
        for i, d in enumerate(self.df_data[:5]):
            print(f"  Image: {d['image_path']}, Label File: {os.path.join(self.label_dir, d['img_id'] + '.txt')}")
            with open(os.path.join(self.label_dir, d['img_id'] + ".txt"), "r") as f:
                print(f"    {f.read()}")

    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, idx):
        data_info = self.df_data[idx]
        img_bgr = cv2.imread(data_info['image_path'])
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = transforms.ToPILImage()(img_rgb)
        img_tensor = self.transform(img_rgb) if self.transform else torch.tensor(img_rgb).permute(2, 0, 1).float() / 255.0

        ann = self.annotations[data_info['img_id']]
        boxes = torch.tensor(ann["bboxes"], dtype=torch.float32)
        labels = torch.tensor([self.label_to_idx[label] for label in ann["labels"]], dtype=torch.long)

        print(f"[DEBUG] Sample {data_info['img_id']} - Boxes: {boxes.shape}, Labels: {labels.shape}")
        return img_tensor, {"boxes": boxes, "labels": labels}

def main():
    root_dir = "hagrid-sample/hagrid-sample-500k-384p/split/"
    annotation_folder = "hagrid-sample/hagrid-sample-500k-384p/ann_train_val/"

    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor()
    ])

    train_dataset = YOLODataset(root_dir, "train", annotation_folder, transform=transform)
    val_dataset = YOLODataset(root_dir, "val", annotation_folder, transform=transform)
    test_dataset = YOLODataset(root_dir, "val", annotation_folder, transform=transform)

    num_classes = len(train_dataset.label_to_idx)
    print(f"[INFO] Number of classes = {num_classes}")

    model = YOLO("yolov10n.pt")  

    model.train(data="hagrid.yaml", epochs=5, imgsz=384, batch=8, device="cuda" if torch.cuda.is_available() else "cpu", amp=True)

    print("Training completato.")
    model.save("yolov10n_HaGRID_best.pt")
    print("Modello salvato.")

if __name__ == "__main__":
    main()
