# eval_val.py
from sklearn.metrics import classification_report, confusion_matrix
import torch
import json
import os
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report
import cv2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = "mobilenetv3_student_pruned.pth"
CLASSES_FILE = "class_labels.json"
VAL_DIR = "data/val"
INPUT_SIZE = (400, 400)
BATCH = 16

val_transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE[0]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH, shuffle=False)

with open(CLASSES_FILE, "r", encoding="utf-8") as f:
    classes = json.load(f)

num_classes = len(classes)
print("Classes:", classes)

model = timm.create_model('mobilenetv3_small_100',
                          pretrained=False, num_classes=num_classes)
state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.to(DEVICE).eval()

all_preds = []
all_labels = []
mis_save_dir = "misclassified_examples"
os.makedirs(mis_save_dir, exist_ok=True)

with torch.no_grad():
    for i, (imgs, labels) in enumerate(val_loader):
        imgs = imgs.to(DEVICE)
        out = model(imgs)
        probs = torch.softmax(out, dim=1)
        preds = probs.argmax(dim=1).cpu().numpy()
        labels = labels.numpy()
        for j in range(len(labels)):
            all_preds.append(int(preds[j]))
            all_labels.append(int(labels[j]))
            if preds[j] != labels[j]:
                img_np = imgs[j].cpu().permute(1, 2, 0).numpy()
                img_np = (
                    img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
                img_np = (np.clip(img_np, 0, 1) *
                          255).astype('uint8')[:, :, ::-1]
                fname = f"{mis_save_dir}/val_{i}_{j}_pred_{classes[preds[j]]}_gt_{classes[labels[j]]}.jpg"
                cv2.imwrite(fname, img_np)

print(classification_report(all_labels, all_preds, target_names=classes, digits=4))
print("Confusion matrix:\n", confusion_matrix(all_labels, all_preds))
