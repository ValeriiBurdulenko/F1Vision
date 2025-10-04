import torch.nn.utils.prune as prune
import torch.nn.functional as F
import torch
import torch.nn as nn
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import json
import argparse
from torch.utils.data.sampler import WeightedRandomSampler

# --- GLOBAL CONSTANTS ---
CLASS_NAMES_FILE = "class_labels.json"
STUDENT_MODEL_PATH = "mobilenetv3_student_pruned.pth"
ONNX_OUTPUT_PATH = "mobilenetv3_student_pruned_fp32.onnx"

# Model Hyperparameters
INPUT_SIZE = (300, 300)
BATCH_SIZE = 16
LEARNING_RATE = 1e-4

# Knowledge Distillation (KD) Parameters
KD_TEMPERATURE = 4.0
KD_ALPHA = 0.5

# Pruning Parameters
PRUNING_AMOUNT = 0.3  # 30% structured pruning

# Export Parameters
OPSET_VERSION = 14

# Data Normalization Constants
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_sample_weights(dataset):
    """Рассчитывает веса для WeightedRandomSampler на основе частоты классов."""
    # Получаем метки классов для всего набора данных
    targets = dataset.targets

    # Считаем количество образцов в каждом классе
    class_sample_count = torch.tensor(
        [len(torch.where(torch.tensor(targets) == t)[0])
         for t in torch.unique(torch.tensor(targets))]
    )

    # Рассчитываем веса классов: 1.0 / количество образцов
    weight = 1.0 / class_sample_count.float()

    # Создаем тензор весов для каждого образца
    samples_weight = weight[targets]

    print(
        f"✅ Class balancing applied. Max weight: {weight.max().item():.4f}, Min weight: {weight.min().item():.4f}")
    return samples_weight.tolist()


def main():
    # Setup argument parser for export type
    parser = argparse.ArgumentParser(description="Train and export model")
    parser.add_argument(
        "--export", choices=["onnx", "pth"], required=True,
        help="Export type: 'onnx' or 'pth'"
    )
    args = parser.parse_args()

    # --- 1. Data Preparation and Device Setup ---
    # Aggressive data augmentation for the training set (Teacher and Student)
    aggressive_transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.RandomAffine(
            degrees=20,                         # Stronger rotation
            translate=(0.1, 0.1),               # Shift up to 20%
            scale=(0.9, 1.1),                   # Scaling
            shear=5                            # Shear
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.RandomRotation(15),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.2, scale=(
            0.02, 0.33), ratio=(0.3, 3.3), value='random')
    ])

    # Validation/Inference transformation (simple resize and normalize)
    val_transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    # Dummy Dataset for testing without actual data
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size=10):
            self.size = size
            self.classes = [f"Class {i+1}" for i in range(10)]

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return torch.randn(3, *INPUT_SIZE), torch.randint(0, 10, (1,)).item()

    # Try loading real data or use dummy data if folders are missing
    try:
        train_dataset = datasets.ImageFolder(
            'data/train', transform=aggressive_transform)
        val_dataset = datasets.ImageFolder('data/val', transform=val_transform)
        # --- ПРИМЕНЕНИЕ BALANCING (WeightedRandomSampler) ---
        sample_weights = get_sample_weights(train_dataset)

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),  # Размер выборки для эпохи
            replacement=True
        )

        # Создаем DataLoader, используя Sampler и отключая shuffle
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler,  # <-- WeightedRandomSampler
            shuffle=False    # <-- Обязательно False при использовании sampler
        )
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Set dynamic epoch counts based on the number of classes for a reasonable run time
        num_epochs_teacher = max(2, min(10, len(train_dataset.classes)))
        num_epochs_student = max(1, min(5, len(train_dataset.classes) // 2))

    except Exception:
        print("WARNING: 'data/train'/'data/val' folders not found. Using dummy data.")

        train_dataset = DummyDataset()
        val_dataset = DummyDataset()
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        num_epochs_teacher = 1
        num_epochs_student = 1

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_classes = len(train_dataset.classes)

    # --- 2. Teacher Model Initialization and Training (EfficientNet-B3) ---
    teacher_model = timm.create_model('efficientnet_b3', pretrained=True)
    # Adjust the classifier head to match the number of classes
    teacher_model.classifier = nn.Linear(
        teacher_model.classifier.in_features, num_classes)
    teacher_model = teacher_model.to(device)

    criterion = nn.CrossEntropyLoss()
    teacher_optimizer = torch.optim.Adam(
        teacher_model.parameters(), lr=LEARNING_RATE)

    print("--- Starting Teacher Model Training ---")
    for epoch in range(num_epochs_teacher):
        teacher_model.train()
        running_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            teacher_optimizer.zero_grad()
            outputs = teacher_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            teacher_optimizer.step()
            running_loss += loss.item()
            # Break condition for dummy data to speed up the process
            if isinstance(train_dataset, DummyDataset) and running_loss > 0 and len(train_loader) > 0:
                break

        if not isinstance(train_dataset, DummyDataset):
            print(
                f"Teacher Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

    # --- 3. Knowledge Distillation for Student Model (MobileNetV3) ---
    student_model = timm.create_model('mobilenetv3_small_100', pretrained=True)
    # Adjust the classifier head
    student_model.classifier = nn.Linear(
        student_model.classifier.in_features, num_classes)
    student_model = student_model.to(device)

    student_optimizer = torch.optim.Adam(
        student_model.parameters(), lr=LEARNING_RATE)

    print("--- Starting Knowledge Distillation ---")
    teacher_model.eval()
    student_model.train()

    for epoch in range(num_epochs_student):
        running_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            student_optimizer.zero_grad()

            with torch.no_grad():
                teacher_outputs = teacher_model(images)

            student_outputs = student_model(images)

            # Cross Entropy Loss (regular classification loss)
            loss_ce = criterion(student_outputs, labels)

            # Knowledge Distillation Loss (KL Divergence)
            loss_kd = F.kl_div(F.log_softmax(student_outputs / KD_TEMPERATURE, dim=1),
                               F.softmax(teacher_outputs /
                                         KD_TEMPERATURE, dim=1),
                               reduction='batchmean') * (KD_TEMPERATURE**2)

            # Combined loss
            loss = KD_ALPHA * loss_ce + (1 - KD_ALPHA) * loss_kd
            loss.backward()
            student_optimizer.step()
            running_loss += loss.item()
            # Break condition for dummy data
            if isinstance(train_dataset, DummyDataset) and running_loss > 0 and len(train_loader) > 0:
                break

        if not isinstance(train_dataset, DummyDataset):
            print(
                f"Student Epoch {epoch+1}, Distillation Loss: {running_loss/len(train_loader):.4f}")

    # --- 4. Pruning ---
    print("--- Starting Student Model Pruning ---")
    modules_to_prune = []

    # Step A: Apply structured pruning (L1 norm based) and collect modules
    for name, module in student_model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune.ln_structured(module, name='weight',
                                amount=PRUNING_AMOUNT, n=2, dim=0)
            modules_to_prune.append(module)

    # Step B: Permanently remove pruning buffers, "baking" the sparsified weights
    # This is crucial for ONNX export and standard PTH saving
    print("--- Removing Pruning Buffers for Export ---")
    for module in modules_to_prune:
        prune.remove(module, 'weight')

    print("Check: Pruning successfully removed.")

    # --- Save Class Names to JSON ---
    try:
        with open(CLASS_NAMES_FILE, "w", encoding="utf-8") as f:
            json.dump(train_dataset.classes, f, ensure_ascii=False, indent=2)
        print(f"✅ Class names successfully saved to {CLASS_NAMES_FILE}.")
    except Exception as e:
        print(f"❌ Error saving class names: {e}")

    # --- 5. Export based on argument ---
    if args.export == "onnx":
        print("--- Exporting to ONNX with Tracing ---")
        student_model.eval()
        # Create a dummy input tensor for tracing
        dummy_input = torch.randn(1, 3, *INPUT_SIZE, device=device)
        output_path = ONNX_OUTPUT_PATH

        # Use torch.jit.trace on the CPU model for better ONNX compatibility
        traced_model = torch.jit.trace(student_model.cpu(), dummy_input.cpu())

        # Export the traced model
        torch.onnx.export(
            traced_model,
            dummy_input.cpu(),  # Use CPU tensor
            output_path,
            export_params=True,
            opset_version=OPSET_VERSION,   # Use the defined Opset version
            input_names=['input'],
            output_names=['output']
        )

        print(f"✅ Model successfully exported to {output_path} via Tracing.")
    else:
        print("--- Saving to .pth ---")
        torch.save(student_model.state_dict(), STUDENT_MODEL_PATH)
        print(f"✅ Model saved as {STUDENT_MODEL_PATH}")


if __name__ == "__main__":
    main()
