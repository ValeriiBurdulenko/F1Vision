import cv2
import numpy as np
import sys
import time
import torch
import torch.nn as nn
import timm
import torch.optim as optim
import torch.nn.functional as F
import os
import json
from ultralytics import YOLO

# --- SYSTEM AND MODEL CONSTANTS ---
MODEL_PATH = "mobilenetv3_student_pruned.pth"
CLASS_NAMES_FILE = "class_labels.json"
INPUT_SIZE = (300)
# Mean and Standard Deviations used during training
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

# Global variables for class names and count, will be loaded dynamically
CLASS_NAMES = []
NUM_CLASSES = 0

# --- DETECTION CONSTANTS (YOLO) ---
YOLO_MODEL_NAME = 'yolov8n.pt'
YOLO_TARGET_CLASSES = [2, 3, 5, 7]
YOLO_CONFIDENCE_THRESHOLD = 0.3

# --- SELF-LEARNING CONSTANTS ---
SELF_LEARNING_THRESHOLD = 0.95
MAX_SAMPLES_IN_BUFFER = 32
FINETUNE_INTERVAL_FRAMES = 100
FINETUNE_EPOCHS = 1
SELF_LEARNING_LR = 1e-5
MIN_BUFFER_SIZE_TO_FINETUNE = 16

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# --- HELPER CLASSES AND FUNCTIONS ---

class PseudoLabelDataset(torch.utils.data.Dataset):
    """Class for treating the accumulated buffer as a dataset."""

    def __init__(self, data_buffer):
        self.data = data_buffer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # The buffer stores (input_tensor, pseudo_label_index)
        return self.data[idx]


def load_class_names(file_path):
    """Dynamically loads class names from a JSON file."""
    global CLASS_NAMES, NUM_CLASSES
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            names = json.load(f)
        if not isinstance(names, list) or not names:
            raise ValueError(
                "JSON must contain a non-empty list of class names.")

        CLASS_NAMES = names
        NUM_CLASSES = len(names)
        print(
            f"✅ Class names successfully loaded from {file_path} (JSON). Total classes: {NUM_CLASSES}")
    except (FileNotFoundError, json.JSONDecodeError, ValueError, Exception) as e:
        print(
            f"❌ WARNING: Could not load classes from {file_path}. Reason: {e}")
        print("Using default class names (10 classes).")
        CLASS_NAMES = [f"Class {i}" for i in range(10)]
        NUM_CLASSES = len(CLASS_NAMES)

    return CLASS_NAMES, NUM_CLASSES


def load_pytorch_model(model_path, num_classes, device):
    """Initializes the MobileNetV3 architecture and loads saved weights."""
    try:
        model = timm.create_model('mobilenetv3_small_100', pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)

        model.to(device).eval()
        print(
            f"✅ PyTorch classification model successfully loaded from {model_path} onto {device}.")
        return model
    except FileNotFoundError:
        print(f"❌ Error: Model file {model_path} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading PyTorch model: {e}")
        sys.exit(1)


def preprocess_image(frame_fragment, device):
    """Preprocesses a frame fragment for model input."""
    rgb_fragment = cv2.cvtColor(frame_fragment, cv2.COLOR_BGR2RGB)
    resized_fragment = cv2.resize(rgb_fragment, INPUT_SIZE)

    # Normalization and Standardization
    normalized_fragment = resized_fragment.astype(np.float32) / 255.0
    standardized_fragment = (normalized_fragment - MEAN) / STD

    # HWC -> CHW, add batch dimension, convert to PyTorch Tensor
    input_data = np.transpose(standardized_fragment, (2, 0, 1))
    input_tensor = torch.from_numpy(input_data).unsqueeze(0).to(device)

    return input_tensor


def run_inference(model, input_tensor):
    """Performs classification inference."""
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_tensor = torch.max(probabilities, 1)

        predicted_index = predicted_tensor.item()
        confidence = confidence.item()

    return predicted_index, confidence


def run_self_finetuning(model, data_buffer, device):
    """Performs a quick fine-tuning cycle on the accumulated frames."""
    if len(data_buffer) < MIN_BUFFER_SIZE_TO_FINETUNE:
        print(
            f"[{time.strftime('%H:%M:%S')}] Self-learning skipped: Insufficient data.")
        return data_buffer

    print(f"\n[{time.strftime('%H:%M:%S')}] 🔥 Starting Self-Finetuning on {len(data_buffer)} samples...")

    dataset = PseudoLabelDataset(data_buffer)
    batch_size = min(MIN_BUFFER_SIZE_TO_FINETUNE, len(data_buffer))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=SELF_LEARNING_LR)
    criterion = nn.CrossEntropyLoss()

    model.train()
    finetune_loss = 0.0

    for epoch in range(FINETUNE_EPOCHS):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_epoch_loss = running_loss / len(dataloader)
        finetune_loss += avg_epoch_loss
        print(
            f"  > Epoch {epoch+1}/{FINETUNE_EPOCHS}, Loss: {avg_epoch_loss:.4f}")

    model.eval()
    print(f"[{time.strftime('%H:%M:%S')}] Self-Finetuning complete. Average Loss: {finetune_loss / FINETUNE_EPOCHS:.4f}\n")

    return []  # Clear the buffer

# --- MAIN FUNCTION ---


def main():
    """Main function for video/webcam inference with self-learning using YOLO detection."""
    global CLASS_NAMES, NUM_CLASSES

    if len(sys.argv) < 2:
        print("Usage:")
        print("  For webcam: python real_time_inference_and_self_learn.py webcam")
        print(
            "  For video file: python real_time_inference_and_self_learn.py <path_to_video>")
        return

    # 1. Load class names and PyTorch classification model
    load_class_names(CLASS_NAMES_FILE)
    cls_model = load_pytorch_model(MODEL_PATH, NUM_CLASSES, DEVICE)

    # 2. Load YOLO detection model
    try:
        yolo_model = YOLO(YOLO_MODEL_NAME)
        print(f"✅ YOLO detection model {YOLO_MODEL_NAME} loaded.")
    except Exception as e:
        print(
            f"❌ Error loading YOLO model. Ensure 'ultralytics' is installed: {e}")
        sys.exit(1)

    # 3. Handle input source
    input_source = sys.argv[1]
    if input_source.lower() == 'webcam':
        cap = cv2.VideoCapture(0)
        source_name = "Webcam"
    else:
        cap = cv2.VideoCapture(input_source)
        source_name = input_source

    if not cap.isOpened():
        print(f"❌ Error: Could not open source {source_name}")
        return

    # Control variables for the loop
    frame_count = 0
    start_time = time.time()
    finetune_frame_counter = 0
    self_learning_buffer = []

    print(f"--- Starting Analysis: {source_name} (YOLO + Classification) ---")

    # Main frame processing loop
    while True:
        ret, frame = cap.read()

        if not ret:
            print("End of video stream or frame read error.")
            break

        # --- 1. YOLO DETECTION ---
        results = yolo_model(
            frame,
            conf=YOLO_CONFIDENCE_THRESHOLD,
            classes=YOLO_TARGET_CLASSES,
            verbose=False
        )

        cars_detected = 0

        # --- 2. ITERATION AND CLASSIFICATION ---
        if results and results[0].boxes:
            cars_detected = len(results[0].boxes)

            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                yolo_conf = box.conf.item()

                car_fragment = frame[y1:y2, x1:x2].copy()

                if car_fragment.size == 0:
                    continue

                input_tensor = preprocess_image(car_fragment, DEVICE)
                predicted_index, confidence = run_inference(
                    cls_model, input_tensor)

                # --- THE LOGIC OF SELF-LEARNING
                if confidence > SELF_LEARNING_THRESHOLD:
                    tensor_to_save = input_tensor.squeeze(0).cpu()
                    label_to_save = torch.tensor(predicted_index).cpu()

                    if len(self_learning_buffer) < MAX_SAMPLES_IN_BUFFER:
                        self_learning_buffer.append(
                            (tensor_to_save, label_to_save))

                # --- VISUALIZATION ---
                predicted_class = CLASS_NAMES[predicted_index]

                color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Add text above the rectangle
                label = f"{predicted_class}: {confidence:.2f} (YOLO:{yolo_conf:.2f})"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        # --- 3. Global logik self-learning (Trigger) ---

        # Trigger: Buffer full
        if len(self_learning_buffer) >= MAX_SAMPLES_IN_BUFFER:
            self_learning_buffer = run_self_finetuning(
                cls_model, self_learning_buffer, DEVICE)

        # Trigger: Frame interval reached
        finetune_frame_counter += 1
        if finetune_frame_counter >= FINETUNE_INTERVAL_FRAMES:
            if len(self_learning_buffer) > 0:
                self_learning_buffer = run_self_finetuning(
                    cls_model, self_learning_buffer, DEVICE)
            finetune_frame_counter = 0

        # --- 4. VISUALIZATION OF GLOBAL RESULTS ---

        # FPS calculation
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1.0:
            fps = frame_count / elapsed_time
            start_time = time.time()
            frame_count = 0

        fps_text = f"FPS: {fps:.1f}" if 'fps' in locals() else "FPS: Calc..."
        cv2.putText(frame, fps_text, (frame.shape[1] - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Buffer status
        buffer_status = f"Buffer: {len(self_learning_buffer)}/{MAX_SAMPLES_IN_BUFFER}"
        cv2.putText(frame, buffer_status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        # Total number of objects detected
        objects_count = f"Cars Detected: {cars_detected}"
        cv2.putText(frame, objects_count, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('YOLO + PyTorch Classification (Press Q to exit)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Inference complete.")


if __name__ == '__main__':
    main()
