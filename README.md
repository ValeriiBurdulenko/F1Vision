# F1Vision

# Efficient Mobile Perception: Pruning, Distillation, and Self-Learning

This project implements an **efficient image classification pipeline** focused on **Model Compression** and **Real-Time Deployment with Self-Learning**.  
It follows a multi-stage approach to create a highly optimized model suitable for **edge devices**.

---

## 🧩 Pipeline Overview

The project pipeline consists of the following stages:

1. **Training and Export (`train_export.py`)**  
   Trains a powerful **Teacher Model** (EfficientNet-B3), applies **Knowledge Distillation (KD)** to train a compact **Student Model** (MobileNetV3), and performs **Structured Pruning** for further compression.

2. **Real-Time Inference and Self-Learning (`real_time_inference_and_self_learn.py`)**  
   Deploys the compressed MobileNetV3 for real-time video analysis, integrates **YOLOv8** for object detection, and implements a **Self-Learning** (pseudo-labeling) mechanism for runtime adaptation.

3. **Evaluation (`eval_val.py`)**  
   Provides comprehensive performance metrics for the final compressed model.

---


## Requirements

- Python **3.8+**
- It is **highly recommended** to use a virtual environment.
- All dependencies are listed in `requirements.txt`.

## Installation

```bash
# Create and activate a virtual environment (recommended)
# python -m venv env
# source env/bin/activate      # Linux/macOS
# .\\env\\Scripts\\activate       # Windows
```

## Install dependencies
`pip install -r requirements.txt`

## Project Structure

<p data-start="206" data-end="849">.<br>
├── data/<br>
│   ├── train/            # Training images (ImageFolder format: class_A/, class_B/, ...)<br>
│   └── val/              # Validation images (ImageFolder format)<br>
├── train_export.py       # Model training, KD, pruning, and export<br>
├── real_time_inference_and_self_learn.py # Real-time inference with YOLO, classification, and self-learning<br>
├── eval_val.py           # Model evaluation (classification report, confusion matrix)<br>
├── requirements.txt      # Python dependencies<br>
├── mobilenetv3_student_pruned.pth  # (Generated) Final compressed PyTorch model<br>
├── class_labels.json     # (Generated) Mapping of class indices to names<br>
└── ...</p>
.
<code class="whitespace-pre!"><span><span>.
├── data/
│   ├── train/            </span><span><span class="hljs-comment"># Training images (ImageFolder format: class_A/, class_B/, ...)</span></span><span>
│   └── val/              </span><span><span class="hljs-comment"># Validation images (ImageFolder format)</span></span><span>
├── train_export.py       </span><span><span class="hljs-comment"># Model training, KD, pruning, and export</span></span><span>
├── real_time_inference_and_self_learn.py </span><span><span class="hljs-comment"># Real-time inference with YOLO, classification, and self-learning</span></span><span>
├── eval_val.py           </span><span><span class="hljs-comment"># Model evaluation (classification report, confusion matrix)</span></span><span>
├── requirements.txt      </span><span><span class="hljs-comment"># Python dependencies</span></span><span>
├── mobilenetv3_student_pruned.pth  </span><span><span class="hljs-comment"># (Generated) Final compressed PyTorch model</span></span><span>
├── class_labels.json     </span><span><span class="hljs-comment"># (Generated) Mapping of class indices to names</span></span><span>
└── </span><span><span class="hljs-punctuation">...</span></span><span>
</span></span></code>

## 1. Training and Export (train_export.py)

This script performs the full model compression pipeline.
Run the script with the desired export format:
| Command                                | Description                                                                           | Output Files                                                |
| -------------------------------------- | ------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| `python train_export.py --export pth`  | Trains and saves the model as a PyTorch state dict. *(Recommended for Self-Learning)* | `mobilenetv3_student_pruned.pth`, `class_labels.json`       |
| `python train_export.py --export onnx` | Trains and exports the model in ONNX format.                                          | `mobilenetv3_student_pruned_fp32.onnx`, `class_labels.json` |

## 2. Real-Time Inference and Self-Learning (real_time_inference_and_self_learn.py)

This script deploys the compressed MobileNetV3 for real-time video or webcam analysis.
Specify the video stream source as a command-line argument:
| Command                                                          | Description                        |
| ---------------------------------------------------------------- | ---------------------------------- |
| `python real_time_inference_and_self_learn.py webcam`            | Uses the default webcam (index 0). |
| `python real_time_inference_and_self_learn.py path/to/video.mp4` | Analyzes the given video file.     |

## 3. Model Evaluation (eval_val.py)

This script loads the final model (mobilenetv3_student_pruned.pth) and evaluates its performance on the validation dataset (data/val).

Usage
`python eval_val.py`

Output
Displays a Classification Report (accuracy, precision, recall, F1-score).<br />
Outputs a Confusion Matrix.<br />
Saves misclassified images to misclassified_examples/ for further analysis.<br />

