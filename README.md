# Alzheimer’s MRI Classification 

## Overview
This project builds five deep learning models to classify brain MRI scans into four stages of Alzheimer’s disease progression. The goal is to combine accurate medical image classification with model interpretability.

---

## Objectives
- Classify MRI scans into 4 categories:
  - Non Demented (Healthy)
  - Very Mild Dementia
  - Mild Dementia
  - Moderate Dementia
- Handle class imbalance
- Apply transfer learning using various ML models
- Evaluate model performance using standard metrics

---

## Dataset
Dataset: OASIS Alzheimer’s MRI Dataset

### Key Details:
- ~80,000 MRI slices from 461 patients
- Original 3D MRI scans (.nii) converted to 2D `.jpg` images
- Sliced along the z-axis (slices 100–160 used)
- Dataset size: ~1.3 GB

### Classes:
- Non Demented  
- Very Mild Dementia  
- Mild Dementia  
- Moderate Dementia  

**Note:** This is a slice-level dataset (not patient-level), which may introduce data leakage if slices from the same patient appear in both training and validation sets.

---

## Model Architecture
- Base Model: EfficientNetB0 (pre-trained on ImageNet)
- Custom Layers:
  - Conv2D (used for Grad-CAM)
  - Global Average Pooling
  - Dropout
  - Dense (Softmax output)

---

## Training Strategy

### Preprocessing
- Resize images to 224×224
- Normalize using EfficientNet preprocessing
- Data augmentation:
  - Rotation
  - Zoom
  - Contrast adjustment  

### Class Imbalance Handling
Class weights are applied during training to balance the dataset.

### Training Phases
1. Freeze base model → train classification layers  
2. Fine-tune top layers of EfficientNet with lower learning rate  

---

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## Grad-CAM (Explainability)
Grad-CAM (Gradient-weighted Class Activation Mapping) is used to visualize which regions of the MRI influenced the model’s prediction.

Outputs include:
- Original MRI image
- Heatmap
- Overlay visualization

This improves interpretability, which is critical in medical applications.

## How to Run

### Install dependencies
```bash
pip install tensorflow-macos tensorflow-metal kagglehub numpy matplotlib scikit-learn

Download Dataset:
import kagglehub
path = kagglehub.dataset_download("ninadaithal/imagesoasis")

## Train Model:
python train.py

Run Grad-CAM:
Python gradcam.py

---
Limitations:
Slice-level classification instead of patient-level diagnosis
Potential data leakage due to slicing
Class imbalance (especially Moderate Dementia)
Loss of 3D spatial information

Future Improvements:
Patient-level splitting
3D CNN models on full MRI volumes
Attention-based architectures
Improved imbalance handling (e.g., focal loss)
Deployment via web app

Team Members:
Akshaya Shankar Bharadhwaj
Leo Zheng
Nisarga Atul Patil
Tyler Crosbie
Wonjae Lee

References:
https://www.kaggle.com/datasets/ninadaithal/imagesoasis/data?select=Data
---
