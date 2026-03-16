## TensorFlow-&-Keras-CNN-Image-Classification
Els models i documents complementaris estan pujats a Hugging Face:
[https://huggingface.co/Oriac/cnn-image-cats-and-dogs-classifier](https://huggingface.co/Oriac/cnn-image-cats-and-dogs-classifier)
# CNN Models for Image Classification
# TensorFlow CNN Image Classification (Cats vs. Dogs)

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-API-D00000?logo=keras)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow)](https://huggingface.co/Oriac/cnn-image-cats-and-dogs-classifier)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains a complete **end-to-end pipeline for image classification** using custom **Convolutional Neural Networks (CNNs)** built with TensorFlow/Keras. The project is designed to be easily adaptable for small to medium-sized image datasets (a few thousand images) and serves as a practical template for similar tasks.

The trained model is publicly available on:
👉 TensorFlow-CNN-Image-Classification.html
**Hugging Face Hub**:  
👉 [**Oriac/cnn-image-cats-and-dogs-classifier**](https://huggingface.co/Oriac/cnn-image-cats-and-dogs-classifier)

---

## 📌 Table of Contents

- [Overview](#overview)
- [Project Goals](#project-goals)
- [Dataset & Structure](#dataset--structure)
- [Model Architecture](#model-architecture)
- [Training Pipeline](#training-pipeline)
- [Results](#results)
- [How to Use](#how-to-use)
  - [Installation](#installation)
  - [Training](#training)
  - [Inference](#inference)
- [Repository Structure](#repository-structure)
- [Model on Hugging Face](#model-on-hugging-face)
- [Technologies Used](#technologies-used)
- [Future Work](#future-work)
- [Contact](#contact)

---

## 🔍 Overview

This project implements a custom CNN from scratch to classify images of **cats and dogs**. It demonstrates:

- **Data loading and preprocessing** (image resizing, normalization, augmentation).
- **Building a custom CNN architecture** with convolutional, pooling, and dense layers.
- **Training the model** with configurable hyperparameters (epochs, batch size).
- **Evaluating performance** on a validation set.
- **Saving and loading** the trained model for inference.
- **Deploying the model** to Hugging Face Hub for easy sharing and reuse.

The code is structured to be modular and reusable for any binary (or multi-class) image classification problem.

---

## 🎯 Project Goals

1. Build a custom CNN that achieves **>90% validation accuracy** on the cats vs. dogs dataset.
2. Create a **clean, documented pipeline** that can be easily adapted to other datasets.
3. Demonstrate **best practices** in TensorFlow/Keras model development.
4. Share the trained model on **Hugging Face** to make it accessible to the community.

---

## 📂 Dataset & Structure

The model expects data in the following structure:
data/
├── train/
│ ├── class_0/ # e.g., cats
│ │ ├── img001.jpg
│ │ ├── img002.jpg
│ │ └── ...
│ └── class_1/ # e.g., dogs
│ ├── img101.jpg
│ └── ...
└── val/ # same structure as train
├── class_0/
└── class_1/

For this project, the [Dogs vs. Cats dataset](https://www.kaggle.com/c/dogs-vs-cats) from Kaggle was used. A subset of **2,000 images per class** was used for training, and **500 per class** for validation to keep training times manageable.

---

## 🧠 Model Architecture

A custom CNN was designed with the following layers:

Input (150x150x3)
│
├─ Conv2D (32 filters, 3x3, ReLU) → MaxPooling2D (2x2)
├─ Conv2D (64 filters, 3x3, ReLU) → MaxPooling2D (2x2)
├─ Conv2D (128 filters, 3x3, ReLU) → MaxPooling2D (2x2)
├─ Flatten()
├─ Dense (512 units, ReLU) → Dropout(0.5)
└─ Dense (1 unit, Sigmoid) # Binary classification

**Key features:**
- **ReLU activation** for non-linearity.
- **MaxPooling** to reduce spatial dimensions.
- **Dropout** (0.5) to prevent overfitting.
- **Sigmoid** output for binary classification.

---

## 🏋️ Training Pipeline

### Data Preprocessing
- Images resized to **150x150** pixels.
- Pixel values normalized to **[0, 1]**.
- **Data augmentation** applied to training set (rotation, zoom, flip, etc.) to improve generalization.

### Training Configuration
- **Loss function:** `binary_crossentropy`
- **Optimizer:** `Adam` (learning rate = 0.001)
- **Batch size:** 32
- **Epochs:** 30 (with early stopping)

The training script (`src/train.py`) accepts command-line arguments for easy experimentation.

---

## 📊 Results

After 30 epochs of training, the model achieved:

| Metric | Value |
|--------|-------|
| **Training Accuracy** | 96.8% |
| **Validation Accuracy** | **91.2%** |
| **Validation Loss** | 0.21 |

The training history (accuracy/loss curves) can be visualized using the provided notebook.

---

## 🚀 How to Use

### Installation

1. **Clone the repository:**

   git clone https://github.com/oriac-gimeno/TensorFlow-CNN-Image-Classification.git
   cd TensorFlow-CNN-Image-Classification
2. Install dependencies:

pip install -r requirements.txt

Training
To train the model on your own data:

python src/train.py \
    --data_dir /path/to/data \
    --epochs 30 \
    --batch_size 32 \
    --img_height 150 \
    --img_width 150 \
    --save_path models/custom_cnn.h5
Arguments:

--data_dir: path to dataset with train/ and val/ subfolders.

--epochs: number of training epochs (default: 30).

--batch_size: batch size (default: 32).

--img_height, --img_width: target image dimensions (default: 150x150).

--save_path: where to save the trained model (default: models/cnn_model.h5)

Inference
To classify a single image using a trained model:
python src/predict.py \
    --model models/cnn_model.h5 \
    --image_path /path/to/image.jpg \
    --img_height 150 \
    --img_width 150

📁 Repository Structure
text
TensorFlow-CNN-Image-Classification/
├── 📜 README.md
├── 📜 requirements.txt
├── 📓 Practica_Oriac_Gimeno_...ipynb   # Jupyter notebook with full experiment
├── 📁 src/
│   ├── train.py          # Training script
│   └── predict.py        # Inference script
├── 📁 models/             # Saved models (created after training)
└── 📁 samples/            # Example images for inference

🤗 Model on Hugging Face
The fully trained model is available on the Hugging Face Hub for easy integration into other projects:

🔗 Oriac/cnn-image-cats-and-dogs-classifier

You can load and use the model directly with the transformers library (or with native TensorFlow/Keras):

python
import tensorflow as tf

model = tf.keras.models.load_model("Oriac/cnn-image-cats-and-dogs-classifier")
# or using Hugging Face's from_pretrained() if a proper model card is set up
The Hugging Face repository includes:

The model weights (.h5 format).

Configuration files.

A model card (this README will also be adapted there).

🛠️ Technologies Used
TensorFlow 2.x / Keras – deep learning framework.

Python 3.8+ – core programming language.

NumPy, Pillow – data manipulation and image handling.

Matplotlib – visualization of training history.

Hugging Face Hub – model hosting and sharing.

Git LFS – for handling large model files.

🔮 Future Work
Experiment with transfer learning (e.g., MobileNetV2, ResNet50) to improve accuracy with limited data.

Add more classes (e.g., multi-category classification).

Deploy the model as a simple web app (e.g., using Gradio or Streamlit).

Create a Docker container for reproducible environments.

Write a more detailed Hugging Face model card with example usage and performance metrics.

📬 Contact
GitHub: oriac-gimeno

LinkedIn: Oriac Gimeno

Portfolio: https://oriac-gimeno.github.io/

Hugging Face: Oriac

⭐ If you find this project useful, please consider giving it a star on GitHub!
