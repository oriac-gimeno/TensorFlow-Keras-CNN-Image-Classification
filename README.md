# TensorFlow-Keras-CNN-Image-Classification (Cats vs. Dogs)

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-API-D00000?logo=keras)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)

A custom convolutional neural network, built from scratch with TensorFlow/Keras, that tells cats from dogs. It reaches **91.42% accuracy on a held-out test set of 3,743 images** it never saw during training.

The whole experiment lives in one notebook. There is no package to install and no training script: what is published here is the record of the run, with its outputs, so that every figure can be traced back to the cell that produced it.

👉 **[Read the full report](TensorFlow-Keras-CNN-Image-Classification.html)** (the notebook exported to HTML, with its outputs)
👉 **[Open the notebook](TensorFlow-Keras-CNN-Image-Classification.ipynb)**

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [How to Use](#how-to-use)
- [Repository Structure](#repository-structure)
- [On Hugging Face](#on-hugging-face)
- [Technologies Used](#technologies-used)
- [Future Work](#future-work)
- [Contact](#contact)

---

## 🔍 Overview

The notebook covers the full path, in Catalan, from raw folders to a scored test set:

- Cleaning the dataset and splitting it once into three disjoint sets.
- Preprocessing: resizing, normalisation and augmentation on the training set only.
- Building a custom CNN with convolutional, pooling, batch-normalisation and dense layers.
- Training for 30 epochs with early stopping, learning-rate reduction and checkpoints.
- Scoring the held-out test set and reading the confusion matrix.

---

## 📂 Dataset

The [Dogs vs. Cats dataset](https://www.kaggle.com/c/dogs-vs-cats) from Kaggle, downloaded with `kagglehub`.

The images are split **once** into three disjoint sets, and the test images are sampled first so they are never seen during training:

| Set | Images | Used for |
|---|---:|---|
| Train | 17,473 | Fitting the weights |
| Validation | 3,743 | Driving the callbacks |
| **Test** | **3,743** | Scored once, at the end |

---

## 🧠 Model Architecture

Input is **128 x 128 x 3**. Four convolutional blocks, each one `Conv2D → BatchNormalization → MaxPooling2D → Dropout(0.25)`:

```text
Input (128x128x3)
│
├─ Conv2D  (32 filters, 3x3, ReLU) → BatchNorm → MaxPool (2x2) → Dropout(0.25)
├─ Conv2D  (64 filters, 3x3, ReLU) → BatchNorm → MaxPool (2x2) → Dropout(0.25)
├─ Conv2D (128 filters, 3x3, ReLU) → BatchNorm → MaxPool (2x2) → Dropout(0.25)
├─ Conv2D (256 filters, 3x3, ReLU) → BatchNorm → MaxPool (2x2) → Dropout(0.25)
├─ Flatten()
├─ Dense (512 units, ReLU) → BatchNorm → Dropout(0.25)
└─ Dense (1 unit, Sigmoid)
```

**12,942,273 parameters.** Batch normalisation after every convolution and dropout after every block are what keep a network this size from overfitting 17,473 images.

---

## 🏋️ Training

| | |
|---|---|
| Loss | `binary_crossentropy` |
| Optimizer | `adam` (default learning rate, 0.001) |
| Batch size | 32 |
| Epochs | 30, with early stopping |
| Augmentation | rotation 15°, shear 0.2, zoom 0.2, horizontal flip, on the training set only |
| Callbacks | `EarlyStopping` (patience 10), `ReduceLROnPlateau` (on `val_accuracy`, patience 4, factor 0.75), `ModelCheckpoint`, `CSVLogger` |
| Training time | 5 h 9 min 20 s |

---

## 📊 Results

Scored once, at the end, with `model.evaluate(test_generator)` on the held-out set.

| Metric | Value |
|--------|-------|
| **Test Accuracy** (held-out, 3,743 images) | **91.42%** |
| **Test Loss** | 0.1983 |

**Confusion matrix** at a 0.5 decision threshold, over the 3,712 test images the evaluation loop scores (it drops the final partial batch of the 3,743):

|              | Predicted Cat | Predicted Dog | Recall |
|--------------|--------------:|--------------:|-------:|
| **Real Cat** |         1,771 |            83 | 95.52% |
| **Real Dog** |           244 |         1,614 | 86.87% |

Accuracy on that subset is 91.19%. The model is clearly stronger on cats than on dogs: 244 dogs are called cats, against 83 cats called dogs.

The training history (accuracy/loss curves) can be visualized using the provided notebook.

---

## 🚀 How to Use

There is no CLI and no package to install. The experiment is the notebook.

**To read it**, open [`TensorFlow-Keras-CNN-Image-Classification.html`](TensorFlow-Keras-CNN-Image-Classification.html) in any browser. It is the notebook exported with its outputs, so you get the figures and the printed results without running anything.

**To run it**, open [`TensorFlow-Keras-CNN-Image-Classification.ipynb`](TensorFlow-Keras-CNN-Image-Classification.ipynb) in Google Colab. The notebook downloads the dataset with `kagglehub` and expects a Google Drive mount for the working folders; both paths are set at the top and need editing for your own environment. Training from scratch took a little over five hours on a Colab GPU.

```sh
git clone https://github.com/oriac-gimeno/TensorFlow-Keras-CNN-Image-Classification.git
cd TensorFlow-Keras-CNN-Image-Classification
```

Note on reproducibility

The cells of the prediction and confusion-matrix section carry no outputs: the ones they had were produced with a label conversion that truncated instead of thresholding at 0.5. The source is corrected and a note in that section records the recomputed figures, but the cells need a re-run to show them again.

---

## 📁 Repository Structure

```text
TensorFlow-Keras-CNN-Image-Classification/
├── README.md
├── TensorFlow-Keras-CNN-Image-Classification.ipynb   # the experiment
└── TensorFlow-Keras-CNN-Image-Classification.html    # the same notebook, exported with its outputs
```

---

## 🤗 On Hugging Face

[**Oriac/cnn-image-cats-and-dogs-classifier**](https://huggingface.co/Oriac/cnn-image-cats-and-dogs-classifier) holds the written report of this project, as a document.

**The trained weights are not published yet.** Until they are, this repository is the place to read what the model does and how it was built.

---

## 🛠️ Technologies Used

- **TensorFlow 2.x / Keras** for the model.
- **Python 3.10**.
- **NumPy, Pillow, OpenCV** for data and image handling.
- **Matplotlib** for the training curves and the confusion matrix.
- **scikit-learn** for the confusion matrix itself.
- **kagglehub** to fetch the dataset.

---

## 🔮 Future Work

- Publish the trained weights so the model can be loaded and run.
- Re-run the prediction and confusion-matrix section to regenerate its figures under the corrected threshold.
- Try transfer learning (MobileNetV2, ResNet50) against this baseline.
- Deploy it as a small web app, with Gradio or Streamlit.

---

## 📬 Contact

- **GitHub**: [oriac-gimeno](https://github.com/oriac-gimeno)
- **LinkedIn**: [Oriac Gimeno](https://www.linkedin.com/in/oriacgimeno/)
- **Website**: [oriacgimeno.com](https://oriacgimeno.com)
- **Hugging Face**: [Oriac](https://huggingface.co/Oriac)

⭐ If you find this project useful, please consider giving it a star on GitHub!
