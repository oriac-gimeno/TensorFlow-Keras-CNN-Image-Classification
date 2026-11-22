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
- [Reconciling the reported figures](#reconciling-the-reported-figures)
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
- Preprocessing: resizing and normalisation on all three sets, augmentation on the training set only.
- Building a custom CNN with convolutional, pooling, batch-normalisation and dense layers.
- Training for 30 epochs with early stopping, learning-rate reduction and checkpoints.
- Scoring the held-out test set and reading the confusion matrix.

---

## 📂 Dataset

The [Dogs vs. Cats dataset](https://www.kaggle.com/c/dogs-vs-cats) from Kaggle. The notebook imports `kagglehub` to fetch it, but the download call is commented out in the run on record, which read the images from an already populated Google Drive folder.

The images are split **once** into three disjoint sets, and the test images are sampled first so they are never seen during training:

| Set | Images | Used for | Seen per epoch |
|---|---:|---|---:|
| Train | 17,473 | Fitting the weights | 17,472 |
| Validation | 3,743 | Driving the callbacks | 3,712 |
| **Test** | **3,743** | Scored once, at the end | 3,712 |

The last column is lower because the step counts are floor divisions of the split size by the batch size of 32, so the final partial batch of each set is never reached: 546 training steps and 116 validation steps per epoch.

---

## 🧠 Model Architecture

Input is **128 x 128 x 3**. Four convolutional blocks, each one `Conv2D → BatchNormalization → MaxPooling2D → Dropout(0.25)`:

```text
Input (128x128x3)
│                                                                            after conv → after pool
├─ Conv2D  (32 filters, 3x3, ReLU) → BatchNorm → MaxPool (2x2) → Dropout       126x126  →  63x63
├─ Conv2D  (64 filters, 3x3, ReLU) → BatchNorm → MaxPool (2x2) → Dropout        61x61   →  30x30
├─ Conv2D (128 filters, 3x3, ReLU) → BatchNorm → MaxPool (2x2) → Dropout        28x28   →  14x14
├─ Conv2D (256 filters, 3x3, ReLU) → BatchNorm → MaxPool (2x2) → Dropout        12x12   →   6x6
├─ Flatten()                                                                 6x6x256   =  9,216
├─ Dense (512 units, ReLU) → BatchNorm → Dropout(0.25)                       4,719,104 parameters
└─ Dense (1 unit, Sigmoid)
```

**5,112,001 parameters**, and 4,719,104 of them, 92%, are the single dense layer that follows the flatten. The convolutions use `valid` padding, so each one trims the border before the pooling halves what is left: 128 becomes 126, then 61, 28 and 12, which is why the flatten sees 9,216 features and not the 16,384 a padded network would hand it.

Batch normalisation after every convolution and dropout after every block are what keep a network this size from overfitting 17,473 images.

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

## 🔎 Reconciling the reported figures

Training this model left eleven version folders behind, produced over four days in December 2024, and three different accuracy figures ended up in circulation: 91.42%, 93.08% and 93.53%. None of them was wrong. They measured different things, and nothing recorded which was which.

This section reconstructs the answer from the saved artifacts alone, because a number you cannot trace back to a file is not a result.

### The three figures

| Figure | What it actually is |
|---|---|
| **91.42%** | `first_try.h5`, the model as it stood at the **end of training**. This is what the notebook above documents |
| **93.08%** | `model_best_acc.keras` from the same run, the **best-validation-accuracy checkpoint** |
| **93.53%** | `model_best_acc.keras` of the 5x5 variant, evaluated on a separate held-out set |

The gap between the first two is not a better model. It is the difference between saving at the end and saving at the best epoch, in the same run.

### Six of the eleven folders are one training run

The folders v1, v2, v4, v5 and v6 look like five variants. They are not. Their layer configurations are byte-identical: same filters, same kernels, same dropout rates, same dense head, 5,112,001 parameters in all of them. The only value that differs is the learning rate stored in the optimizer state:

| Folder | Stored learning rate | Equals |
|---|---:|---|
| v1, v2 | 0.00056250 | 0.001 x 0.75² |
| v5, v6 | 0.00042187 | 0.001 x 0.75³ |
| v4 | 0.00023730 | 0.001 x 0.75⁵ |

Adam starts at 0.001 and `ReduceLROnPlateau` is configured with factor 0.75. Those three values are exactly two, three and five reductions in. They are the same training, saved at three different moments, and v1 and v2 are the same file twice.

The CSV training logs confirm it independently: they are cumulative. The log in v6 contains the runs recorded in v4 and v5 plus later ones, and the epoch counter restarts at each run boundary.

### The one architectural change that was made

Between that baseline and v10/v11, three things changed and nothing else:

| | Baseline (v1 to v6) | Variant (v10, v11) |
|---|---|---|
| Conv block 3 | 3x3, 128 filters | **5x5**, 128 filters |
| Conv block 4 | 3x3, 256 filters | **5x5**, 256 filters |
| Flatten | 9,216 features | 4,096 features |
| Dense head | 512 units | **1,024 units** |
| Parameters | 5,112,001 | 5,246,145 |

The hypothesis: the deeper blocks operate on already-downsampled feature maps and should benefit from a wider receptive field. Wider kernels consume more of the spatial dimension, so the flattened vector shrinks from 9,216 to 4,096, and doubling the dense head compensates. Net cost, 134,144 parameters, a 2.6% increase.

### What the comparison showed

Both best checkpoints, same held-out directory of 3,293 images, same preprocessing, same 0.5 threshold:

| Metric | Baseline (3x3) | Variant (5x5) |
|---|---:|---:|
| Accuracy | 93.08% | **93.53%** |
| Loss | 0.1687 | **0.1625** |
| F1 | 0.9296 | **0.9337** |
| Precision | 0.9448 | **0.9573** |
| Recall | **0.9150** | 0.9112 |
| AUC-ROC | 0.9834 | **0.9859** |

Accuracy moved 0.45 points, which alone would be noise. The movement worth reading is elsewhere: precision rose 1.25 points while recall fell 0.38. The wider model commits to the positive class less often and is right more often when it does. And AUC-ROC rose from 0.9834 to 0.9859, which is threshold-independent, so the ranking of the predictions genuinely improved rather than just their placement around a 0.5 cut.

The published weights on Hugging Face are the 5x5 variant.

---

## 🚀 How to Use

There is no CLI and no package to install. The experiment is the notebook.

**To read it**, open [`TensorFlow-Keras-CNN-Image-Classification.html`](TensorFlow-Keras-CNN-Image-Classification.html) in any browser. It is the notebook exported with its outputs, so you get the figures and the printed results without running anything.

**To run it**, open [`TensorFlow-Keras-CNN-Image-Classification.ipynb`](TensorFlow-Keras-CNN-Image-Classification.ipynb) in Google Colab. It will not run end to end as it stands, and two things need doing first:

1. **Re-enable the download.** The `kagglehub.dataset_download` call near the top is commented out, because the run on record read the images from a Drive folder that already held them. Left commented, the later cells look for images that are not there and stop at a `FileNotFoundError`.
2. **Edit the paths.** The working folders are hard-coded to a Google Drive mount and point at one specific account. They need to be your own, and they should be unique to the run: the notebook empties and repopulates those folders when it builds the splits, so two sessions pointed at the same place would overwrite each other's data.

Training from scratch took a little over five hours on a Colab GPU.

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

[**Oriac/cnn-image-cats-and-dogs-classifier**](https://huggingface.co/Oriac/cnn-image-cats-and-dogs-classifier) holds the **published weights**: `model_best_acc.keras`, the best checkpoint of the 5x5 variant, at 93.53% accuracy and 0.9859 AUC-ROC. Its model card carries the full evaluation and a loadable code example. The written report of the project is there too, as a document.

The weights published there are **not** the model this notebook documents. This notebook documents the 3x3 baseline at the end of training, 91.42%. The section above explains the difference.

---

## 🛠️ Technologies Used

- **TensorFlow 2.x / Keras** for the model.
- **Python 3.10**.
- **NumPy, Pillow, OpenCV** for data and image handling.
- **Matplotlib** for the training curves and the confusion matrix.
- **scikit-learn** for the confusion matrix itself.
- **kagglehub** for the dataset download.

---

## 🔮 Future Work

- Re-run the prediction and confusion-matrix section to regenerate its figures under the corrected threshold.
- Try transfer learning (MobileNetV2, ResNet50) against this baseline. The gap to close is large and known: fine-tuning reaches around 98% on this dataset, against 93.53% here.
- Evaluate a stock ImageNet classifier zero-shot, by mapping its dog and cat classes onto this binary task. No training at all, and it would put a second reference point beside the from-scratch number.
- Deploy it as a small web app, with Gradio or Streamlit.

---

## 📬 Contact

- **GitHub**: [oriac-gimeno](https://github.com/oriac-gimeno)
- **LinkedIn**: [Oriac Gimeno](https://www.linkedin.com/in/oriacgimeno/)
- **Website**: [oriacgimeno.com](https://oriacgimeno.com)
- **Hugging Face**: [Oriac](https://huggingface.co/Oriac)

⭐ If you find this project useful, please consider giving it a star on GitHub!
