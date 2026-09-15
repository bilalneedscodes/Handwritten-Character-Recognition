**Task:** Identify handwritten characters/digits using image processing and deep learning.

CodeAlpha Machine Learning Internship — Task 3

## 📌 Overview

This project trains a **Convolutional Neural Network (CNN)** to recognize
handwritten digits (0-9), using scikit-learn's built-in `load_digits`
dataset (1,797 samples of 8x8 grayscale digit images) — so it runs fully
offline with **no external download** required.

## 🗂 Project Structure

```
├── src/
│   └── train_model.py     # Loads data, builds CNN, trains, evaluates
├── data/                  # (unused — dataset ships with scikit-learn)
├── models/                # Saved CNN model (.keras)
├── outputs/                # Training curves, confusion matrix, sample predictions
├── requirements.txt
└── README.md
```

## 🚀 How to Run

```bash
pip install -r requirements.txt
python src/train_model.py
```

## 🧠 Model Architecture

```
Conv2D(32) → BatchNorm → Conv2D(32) → MaxPool → Dropout
→ Conv2D(64) → BatchNorm → Flatten → Dense(128) → Dropout → Dense(10, softmax)
```

Trained for 30 epochs with the Adam optimizer and sparse categorical
cross-entropy loss.

## 📈 Results

**Test accuracy: ~99.2%** on the held-out digits test set.

Outputs generated in `outputs/`:
- `training_curves.png` — accuracy/loss over epochs (train vs validation)
- `confusion_matrix.png` — per-digit performance
- `sample_predictions.png` — 10 random test images with true vs predicted labels

## 🔧 Extending to Full MNIST / EMNIST / Word Recognition

The task brief also mentions MNIST (28x28 digits), EMNIST (letters), and
extending to full word/sentence recognition:

- **MNIST**: replace `load_digits_data()` with
  `tf.keras.datasets.mnist.load_data()` and adjust `input_shape` to `(28, 28, 1)`.
  (Requires internet access to download the dataset the first time.)
- **EMNIST** (letters + digits): use the `emnist` PyPI package or
  TensorFlow Datasets' `emnist` loader; same CNN architecture applies with
  `num_classes` changed to match the character set.
- **Full word/sentence recognition**: swap the CNN classifier for a
  **CRNN** (CNN + BiLSTM + CTC loss) to handle variable-length sequences of
  characters instead of single fixed-size images.

## 🛠 Tech Stack

Python, TensorFlow/Keras, scikit-learn, NumPy, matplotlib, seaborn

## ✍️ Author

Built as part of the CodeAlpha Machine Learning Internship.
