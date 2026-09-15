
"""
Dataset: scikit-learn's built-in `load_digits` (1,797 samples of 8x8
handwritten digits, 0-9). It ships with scikit-learn so the project runs
fully offline with no external download.

To extend to full-size MNIST (28x28) or EMNIST (letters + digits), see the
"Extending this project" section in the README — swap `load_digits_data()`
for `tf.keras.datasets.mnist.load_data()` or an EMNIST loader and the rest
of the pipeline (model, training loop, evaluation) works unchanged.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras import layers, models

RANDOM_SEED = 42
OUTPUTS_DIR = "outputs"
MODELS_DIR = "models"


def load_digits_data():
    digits = load_digits()
    X = digits.images  # shape (n_samples, 8, 8)
    y = digits.target
    X = X.astype("float32") / 16.0  # pixel values range 0-16
    X = np.expand_dims(X, -1)  # add channel dim -> (n, 8, 8, 1)
    return X, y


def build_cnn(input_shape=(8, 8, 1), num_classes=10):
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    tf.random.set_seed(RANDOM_SEED)

    X, y = load_digits_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=RANDOM_SEED, stratify=y_train
    )

    model = build_cnn()
    model.summary()

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        verbose=2,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("\n" + classification_report(y_test, y_pred))

    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUTS_DIR}/training_curves.png", dpi=150)
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix (Test Accuracy = {test_acc:.3f})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{OUTPUTS_DIR}/confusion_matrix.png", dpi=150)
    plt.close()

    # Sample predictions
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
    sample_idx = np.random.default_rng(RANDOM_SEED).choice(len(X_test), 10, replace=False)
    for ax, idx in zip(axes.flat, sample_idx):
        ax.imshow(X_test[idx].squeeze(), cmap="gray")
        color = "green" if y_pred[idx] == y_test[idx] else "red"
        ax.set_title(f"True: {y_test[idx]} Pred: {y_pred[idx]}", color=color, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{OUTPUTS_DIR}/sample_predictions.png", dpi=150)
    plt.close()

    model.save(f"{MODELS_DIR}/cnn_digit_classifier.keras")
    print(f"\nModel saved to {MODELS_DIR}/cnn_digit_classifier.keras")


if __name__ == "__main__":
    main()
