"""Train a small CNN for eye state classification (open/closed) on MRL Eye.

Baseline comparisons on the same subject-independent split (docs/data/mrl-summary.json):

  Logistic Regression (flattened 24x24 grayscale pixels, normalized 0-1):
      accuracy: 0.8491
      closed: precision 0.92, recall 0.78, f1 0.84
      open:   precision 0.80, recall 0.92, f1 0.86

  Linear SVC (flattened 24x24 grayscale pixels, normalized 0-1):
      accuracy: 0.8522
      closed: precision 0.91, recall 0.79, f1 0.85
      open:   precision 0.80, recall 0.92, f1 0.86

This CNN keeps the 2D image structure instead of flattening, aiming to beat both.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report
from drivesafe.models.model import EyeStateCNN

def load_split (repo_root: Path):
    """
    Return (train_subjects, test_subjects) as sets of subject IDs.

    Same split used, read from the docs/data/mrl-summary.fson.
    """
    summary_path = repo_root / "docs" / "data" / "mrl-summary.json"
    with open(summary_path) as fh:
        summary = json.load(fh)

    train_subjects = set(summary["splits"]["train"]["subjects_list"])
    test_subjects = set(summary["splits"]["test"]["subjects_list"])
    return train_subjects, test_subjects

IMG_SIZE = 48


def load_dataset(mrl_dir: Path, subjects: set[str]):
    """Return (X, y) for every image belonging to the given subjects.

    Images stay as 2D arrays (not flattened) because
    a CNN's convolution layers need the spatial structure (rows/columns) to find edges
    and shapes
    """
    features = []
    labels = []

    for subject_dir in sorted(mrl_dir.iterdir()):
        if subject_dir.name not in subjects:
            continue

        for image_path in subject_dir.glob("*.png"):
            fields = image_path.stem.split("_")
            eye_state = int(fields[4])

            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            features.append(img.astype(np.float32) / 255.0)
            labels.append(eye_state)

    return np.array(features), np.array(labels)



def main():
    repo_root = Path(__file__).resolve().parents[3]
    mrl_dir = repo_root / "data" / "mrlEyes_2018_01"

    train_subjects, test_subjects = load_split(repo_root)

    print("loading train images...")
    X_train, y_train = load_dataset(mrl_dir, train_subjects)
    print("loading test images...")
    X_test, y_test = load_dataset(mrl_dir, test_subjects)

    print(f"train: {X_train.shape[0]} images, test: {X_test.shape[0]} images")

    # Add the channel dimension conv layers expect: (N, H, W) -> (N, 1, H, W)
    X_train_t = torch.from_numpy(X_train).unsqueeze(1)
    y_train_t = torch.from_numpy(y_train).long()
    X_test_t = torch.from_numpy(X_test).unsqueeze(1)
    y_test_t = torch.from_numpy(y_test).long()

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)

    model = EyeStateCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 15
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"epoch {epoch + 1}/{epochs}, loss: {total_loss / len(train_loader):.4f}")

    model_path = repo_root / "models" / "eye_state_cnn.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")

    model.eval()
    with torch.no_grad():
        outputs = model(X_test_t)
        y_pred = outputs.argmax(dim=1).numpy()

    print(f"accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=["closed", "open"]))


if __name__ == "__main__":
    main()