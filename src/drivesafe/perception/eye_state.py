"""Eye state (open/closed) classification using a trained CNN.

Wraps model loading, eye cropping, and preprocessing behind one predict() call.
"""
import cv2
import numpy as np
import torch

from drivesafe.models.model import EyeStateCNN
from drivesafe.perception.landmarks import extract_points, LEFT_EYE_INDICES, RIGHT_EYE_INDICES

IMG_SIZE = 48


class EyeStateClassifier:
    def __init__(self, model_path, pad_ratio=0.3):
        self.model = EyeStateCNN()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.pad_ratio = pad_ratio

    def _crop_eye(self, gray_frame, landmarks, indices):
        points = extract_points(landmarks, indices)
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)

        # Use eye WIDTH for a square crop size — width stays stable whether the eye is
        # open or closed, unlike height, which collapses to near-zero on a closed eye
    
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        half_size = (x_max - x_min) * (1 + self.pad_ratio) / 2

        x_min = max(int(cx - half_size), 0)
        x_max = min(int(cx + half_size), gray_frame.shape[1])
        y_min = max(int(cy - half_size), 0)
        y_max = min(int(cy + half_size), gray_frame.shape[0])

        crop = gray_frame[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            return None

        return cv2.resize(crop, (IMG_SIZE, IMG_SIZE))

    def _predict_crop(self, crop):
        tensor = torch.from_numpy(crop.astype(np.float32) / 255.0)
        tensor = tensor.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            output = self.model(tensor)
            return output.argmax(dim=1).item()

    def predict(self, gray_frame, landmarks):
        """Return "open", "closed", or "n/a" (if either eye crop is invalid)."""
        left_crop = self._crop_eye(gray_frame, landmarks, LEFT_EYE_INDICES)
        
        right_crop = self._crop_eye(gray_frame, landmarks, RIGHT_EYE_INDICES)

        if left_crop is None or right_crop is None:
            return "n/a"

        left_state = self._predict_crop(left_crop)
        right_state = self._predict_crop(right_crop)
        return "closed" if (left_state == 0 and right_state == 0) else "open"