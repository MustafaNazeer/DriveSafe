'''
Module turns MediaPipe's 478 landmarks into the 6 eye points for geometry.py to use.

Finds which vertex number make up each eye, and converts MediaPipe's coordinates into pixels

MediaPipe NOT being imported is intentional, this module is meant to be usable even without a camera
'''
import numpy as np
from drivesafe.perception.geometry import eye_aspect_ratio

# MediaPipe FaceMesh eye aspect ratio common convention values for left and right eye indices
LEFT_EYE_INDICES = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_INDICES = (362, 385, 387, 263, 373, 380)

def to_pixel_array(face_landmarks, frame_width, frame_height) -> np.ndarray:

    pixel_array = []

    # Calculate the pixels for each landmark
    for landmark in face_landmarks:
        pixel_array.append((landmark.x * frame_width, landmark.y * frame_height)) 

    return np.asarray(pixel_array, dtype=float)


def extract_points(landmarks, indices) -> np.ndarray:
    return landmarks[list(indices)]

def eye_aspect_ratios(landmarks) -> tuple[float, float]:
    # Return EAR for left and right eyes
    return (eye_aspect_ratio(extract_points(landmarks, LEFT_EYE_INDICES)), eye_aspect_ratio(extract_points(landmarks, RIGHT_EYE_INDICES)))

def average_eye_aspect_ratio(landmarks) -> float:
    x, y = eye_aspect_ratios(landmarks)
    return float((x + y) / 2)