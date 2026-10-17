import csv
from pathlib import Path

import cv2
import mediapipe as mp 
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from drivesafe.perception.landmarks import to_pixel_array, mouth_aspect_ratio_from_landmarks

REPO_ROOT = Path(__file__).resolve().parents[3]
VIDEO_DIR = REPO_ROOT / "data" / "YawDD" / "Mirror"
OUT_DIR = REPO_ROOT / "data" / "yawdd_mar"
MODEL_PATH = REPO_ROOT / "models" / "face_landmarker.task"


def parse_video(path: Path):
    # 1-FemaleNoGlasses-Yawning.avi -> subject, person, action
    subject, person, action = path.stem.split("-")
    group = "female" if "Female_mirror" in path.parts else "male"
    low = action.lower()
    yawn = "yawn" in low
    talking = "talk" in low
    if yawn and talking:
        label = "TalkingYawning"
    elif yawn:
        label = "Yawning"
    elif talking:
        label = "Talking"
    else:
        label = "Normal"
    return group, subject, label, int(yawn)

def video_mar(video_path: Path) -> np.ndarray:
    # VIDEO mode needs timestamps that only increasem so each video gets a fresh landmarker
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)

    values = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(image, int(frame_idx / fps * 1000))
        if result.face_landmarks:
            h, w = frame.shape[:2]
            landmarks = to_pixel_array(result.face_landmarks[0], w, h)
            values.append(mouth_aspect_ratio_from_landmarks(landmarks))
        else:
            values.append(np.nan)
        frame_idx += 1
    cap.release()
    landmarker.close()
    return np.array(values, dtype=np.float32)

def main(limit=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    videos = sorted(VIDEO_DIR.rglob("*.avi"))
    if limit:
        videos = videos[:limit]

    for video_path in videos:
        group, subject, label, yawn = parse_video(video_path)
        out = OUT_DIR / f"{group}_{video_path.stem}.npy"

        if out.exists():
            print(f"skipping {out.name}, already saved")
            continue
        print(f"Processing {video_path.name} ({group}, {label})")
        np.save(out, video_mar(video_path))

if __name__ == "__main__":
    main()