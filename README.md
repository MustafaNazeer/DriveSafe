# DriveSafe

In cabin driver monitoring for real time drowsiness and distraction detection, built to run on edge hardware.

DriveSafe watches a driver through a single cabin camera and raises tiered alerts of fatigue (eye closure, slow blinks, yawning, head nodding) and distraction (eyes off road, gaze away from the road), before either leads to an incident. It is designed to run on device with no cloud dependency, all local

This is a senior design project spanning two semesters (CSE 4316 and CSE 4317) at the University of Texas at Arlington.

## Approach

A perception stage extracts interpretable per frame signals from facial landmarks, such as, eye closure, yawning, head pose, and gaze direction. A temporal stage brings those signals together over a window to estimate fatigue and distraction

Models are trained and evaluated offline on public datasets using subject independent splits, then optimized for real time inference on the target edge device.

## Data

The datasets used for training and evaluation, and where each one comes from, are documented in [docs/data/datasets.md](docs/data/datasets.md). The raw data is large and is not stored in this repository. See [data/README.md](data/README.md)

## Status

Early development / Demo.

The current live demo supports both a laptop webcam and a Raspberry Pi 5 with a Raspberry Pi Camera Module 3. The demo detects the driver's face, extracts facial landmarks, computes eye aspect ratio (EAR), counts blinks, tracks eye closure using PERCLOS, and warns on sustained eye closure.

An Eye-State (open/closed) Classifier now exists (trained on MRL Eye, ~94% offline accuracy, integrated live)

Model training and additional fatigue/distraction features such as yawning, head pose, and gaze detection are still in development.

## Demo

The live blink and eye-closure demo can run using either:

- a laptop/desktop webcam
- a Raspberry Pi 5 with a Raspberry Pi Camera Module 3

The camera source is selected using the `--camera` argument:

- `--camera webcam` uses an OpenCV-compatible webcam and is the default.
- `--camera pi` uses the Raspberry Pi Camera Module through Picamera2.

### Windows PowerShell

```powershell
git clone https://github.com/MustafaNazeer/DriveSafe
cd DriveSafe
uv venv --python 3.13
uv pip install -e ".[dev]"
curl.exe -sL -o models\face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
# Download the MRL Eye Dataset from https://mrl.cs.vsb.cz/eyedataset.html and extract into data/mrlEyes_2018_01/
uv run python -m drivesafe.models.train_eye_state_cnn 
uv run python -m drivesafe.demo.blink_demo
```

### Linux and macOS

```bash
git clone https://github.com/MustafaNazeer/DriveSafe && cd DriveSafe && \
uv venv --python 3.13 && \
uv pip install -e ".[dev]" && \
curl -sL -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
# Download the MRL Eye Dataset from https://mrl.cs.vsb.cz/eyedataset.html and extract into data/mrlEyes_2018_01/
uv run python -m drivesafe.models.train_eye_state_cnn && \
uv run python -m drivesafe.demo.blink_demo
```

The model file is roughly 4 MB and is deliberately not stored in this repository, so that
download step is required on every machine. On later runs only the last line is needed:

```bash
uv run python -m drivesafe.demo.blink_demo
```

Select an OpenCV-compatible webcam, use:

```bash
uv run python -m drivesafe.demo.blink_demo --camera webcam
```

The overlay draws the six landmarks used for each eye, the live eye aspect ratio, a running
blink count, and a drowsiness warning once the eyes have stayed closed for `closure_frames`
consecutive frames (30 by default). That is a frame count rather than a fixed duration, so how
long it takes depends on the frame rate the pipeline actually achieves. About two seconds at
the 14.7 fps measured on the development laptop. Press `q` to quit. The default EAR threshold
of 0.21 sits between typical open and closed values, but it may need adjusting for a given
face, camera, and lighting. See `src/drivesafe/perception/perclos-calibration.md` for measured
open and closed values on one face.

### Raspberry Pi 5

The Raspberry Pi version has been tested with:

- Raspberry Pi 5 (8 GB)
- 64-bit Raspberry Pi OS
- Raspberry Pi Camera Module 3
- Picamera2
- MediaPipe Face Landmarker

Clone the repository:

```bash
git clone https://github.com/MustafaNazeer/DriveSafe
cd DriveSafe
uv venv --python /usr/bin/python3 --system-site-packages
source .venv/bin/activate
uv pip install -e ".[dev]"
curl -fL \
  -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
python -m drivesafe.demo.blink_demo --camera pi
```


## AI Assistance

Anthropic's Claude was used as a tutoring resource to explain computer vision concepts
(facial landmarks, eye aspect ratio, blink detection). All code under
`src/drivesafe/perception/` and `src/drivesafe/demo/` was written by the author.

Claude was also used to measure the PERCLOS calibration constants. Claude wrote the
throwaway measurement instrument, ran it, computed the statistics, and wrote
`src/drivesafe/perception/perclos-calibration.md`. The measurement instrument is
deliberately not part of this repository.

The dataset split tooling under `src/drivesafe/data/` was written by Claude to a
specification written by the author. It builds subject independent split manifests and
summaries from the dataset archives and is not part of the runtime detection pipeline.

### Raspberry Pi 5 Setup and Camera Integration

AI assistance (ChatGPT) was used during the setup and debugging of the Raspberry Pi 5 development environment. This included guidance for configuring 64-bit Raspberry Pi OS, creating the Python virtual environment, installing project dependencies, and verifying the Raspberry Pi Camera Module 3 through Picamera2.

ChatGPT was also used to assist with integrating the Pi camera into the existing blink detection demo. This included adding and debugging the --camera option for selecting either an OpenCV-compatible webcam (--camera webcam) or Raspberry Pi Camera Module (--camera pi), troubleshooting MediaPipe compatibility, verifying RGB/BGR image handling, and testing the MediaPipe Face Landmarker pipeline on the Raspberry Pi 5.

All AI-assisted changes were reviewed and tested on the target hardware before being incorporated into the project.