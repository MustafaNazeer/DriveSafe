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

Early development / Demo. A live demo detects
the driver's face, computes the eye aspect ratio from facial landmarks, counts blinks, and
warns on sustained eye closure. Model training, distraction detection, and edge deployment
are still ahead.

## Demo

A live blink and eye closure demo runs on a laptop webcam. Clone, install, fetch the face
landmarker model, and run, in one paste.

### Windows PowerShell

```powershell
git clone https://github.com/MustafaNazeer/DriveSafe
cd DriveSafe
uv venv --python 3.13
uv pip install -e ".[dev]"
curl.exe -sL -o models\face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
uv run python -m drivesafe.demo.blink_demo
```

### Linux and macOS

```bash
git clone https://github.com/MustafaNazeer/DriveSafe && cd DriveSafe && \
uv venv --python 3.13 && \
uv pip install -e ".[dev]" && \
curl -sL -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task && \
uv run python -m drivesafe.demo.blink_demo
```

The model file is roughly 4 MB and is deliberately not stored in this repository, so that
download step is required on every machine. On later runs only the last line is needed:

```bash
uv run python -m drivesafe.demo.blink_demo
```

The overlay draws the six landmarks used for each eye, the live eye aspect ratio, a running
blink count, and a drowsiness warning once the eyes have stayed closed for `closure_frames`
consecutive frames (30 by default). That is a frame count rather than a fixed duration, so how
long it takes depends on the frame rate the pipeline actually achieves. About two seconds at
the 14.7 fps measured on the development laptop. Press `q` to quit. The default EAR threshold
of 0.21 sits between typical open and closed values, but it may need adjusting for a given
face, camera, and lighting. See `src/drivesafe/perception/perclos-calibration.md` for measured
open and closed values on one face.

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
