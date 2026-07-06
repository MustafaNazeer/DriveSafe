# DriveSafe

In cabin driver monitoring for real time drowsiness and distraction detection, built to run on edge hardware.

DriveSafe watches a driver through a single cabin camera and raises tiered alerts on the onset of fatigue (eye closure, slow blinks, yawning, head nodding) and distraction (eyes off road, gaze away from the road), before either leads to an incident. It is designed to run on device with no cloud dependency.

This is a senior design project spanning two semesters (CSE 4316 and CSE 4317) at the University of Texas at Arlington.

## Approach

The detection pipeline has two stages. A perception stage extracts interpretable per frame signals from facial landmarks: eye closure, yawning, head pose, and gaze direction. A temporal stage fuses those signals over a rolling window to estimate fatigue and distraction, which drive a debounced, tiered alert state machine tuned to fire quickly while keeping false alarms low.

Models are trained and evaluated offline on public datasets using subject independent splits, then optimized for real time inference on the target edge device.

## Data

The datasets used for training and evaluation, and where each one comes from, are documented in [docs/data/datasets.md](docs/data/datasets.md). The raw data is large and is not stored in this repository; see [data/README.md](data/README.md).

## Status

Early development. Project structure is being established.
