# Datasets

Where DriveSafe's training and evaluation data comes from. The raw data is not stored in this repository (it is large and, in some cases, license restricted). This file records the source of each dataset so the provenance is clear. The data itself is kept on the team's storage drive.

## In use

- **MRL Eye Dataset** (open or closed eye state). Source: Media Research Lab, VSB Technical University of Ostrava, https://mrl.cs.vsb.cz/eyedataset.html. Free.
- **UTA-RLDD (Real Life Drowsiness Dataset)** (temporal fatigue labels). Source: the project page at https://sites.google.com/view/utarldd/home, obtained via the Kaggle mirror `rishab260/uta-reallife-drowsiness-dataset` (CC0). Folds 1 to 4, 48 subjects.
- **YawDD (Yawning Detection Dataset)** (yawn signal). Source: IEEE DataPort, obtained via the Kaggle mirror `enider/yawdd-dataset` (MIT).
- **DMD (Driver Monitoring Dataset)** (distraction and gaze). Source: Vicomtech, https://dmd.vicomtech.org, academic use. Distraction and Gaze and Hands bundles, unrestricted subjects only.

## Pending approval

- **DGW (Driver Gaze in the Wild)** (gaze zone estimation for the eyes off road signal). Approved by the authors; access is being arranged.
- **NTHU-DDD (Driver Drowsiness Detection)** (a recognized drowsiness benchmark). Requires a signed license agreement; in progress.

## Handling

- No dataset bytes are committed to this repository.
- Each dataset is used under its stated license, for academic, non commercial research only.
- Splits are made at the subject level, so no subject appears in more than one split.
