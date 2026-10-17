# Yawn detection calibration

A yawn is the Mouth Aspect Ratio (MAR) staying above a threshold for a minimum time.
The threshold and duration were chosen on one YawDD group and scored once on the other.

## Method

MAR uses six inner lip landmarks (78, 82, 312, 308, 317, 87), laid out like the EAR points:
the two mouth corners, and the top and bottom lip points above and below each other.

    MAR = (|p1 - p5| + |p2 - p4|) / (2 * |p0 - p3|)

A closed mouth reads near 0. Talking reads in between. A yawn reads large and stays large.

A yawn is MAR above THRESHOLD for at least MIN_DURATION. The duration is kept in seconds,
not frames, because YawDD runs at about 30 fps and the demo runs at a different rate.

## Data

YawDD mirror videos, labeled from the filename (Normal, Talking, Yawning, TalkingYawning).
Yawn-positive means Yawning or TalkingYawning. The Dash videos are unlabeled and not used.
Our copy has 319 mirror videos (155 female, 164 male); the summary lists 320, so one female
Yawning video is missing.

Per-frame MAR was computed for every video with the MediaPipe Face Landmarker in VIDEO
mode, with NaN where no face was found. The values are kept under data/yawdd_mar/, which
is not tracked by git.

YawDD has no training split, so nothing is trained. Tuning on the same videos that are
scored would overstate the result, so the female group was used to tune and the male
group to report.

## Tuning (female only)

Video-level precision and recall over a grid of thresholds (0.2 to 0.8) and minimum
durations (10 to 45 frames). Chosen: **threshold 0.35, 35 frames (about 1.2 s)**, best F1
(0.90) and inside a stable region (neighbouring settings scored 0.87 to 0.89). Rejected:
0.35 with 40 frames (precision 1.00, recall 0.77, 0/47 Talking false alarms), which trades
about four more missed yawns for two fewer false alarms.

Lower thresholds catch more yawns but trigger on talking (0.2 gave 28 of 47 Talking false alarms).

## Results

| Group | Role | Precision | Recall | Talking false alarms |
|---|---|---|---|---|
| Female | tuned on | 0.96 | 0.84 | 2 / 47 |
| Male | held out, scored once | 1.00 | 0.95 | 0 / 53 |

## Limitations

- Video-level, not frame-level: a video counts as a yawn if any run passes the rule.
- Mirror-camera clips at about 30 fps. These numbers are not a measure of live performance
  on another camera, angle or frame rate.
- The male result is better than the tuned group, which is likely chance or easier yawns
  (about 57 yawn videos per group) and not a real difference between groups.
- One missing female Yawning video.

## Reproduce

    uv run python -m drivesafe.data.yawdd_mar     # per-frame MAR for every video
    uv run python -m drivesafe.data.yawdd_tune    # tuning table (female)