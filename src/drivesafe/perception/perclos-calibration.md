# PERCLOS EAR Calibration

This file records where the PERCLOS calibration constants in this package came from, how they
were measured, and why the measurement was necessary. It exists so that no number in the
PERCLOS implementation is unexplained, and so the figures quoted in the Architectural Design
Specification and the SD1 Gate Review can be traced back to a specific measurement on specific
hardware.

Measured on 2026-07-29 for CSE 4316 Sprint 4, backlog item 5 (drowsiness model), subtask E-1.

## GenAI citation (CSE 4316 course policy, "Cited Use of GenAI")

Per the course policy on cited use of Generative AI, the division of labour for this
calibration was as follows.

Anthropic's Claude (Claude Opus 5, accessed through Claude Code) did the following:

- Designed and wrote the throwaway measurement script used to collect the data.
- Ran that script, computed every statistic reported below, and produced the tables.
- Performed the follow-up analysis of blink run lengths from the raw samples.
- Wrote this document.

Mustafa Nazeer did the following:

- Chose the calibration approach: two-point P80 rather than one-point, a 60 second window,
  real wall-clock timestamps rather than assumed frame counts, and a separate `perclos` module
  rather than an extension of `BlinkDetector`.
- Sat for the measurement and supplied the face being measured.
- Authors `perclos.py`, its unit tests, and the demo overlay changes. No AI-authored source
  code is present in this repository.

The measurement script itself is deliberately **not** committed. It was a throwaway
instrument, it is AI-authored, and keeping it out of the tree preserves the authorship claim
in the project README. The protocol is documented below in enough detail to reproduce the
measurement without it.

## Why calibration is necessary at all

PERCLOS is the percentage of time within a window that the eyelids cover at least 80 percent
of the pupil. The "at least 80 percent covered" criterion is the P80 in the name.

This pipeline does not measure pupil coverage. It measures the eye aspect ratio, which is an
aperture ratio computed in `geometry.py`. Converting an EAR reading into a percent closed
therefore requires knowing what a fully open eye and a fully closed eye actually read as on
this face, with this camera, under this lighting. Those two numbers are the calibration.

Two-point normalisation, clamped to the interval [0, 1]:

```
percent_closed(ear) = (ear_open - ear) / (ear_open - ear_closed)
```

A sample counts toward the PERCLOS numerator when `percent_closed >= 0.80`, which is
algebraically equivalent to:

```
ear <= 0.2 * ear_open + 0.8 * ear_closed
```

The one-point alternative assumes `ear_closed = 0`, reducing the criterion to
`ear <= 0.2 * ear_open`. That alternative was measured too, and the results are below. It was
rejected not because it fails on this hardware, but because it leaves far less margin.

Reference for the PERCLOS definition and its validation is the Dinges and Grace work for the
US Federal Highway Administration (1998), building on earlier work by Wierwille and colleagues.
**The exact report number for that FHWA publication has not been verified and must be checked
before this reference is cited in the ADS.**

## Measurement protocol

Two independent runs were taken, roughly ten minutes apart, in the same room and lighting, to
check reproducibility.

Each run collected two phases. Every phase was preceded by an on-screen instruction stage and
a three second countdown so no phase began unannounced.

| Phase | Duration | Instruction |
|-------|----------|-------------|
| Eyes open | 20 s | Sit at normal driving distance, face the camera, blink naturally |
| Eyes closed | 12 s | Close the eyes and hold them shut for the full window |

For every frame the script called MediaPipe Face Landmarker in VIDEO mode, converted the
result through this package's existing `landmarks.to_pixel_array` and
`landmarks.average_eye_aspect_ratio`, and recorded the EAR together with a `time.monotonic()`
timestamp. No code in this package was modified for the measurement.

Frame timestamps were recorded and analysed strictly within a collection phase. Pooling
timestamps across phases would count the instruction and countdown gap between them as a single
enormous frame interval, which corrupts the mean, the standard deviation and the maximum. The
first version of the analysis had exactly that defect and reported a spurious 10,168 ms frame
interval; the figures below are from the corrected analysis.

### Environment

| Item | Value |
|------|-------|
| Camera | `/dev/video0`, 640 x 480 |
| Model | `models/face_landmarker.task`, MediaPipe Face Landmarker, 478 landmarks |
| mediapipe | 0.10.35 |
| opencv | 5.0.0 |
| numpy | 2.4.6 |
| Python | 3.13 |
| Inference | XNNPACK CPU delegate; renderer reported as Mesa Intel HD Graphics 2500 (IVB GT1) |

The frame rate figure below is specific to this machine. It will differ on the Raspberry Pi 5
target, and it must be re-measured there before any PERCLOS figure from the Pi is reported.

## Results

### EAR distributions

Both runs collected 299 open-eye samples and 179 closed-eye samples, with zero frames in which
the face was not detected.

| Statistic | Run 1 | Run 2 |
|-----------|-------|-------|
| Open, median (p50) | 0.2417 | 0.2403 |
| Open, p25 | 0.2273 | 0.2236 |
| Open, p75 | 0.2503 | 0.2512 |
| Open, p5 | 0.1136 | 0.0671 |
| Open, min / max | 0.0463 / 0.2603 | 0.0225 / 0.2717 |
| Closed, median (p50) | 0.0366 | 0.0357 |
| Closed, p95 | 0.0385 | 0.0383 |
| Closed, min / max | 0.0317 / 0.2270 | 0.0330 / 0.0485 |
| Closed, stdev | 0.0244 | 0.0018 |

The two runs agree to within 0.6 percent on the open baseline and 2.5 percent on the closed
baseline. That agreement is the reproducibility evidence for these constants.

Run 2's closed phase is the cleaner of the two. Run 1's closed phase contains a small number
of samples up to 0.2270, which are frames captured before the eyelids had fully closed; they
inflate that run's standard deviation and maximum but do not move the median. Run 2 spans only
0.0330 to 0.0485 with a standard deviation of 0.0018.

The low tail of both open-eye phases (Run 2 min 0.0225, p5 0.0671) is blinks. Blinks are real
eye closures, so their presence in the open-eye distribution is expected and is not a defect.

### Constants adopted

| Constant | Value | Source |
|----------|-------|--------|
| `ear_open` | 0.240 | median of the open-eye phase, both runs |
| `ear_closed` | 0.036 | median of the closed-eye phase, both runs |
| P80 cutoff | 0.0768 | derived: `0.2 * 0.240 + 0.8 * 0.036` |
| `window_s` | 60.0 | the window length defined in the PERCLOS literature |

These are **provisional defaults only**. They describe one subject, one camera and one lighting
condition. The demo acquires them at runtime through a keypress calibration step, and the
runtime values always take precedence over these.

### Two-point versus one-point P80

| Formula | Cutoff (Run 2) | Closed samples at or below | Open samples at or below |
|---------|----------------|----------------------------|--------------------------|
| Two-point | 0.0766 | 100.0 % | 6.02 % |
| One-point | 0.0481 | 99.4 % | not reported |

Both are reachable on this hardware. An earlier prediction that the one-point formula would
fail here, on the assumption that MediaPipe eyelid landmarks bottom out somewhere near 0.10,
was wrong: the measured closed-eye floor is 0.036. The two-point formula was still adopted,
because its cutoff of 0.0766 sits roughly twice as far above the closed-eye floor as the
one-point cutoff does, which is margin against the lighting changing on presentation day.

Separation between the two phases is clean in both runs. In Run 2, open p5 is 0.0671 against
closed p95 of 0.0383, so the EAR signal discriminates open from closed on this face.

## Frame rate, and why it matters to PERCLOS

Measured across 476 within-phase frame intervals in Run 2:

| Statistic | Value |
|-----------|-------|
| Median frame interval | 67.9 ms, that is 14.7 fps |
| Mean frame interval | 66.9 ms |
| Standard deviation | 1.8 ms |
| min / p95 / p99 / max | 63.3 / 68.2 / 68.5 / 68.9 ms |
| Jitter ratio (p95 / median) | 1.00x |
| Intervals exceeding twice the median | 0 |

PERCLOS is implemented here as a ratio of sample counts, `closed_samples / total_samples`
inside the window, rather than as a time-weighted integral. That substitution is only valid if
frames arrive at an even rate. The measured jitter ratio of 1.00x across 476 intervals, with
zero stalls, is the evidence that the assumption holds on this machine. **This assumption must
be restated and re-measured for the Raspberry Pi 5 target, where it may not hold.**

Two consequences follow from the 14.7 fps figure:

1. A 60 second window holds roughly 883 samples on this machine.
2. `blink_demo.py` currently derives its MediaPipe timestamp from a hardcoded 20 fps. At a real
   14.7 fps that skews MediaPipe's internal timeline by roughly 16 seconds over a 60 second
   window. The demo should derive its timestamp from the same monotonic clock the PERCLOS
   tracker uses.

## Why the PERCLOS threshold is not the blink threshold

`BlinkDetector` uses a single EAR threshold of 0.21. It is tempting to reuse it for PERCLOS.
The measurement shows why that would be a mistake.

Behaviour of a single threshold on the Run 2 samples:

| Threshold | Open frames at or below | Closed frames at or below | Note |
|-----------|-------------------------|---------------------------|------|
| 0.0500 | 2.34 % | 100.00 % | |
| 0.0766 | 6.02 % | 100.00 % | two-point P80 cutoff |
| 0.1000 | 8.70 % | 100.00 % | |
| 0.1400 | 9.70 % | 100.00 % | |
| 0.1800 | 13.38 % | 100.00 % | |
| 0.2100 | 19.06 % | 100.00 % | current `BlinkDetector` default |
| 0.2400 | 49.83 % | 100.00 % | |

At 0.21, 19.06 percent of open-eye frames read as closed. Analysing runs of consecutive
below-threshold frames in the open-eye phase shows that this is not sensor noise:

| Threshold | Frames at or below | Discrete runs | Mean run length | Longest run |
|-----------|--------------------|---------------|-----------------|-------------|
| 0.0766 | 18 | 11 | 111 ms | 272 ms |
| 0.1400 | 29 | 13 | 151 ms | 340 ms |
| 0.1900 | 42 | 13 | 219 ms | 407 ms |
| 0.2100 | 57 | 12 | 323 ms | 611 ms |
| 0.2200 | 67 | 12 | 379 ms | 951 ms |

The number of discrete runs stays between 11 and 13 at every threshold. There really were
about twelve blinks in that 20 second window. Raising the threshold does not invent blinks; it
inflates how long each one appears to last, because a threshold of 0.21 captures a large slice
of the eyelid's travel on a face whose open-eye EAR is only 0.24. The commonly quoted 0.21
figure derives from populations whose open-eye EAR is nearer 0.30, where it sits proportionally
much further below the open value.

The consequences differ by measure:

- **Blink counting is unaffected.** The count is stable from 0.077 to 0.22, so
  `BlinkDetector` at 0.21 with `consecutive_frames=2` counts blinks correctly.
- **PERCLOS would be badly distorted.** PERCLOS measures a fraction of time. Using 0.21 would
  give an alert baseline near 19 percent, against roughly 6 percent at the P80 cutoff. That is
  a threefold difference in the value a drowsiness alarm keys off.

This is the reason PERCLOS lives in its own module with its own closure criterion rather than
as an extension of `BlinkDetector`. The two components measure different quantities and need
different thresholds.

One coupling to be aware of if the two are ever unified: at the P80 cutoff, 6 of the 11
detected runs are a single frame long, so `consecutive_frames=2` would discard them and the
blink count would roughly halve. Threshold and `consecutive_frames` cannot be changed
independently.

## Raw data

The measurement wrote every frame to a CSV alongside the summary statistics:

```
phase,t_seconds,face_found,ear
OPEN,10.093979,1,0.254748
OPEN,10.158782,1,0.256434
...
```

479 rows for Run 2, one header plus 478 frames, covering both phases.

**Why it was necessary.** The first run saved only percentile summaries. When the blink
threshold question came up, the exact false-closure rate at 0.21 could only be bounded between
10 and 25 percent by interpolating between reported percentiles, and the blink run-length
analysis was impossible. Keeping the per-frame samples means any follow-up question about this
calibration can be answered from the data instead of costing another take in front of the
camera. Every Run 2 figure in this document has been recomputed from that CSV and matches.

Run 1 predates the CSV, so its figures are transcribed from that run's summary output and are
not independently recomputable. Run 1 is retained here only as a reproducibility cross-check on
the two medians that matter; the constants adopted above rest on Run 2.

**Why it is not tracked in this repository.** It is measurement scratch data for a single
subject, not a project artefact, and it is regenerated by re-running the protocol. Note that
the `.gitignore` in this repository does not currently exclude `*.csv`, so this file must be
kept outside the repository tree rather than relied upon to be ignored.

**Where it is kept.** `~/drivesafe-calibration/` on the development machine, as
`samples-2026-07-29-run2.csv` together with `report-2026-07-29-run2.txt`, the full tool output
for that run. That directory sits deliberately outside every git repository on the machine.
`~/Documents`, `~/src` and `~/career` are all tracked repositories with remotes, so none of
them was a safe home for data that is meant to stay untracked. Because it is untracked it
exists on one machine only and does not synchronise. Anyone reproducing this work should re-run
the protocol in the section below rather than expect to find the file.

## Limitations

These are known and should be carried into the ADS rather than discovered at the Gate Review.

1. **One subject.** All constants come from a single person. Any driver-monitoring system needs
   per-driver calibration, which is why the demo acquires the baselines at runtime rather than
   trusting the values in this file.
2. **One lighting condition and one camera.** EAR baselines shift with illumination and camera
   distance. The calibration should be repeated under the lighting used for any live
   demonstration.
3. **One machine.** The 14.7 fps figure and the 1.00x jitter ratio are properties of this
   laptop, not of the system. Both must be re-measured on the Raspberry Pi 5.
4. **Sample count rather than time weighting.** PERCLOS here counts samples, not elapsed time.
   Valid at the measured jitter, not valid in general. A time-weighted ratio is the more
   correct implementation and is future work.
5. **P80 only.** The P70 and other variants in the literature were not evaluated.
6. **The FHWA reference above is unverified** and must be confirmed before citation.

## Reproducing this measurement

1. Attach the camera and confirm `models/face_landmarker.task` is present.
2. Collect eyes-open EAR samples for 20 seconds at normal driving distance, blinking naturally,
   recording a `time.monotonic()` timestamp with every sample.
3. Collect eyes-closed EAR samples for 12 seconds, holding the eyes shut for the whole window.
4. Take `ear_open` as the median of phase 1 and `ear_closed` as the median of phase 2.
5. Derive the P80 cutoff as `0.2 * ear_open + 0.8 * ear_closed`.
6. Sanity checks: the proportion of closed-phase samples at or below the cutoff should be near
   100 percent, and phase 1's 5th percentile should sit above phase 2's 95th percentile.
7. Compute frame intervals strictly within each phase, never pooled across the gap between
   them, and report the ratio of the 95th percentile to the median as the jitter figure.
