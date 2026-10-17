import numpy as np

from drivesafe.data.yawdd_mar import VIDEO_DIR, OUT_DIR, parse_video

THRESHOLD = 0.35
MIN_FRAMES = 35

def load_group(group):
    items = []
    for p in sorted(VIDEO_DIR.rglob("*.avi")):
        g, subject, label, yawn = parse_video(p)
        if g == group:
            items.append((np.load(OUT_DIR / f"{g}_{p.stem}.npy"), label, yawn))
    return items

def longest_run(mar, threshold):
    #Longest stretch of consecutive frames above the threshold: no-face frames (NaN) break the run
    best = run = 0
    for above in (mar > threshold):
        run = run + 1 if above else 0
        best = max(best, run)
    return best

def score(items, threshold, min_frames):
    tp = fp = fn = talk_fp = talk_n = 0
    for mar, label, yawn in items:
        predicted = longest_run(mar, threshold) >= min_frames
        if yawn:
            tp += predicted
            fn += not predicted
        else:
            fp += predicted
        if label == "Talking":
            talk_n += 1
            talk_fp += predicted

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn)
    return precision, recall, talk_fp, talk_n

def main():
    items = load_group("female")
    print("threshold min_frames precision recall talking_false_alarms")
    for t in (0.2, 0.25, 0.3, 0.35, 0.4):
        for n in (20, 25, 30, 35, 40):
            p, r, tf, tn = score(items, t, n)
            print(f"{t:<10} {n:<11} {p:.2f}     {r:.2f}  {tf}/{tn}")


if __name__ == "__main__":
    main()