#!/usr/bin/env python3
"""Build the fold-aware split manifest and summary.json for UTA-RLDD.

UTA-RLDD (Real Life Drowsiness Dataset) sits extracted on the SSD, via the Kaggle
mirror rishab260/uta-reallife-drowsiness-dataset (CC0), which carries folds 1 to 4
of the authors' five (48 of 60 subjects). The mirror nests each fold folder inside
an identically named folder:

    Fold{N}_part{M}/Fold{N}_part{M}/<subject>/<class>[_K].<ext>

with class 0 = alert, 5 = low vigilance, 10 = drowsy encoded as the filename stem.
The part folders (part1/part2) are packaging only; the fold number is what matters.

Unlike MRL, no partition is generated here. The authors' folds ARE the split: each
fold holds exactly 12 subjects, no subject appears in two folds, and the published
evaluation protocol is leave-one-fold-out cross validation. Like DGW, this script
adopts the official structure and VERIFIES its subject independence rather than
inventing its own, so there is no seed.

Layout quirks this script must handle, all verified on disk 2026-08-01:

  - Subject 32's drowsy clip ships as two files, 10_1.mp4 and 10_2.mp4. The manifest
    keeps one row per file, with a clip_part column that is empty everywhere else.
    Counting clips by requiring exact stems {0, 5, 10} silently drops subject 32,
    which is exactly the miscount (141 instead of 144 clips) this replaces.
  - Extensions are a five way mix: .mov, .MOV, .mp4, .MP4, .m4v. An extension
    whitelist built from the documented three would drop four files.

There is no archive: the Kaggle zip was consumed during extraction. Provenance is
per file instead, read from SHA256SUMS.txt beside the data (generated locally with
sha256sum over relative paths). The summary records the SHA256 of SHA256SUMS.txt
itself, so the committed repo copy pins all 145 file hashes with one string.

Paths are resolved by probing the known SSD mount points, so the script keeps
working after the drive is renamed or mounted under a different username.
"""

import hashlib
import json
import os
import re
import sys
from datetime import datetime

SUFFIX = "mustafa-ssd/datasets/DriveSafe/uta-rldd"
MOUNT_CANDIDATES = [
    f"/media/{os.environ.get('USER', '')}/{SUFFIX}",
    f"/run/media/{os.environ.get('USER', '')}/{SUFFIX}",
    f"/media/mustafa/{SUFFIX}",
    f"/media/mustafaspc/{SUFFIX}",
    f"/run/media/mustafa/{SUFFIX}",
    f"/run/media/mustafaspc/{SUFFIX}",
]

FOLDS = (1, 2, 3, 4)  # the mirror carries 4 of the authors' 5 folds
PARTS = (1, 2)
SUBJECTS_PER_FOLD = 12

CLASSES = ("0", "5", "10")
CLASS_NAMES = {"0": "alert", "5": "low_vigilance", "10": "drowsy"}

# Everything observed on disk. Lowercased for comparison; anything else fails loudly.
VIDEO_EXTS = {".mov", ".mp4", ".m4v"}

# Stem is the class, optionally with a clip-part suffix (subject 32's 10_1 / 10_2).
STEM_RE = re.compile(r"^(0|5|10)(?:_([0-9]+))?$")

COLUMNS = ("path", "subject", "fold", "class", "clip_part")


def resolve_rldd_dir():
    for cand in MOUNT_CANDIDATES:
        if os.path.isdir(cand):
            return cand
    sys.exit("ERROR: UTA-RLDD folder not found at any known SSD mount point. Is the SSD mounted?")


def read_sha256sums(rldd_dir):
    """Return {relative path: sha256} from SHA256SUMS.txt if present.

    The file is generated locally with sha256sum run from inside the dataset
    directory over relative paths, so its keys match the manifest's path column.
    """
    sums = {}
    path = os.path.join(rldd_dir, "SHA256SUMS.txt")
    if os.path.isfile(path):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(None, 1)
                if len(parts) == 2:
                    sums[parts[1]] = parts[0]
    return sums


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_rows(rldd_dir):
    """Walk the eight fold_part folders into manifest rows.

    Fails loudly rather than skipping. A file that does not parse means the layout
    is not the one this script was verified against, and silently dropping it would
    quietly shrink the dataset instead of reporting the problem.
    """
    rows = []
    for fold in FOLDS:
        for part in PARTS:
            fp = f"Fold{fold}_part{part}"
            inner = os.path.join(rldd_dir, fp, fp)  # doubled nesting, mirror quirk
            if not os.path.isdir(inner):
                sys.exit(f"ERROR: missing fold folder {fp}/{fp} in {rldd_dir}")
            for subject in sorted(os.listdir(inner)):
                sdir = os.path.join(inner, subject)
                if not os.path.isdir(sdir):
                    sys.exit(f"ERROR: unexpected non-directory {fp}/{fp}/{subject}")
                if not re.fullmatch(r"[0-9]{2}", subject):
                    sys.exit(f"ERROR: unexpected subject directory name {subject!r} in {fp}")
                for fname in sorted(os.listdir(sdir)):
                    stem, ext = os.path.splitext(fname)
                    if ext.lower() not in VIDEO_EXTS:
                        sys.exit(f"ERROR: non-video file {fp}/{fp}/{subject}/{fname}")
                    m = STEM_RE.fullmatch(stem)
                    if not m:
                        sys.exit(
                            f"ERROR: cannot parse class from {fp}/{fp}/{subject}/{fname}"
                        )
                    rows.append(
                        {
                            "path": f"{fp}/{fp}/{subject}/{fname}",
                            "subject": subject,
                            "fold": str(fold),
                            "class": m.group(1),
                            "clip_part": m.group(2) or "",
                        }
                    )
    if not rows:
        sys.exit(f"ERROR: no video files found under {rldd_dir}")
    return rows


def clips_of(rows):
    """Group file rows into logical clips: {(subject, class): [clip_part, ...]}."""
    clips = {}
    for row in rows:
        clips.setdefault((row["subject"], row["class"]), []).append(row["clip_part"])
    return clips


def verify(rows):
    """Assert every structural property the summary will claim. Returns fold map."""
    fold_subjects = {}
    for row in rows:
        fold_subjects.setdefault(row["fold"], set()).add(row["subject"])

    # The authors' folds are the split, so fold exclusivity IS the subject
    # independence property: a subject in two folds would leak between CV rounds.
    for a in sorted(fold_subjects):
        for b in sorted(fold_subjects):
            if a < b and fold_subjects[a] & fold_subjects[b]:
                sys.exit(
                    f"ERROR: subject leakage between fold {a} and fold {b}: "
                    f"{sorted(fold_subjects[a] & fold_subjects[b])}"
                )

    for fold, subs in sorted(fold_subjects.items()):
        if len(subs) != SUBJECTS_PER_FOLD:
            sys.exit(f"ERROR: fold {fold} has {len(subs)} subjects, expected {SUBJECTS_PER_FOLD}")

    # Every subject must have exactly one clip of each class; a clip is either a
    # single whole file or contiguous parts 1..K (subject 32's two-part drowsy clip).
    clips = clips_of(rows)
    all_subjects = {row["subject"] for row in rows}
    for subject in sorted(all_subjects):
        for cls in CLASSES:
            parts = clips.get((subject, cls))
            if parts is None:
                sys.exit(f"ERROR: subject {subject} has no class-{cls} clip")
            if parts == [""]:
                continue  # single whole file, the normal case
            if "" in parts:
                sys.exit(
                    f"ERROR: subject {subject} class {cls} mixes a whole clip with parts"
                )
            got = sorted(int(p) for p in parts)
            if got != list(range(1, len(got) + 1)):
                sys.exit(
                    f"ERROR: subject {subject} class {cls} has non-contiguous parts {got}"
                )
    return fold_subjects, clips


def write_tsv(path, rows):
    with open(path, "w") as fh:
        fh.write("\t".join(COLUMNS) + "\n")
        for row in rows:
            fh.write("\t".join(row[col] for col in COLUMNS) + "\n")


def build():
    rldd_dir = resolve_rldd_dir()
    splits_dir = os.path.join(rldd_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    rows = parse_rows(rldd_dir)
    rows.sort(
        key=lambda r: (
            int(r["fold"]),
            r["subject"],
            int(r["class"]),
            int(r["clip_part"] or 0),
        )
    )

    fold_subjects, clips = verify(rows)

    sums = read_sha256sums(rldd_dir)
    unhashed = [r["path"] for r in rows if r["path"] not in sums]
    if unhashed:
        print(
            f"WARNING: {len(unhashed)} of {len(rows)} files have no entry in "
            f"SHA256SUMS.txt; summary.json will carry null provenance for the set",
            file=sys.stderr,
        )
    stray = sorted(set(sums) - {r["path"] for r in rows})
    if stray:
        sys.exit(f"ERROR: SHA256SUMS.txt lists files not on disk: {stray}")

    total_bytes = sum(os.path.getsize(os.path.join(rldd_dir, r["path"])) for r in rows)

    write_tsv(os.path.join(splits_dir, "clips.tsv"), rows)

    per_fold = {}
    for fold in sorted(fold_subjects):
        fold_rows = [r for r in rows if r["fold"] == fold]
        fold_clips = clips_of(fold_rows)
        per_class = {
            cls: sum(1 for (_, c) in fold_clips if c == cls) for cls in CLASSES
        }
        per_fold[f"fold{fold}"] = {
            "subjects": len(fold_subjects[fold]),
            "files": len(fold_rows),
            "clips": len(fold_clips),
            "per_class_clips": per_class,
            "subjects_list": sorted(fold_subjects[fold]),
        }

    per_class_total = {cls: sum(1 for (_, c) in clips if c == cls) for cls in CLASSES}

    sums_path = os.path.join(rldd_dir, "SHA256SUMS.txt")
    sums_hash = sha256_of(sums_path) if (sums and not unhashed) else None

    summary = {
        "dataset": "UTA-RLDD (Real Life Drowsiness Dataset)",
        "source": (
            "UT Arlington, via Kaggle mirror rishab260/uta-reallife-drowsiness-dataset "
            "(CC0); folds 1-4 of the authors' 5, 48 of 60 subjects"
        ),
        "task": "temporal fatigue classification, 3 classes (alert / low vigilance / drowsy)",
        "class_names": CLASS_NAMES,
        "label_note": (
            "the class is the filename stem: 0 = alert, 5 = low vigilance, 10 = drowsy. "
            "Subject 32's drowsy clip ships as two files (10_1, 10_2); the manifest keeps "
            "one row per file with clip_part set on those rows, so the dataset counts 48 "
            "subjects, 144 logical clips, 145 files. Extensions are a five way mix "
            "(.mov/.MOV/.mp4/.MP4/.m4v); the fold folders are doubly nested "
            "(FoldN_partM/FoldN_partM/), both mirror packaging quirks."
        ),
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "subject_independent": True,
        "cross_fold_shared_subjects": 0,
        "split_policy": {
            "official_split": True,
            "unit": "subject",
            "seed": None,
            "protocol": (
                "leave-one-fold-out cross validation over folds 1-4, the authors' own "
                "protocol. The folds partition the 48 subjects into 4 groups of 12; "
                "no partition is generated and no seed is involved. Train on three "
                "folds, evaluate on the held-out fold, report the mean across the "
                "four rounds."
            ),
        },
        "storage_note": (
            "extracted loose on the exFAT SSD (videos are large, so cluster slack is "
            "negligible); the source Kaggle zip was consumed during extraction, so "
            "provenance is per file in SHA256SUMS.txt rather than per archive"
        ),
        "files": {
            "count": len(rows),
            "clips": len(clips),
            "total_bytes": total_bytes,
            "per_file_sha256": "SHA256SUMS.txt beside the data on the SSD",
            "sha256sums_sha256": sums_hash,
        },
        "splits": {
            "per_class_clips": per_class_total,
            "folds": per_fold,
        },
    }

    summary_path = os.path.join(rldd_dir, "summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")

    # Repo copy for team visibility (without needing the SSD). This module sits at
    # src/drivesafe/data/, so the repo root is four levels up.
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    repo_copy = os.path.join(repo_root, "docs", "data", "uta-rldd-summary.json")
    os.makedirs(os.path.dirname(repo_copy), exist_ok=True)
    with open(repo_copy, "w") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")

    n_subs = sum(len(s) for s in fold_subjects.values())
    print(f"{len(rows)} files, {len(clips)} clips, {n_subs} subjects across {len(fold_subjects)} folds")
    print(f"per-class clips: {per_class_total}")
    print(f"cross-fold shared subjects: 0 (verified)")
    print(f"provenance: {'complete' if sums_hash else 'INCOMPLETE'}")
    print(f"wrote {summary_path}")
    print(f"wrote {repo_copy}")
    print(f"wrote manifest {os.path.join(splits_dir, 'clips.tsv')}")


if __name__ == "__main__":
    build()
