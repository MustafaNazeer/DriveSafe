#!/usr/bin/env python3
"""Build subject independent split manifests and summary.json for the MRL Eye dataset.

MRL Eye ships as a single tar on the SSD:

    mrl-eye.tar -> mrl-eye/mrlEyes_2018_01/sNNNN/<annotated filename>.png

The exFAT SSD wastes huge space on many tiny files (128KB clusters), so the archive
stays sealed: 84,898 loose PNGs would burn roughly 11GB in slack alone. This script
never decodes a pixel. It reads only the tar's member listing (tarfile.getnames) to
parse the subject and annotations out of each filename, then writes TSV manifests
plus a summary.json. Extract on an ext4 disk at train time using these manifests.

Labels live in the filename, not in a directory or a label file. The field order is
the dataset authors' own, documented in annotation.txt inside the archive:

    subjectID_imageNumber_gender_glasses_eyeState_reflections_lighting_sensorID

with eyeState 0 = closed and 1 = open. The subject also appears as the parent
directory name, which gives a free consistency check; this script asserts the two
agree rather than trusting either alone.

Unlike DGW, MRL ships no official split, so this script CREATES the partition rather
than verifying someone else's. Subjects are sorted into a canonical order, shuffled
with a fixed seed, then sliced at the subject level. Sorting before the shuffle is
required, not stylistic: Python randomizes string hashing per process, so shuffling
an unsorted set would silently produce a different partition on every run despite the
fixed seed.

Paths are resolved by probing the known SSD mount points, so the script keeps working
after the drive is renamed or mounted under a different username.
"""

import json
import os
import random
import sys
import tarfile
from datetime import datetime

SUFFIX = "mustafa-ssd/datasets/DriveSafe/mrl-eye"
MOUNT_CANDIDATES = [
    f"/media/{os.environ.get('USER', '')}/{SUFFIX}",
    f"/run/media/{os.environ.get('USER', '')}/{SUFFIX}",
    f"/media/mustafa/{SUFFIX}",
    f"/media/mustafaspc/{SUFFIX}",
    f"/run/media/mustafa/{SUFFIX}",
    f"/run/media/mustafaspc/{SUFFIX}",
]

ARCHIVE = "mrl-eye.tar"

# Split policy. MRL has no official split, so the partition is generated here.
# A validation ratio of 0 gives the two way train/test arrangement that datasets.md
# documents as the common one for the eye state classifier.
SEED = 4316
TRAIN_RATIO = 0.8

# Filename fields, in the authors' documented order (annotation.txt).
FIELDS = (
    "subject",
    "image",
    "gender",
    "glasses",
    "eye_state",
    "reflections",
    "lighting",
    "sensor",
)

# Manifest columns. The covariates ride along because they cost nothing to parse and
# they are what later lets accuracy be broken out by glasses and by lighting, which is
# where an infrared eye state classifier actually fails.
COLUMNS = (
    "path",
    "subject",
    "eye_state",
    "gender",
    "glasses",
    "reflections",
    "lighting",
    "sensor",
)

CLASS_NAMES = {"0": "closed", "1": "open"}


def resolve_mrl_dir():
    for cand in MOUNT_CANDIDATES:
        if os.path.isdir(cand):
            return cand
    sys.exit("ERROR: MRL folder not found at any known SSD mount point. Is the SSD mounted?")


def read_sha256sums(mrl_dir):
    """Return {filename: sha256} from SHA256SUMS.txt if present.

    The authors publish no checksum for this dataset, so the file is generated locally
    with `sha256sum mrl-eye.tar` run from inside the dataset directory. Running it with
    an absolute path instead would key the entry on the full path and this lookup would
    silently miss.
    """
    sums = {}
    path = os.path.join(mrl_dir, "SHA256SUMS.txt")
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


def parse_rows(tar_path):
    """Parse the archive listing into manifest rows.

    Fails loudly rather than skipping. A filename that does not match the documented
    format means the archive is not the dataset this script was written against, most
    likely a mirror that renamed things, and silently dropping those files would
    quietly shrink the dataset instead of reporting the problem.
    """
    with tarfile.open(tar_path) as tf:
        names = tf.getnames()

    rows = []
    for name in names:
        if not name.endswith(".png"):
            continue  # directory members, annotation.txt, stats_2018_01.ods
        parts = name.split("/")
        if len(parts) < 2:
            sys.exit(f"ERROR: unexpected member path {name!r}")

        stem = parts[-1][: -len(".png")]
        fields = stem.split("_")
        if len(fields) != len(FIELDS):
            sys.exit(
                f"ERROR: expected {len(FIELDS)} underscore fields in {name!r}, "
                f"got {len(fields)}"
            )

        subject = fields[0]
        parent = parts[-2]
        if subject != parent:
            sys.exit(
                f"ERROR: filename subject {subject!r} disagrees with directory "
                f"{parent!r} in {name!r}"
            )

        row = dict(zip(FIELDS, fields))
        row["path"] = name
        rows.append(row)

    if not rows:
        sys.exit(f"ERROR: no .png members found in {tar_path}")
    return rows


def partition(subjects, seed, train_ratio):
    """Split subject IDs into train and test. Returns (train_set, test_set)."""
    # Canonical order BEFORE the shuffle. Without this the seed does not reproduce:
    # Python randomizes string hashing per process, so set iteration order differs
    # between runs and the seeded shuffle would permute a different starting list.
    ordered = sorted(subjects)

    rng = random.Random(seed)  # isolated generator, not the global random state
    rng.shuffle(ordered)

    # Floor, computed in integer arithmetic. `int(n * 0.7)` would floor 10 * 0.7 to 6,
    # because 0.7 is not exactly representable in binary. This avoids that entirely.
    n_train = (len(ordered) * round(train_ratio * 100)) // 100
    return set(ordered[:n_train]), set(ordered[n_train:])


def summarize(rows):
    """Return (per_class counts, sorted subject list) for a set of rows."""
    per_class = {}
    subjects = set()
    for row in rows:
        per_class[row["eye_state"]] = per_class.get(row["eye_state"], 0) + 1
        subjects.add(row["subject"])
    per_class = {k: per_class[k] for k in sorted(per_class)}
    return per_class, sorted(subjects)


def write_tsv(path, rows):
    with open(path, "w") as fh:
        fh.write("\t".join(COLUMNS) + "\n")
        for row in rows:
            fh.write("\t".join(row[col] for col in COLUMNS) + "\n")


def build():
    mrl_dir = resolve_mrl_dir()
    archive_path = os.path.join(mrl_dir, ARCHIVE)
    if not os.path.isfile(archive_path):
        sys.exit(f"ERROR: missing archive {ARCHIVE} in {mrl_dir}")

    splits_dir = os.path.join(mrl_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    sums = read_sha256sums(mrl_dir)
    if ARCHIVE not in sums:
        print(
            f"WARNING: no SHA256 recorded for {ARCHIVE}; "
            f"summary.json will carry null provenance",
            file=sys.stderr,
        )

    rows = parse_rows(archive_path)
    _, all_subjects = summarize(rows)

    train_subs, test_subs = partition(all_subjects, SEED, TRAIN_RATIO)

    # Subject independence. This is the single correctness property of the whole
    # exercise: a model that saw a test subject during training reports an inflated,
    # indefensible number. Verify the partition we just made, do not assume it.
    shared = sorted(train_subs & test_subs)
    if shared:
        sys.exit(f"ERROR: subject leakage between train and test: {shared}")

    dropped = set(all_subjects) - (train_subs | test_subs)
    if dropped:
        sys.exit(f"ERROR: subjects lost by the partition: {sorted(dropped)}")

    train_rows = [r for r in rows if r["subject"] in train_subs]
    test_rows = [r for r in rows if r["subject"] in test_subs]
    if len(train_rows) + len(test_rows) != len(rows):
        sys.exit("ERROR: rows lost between parsing and partitioning")

    train_pc, train_list = summarize(train_rows)
    test_pc, test_list = summarize(test_rows)

    write_tsv(os.path.join(splits_dir, "train.tsv"), train_rows)
    write_tsv(os.path.join(splits_dir, "test.tsv"), test_rows)

    summary = {
        "dataset": "MRL Eye Dataset",
        "source": "Media Research Lab, VSB Technical University of Ostrava",
        "task": "eye state classification, 2 classes (open / closed)",
        "class_names": CLASS_NAMES,
        "label_note": (
            "labels live in the filename, not in a directory or a label file. Field order "
            "is the authors' own, from annotation.txt inside the archive: "
            "subjectID_imageNumber_gender_glasses_eyeState_reflections_lighting_sensorID. "
            "eyeState 0 = closed, 1 = open. The subject also appears as the parent "
            "directory name and the two are asserted to agree on every file."
        ),
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "subject_independent": True,
        "train_test_shared_subjects": len(shared),
        "split_policy": {
            "official_split": False,
            "unit": "subject",
            "seed": SEED,
            "train_ratio": TRAIN_RATIO,
            "val_ratio": 0.0,
            "test_ratio": round(1.0 - TRAIN_RATIO, 10),
            "rounding": "n_train = floor(n_subjects * train_ratio), in integer arithmetic",
            "determinism_note": (
                "subjects are sorted into a canonical order before the seeded shuffle. "
                "Without the sort, per-process string hash randomization would change the "
                "partition on every run despite the fixed seed."
            ),
        },
        "storage_note": "kept as a tar on the exFAT SSD; extract on ext4 at train time",
        "archives": [
            {
                "name": ARCHIVE,
                "bytes": os.path.getsize(archive_path),
                "sha256": sums.get(ARCHIVE),
            }
        ],
        "splits": {
            "train": {
                "images": len(train_rows),
                "subjects": len(train_list),
                "per_class": train_pc,
                "subjects_list": train_list,
            },
            "test": {
                "images": len(test_rows),
                "subjects": len(test_list),
                "per_class": test_pc,
                "subjects_list": test_list,
            },
        },
    }

    summary_path = os.path.join(mrl_dir, "summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")

    # Repo copy for team visibility (without needing the SSD). This module sits at
    # src/drivesafe/data/, so the repo root is four levels up.
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    repo_copy = os.path.join(repo_root, "docs", "data", "mrl-summary.json")
    os.makedirs(os.path.dirname(repo_copy), exist_ok=True)
    with open(repo_copy, "w") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")

    print(f"train: {len(train_rows)} imgs, {len(train_list)} subjects, per_class {train_pc}")
    print(f"test:  {len(test_rows)} imgs, {len(test_list)} subjects, per_class {test_pc}")
    print(f"train/test shared subjects: {len(shared)}")
    print(f"seed: {SEED}")
    print(f"wrote {summary_path}")
    print(f"wrote {repo_copy}")
    print(f"wrote manifests in {splits_dir}")


if __name__ == "__main__":
    build()
