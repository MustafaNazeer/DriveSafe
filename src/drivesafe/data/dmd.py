#!/usr/bin/env python3
"""Build subject independent splits, manifests, and summary.json for DMD.

DMD (Driver Monitoring Dataset, Vicomtech) sits on the SSD as three bundles of
per-subject tar.gz archives covering the 14 unrestricted subjects
(1,5,6,7,9,10,13,14,23,28,29,33,36,37; the drowsiness bundle ships 13 archives
because subject 28 has no drowsiness recording):

    distraction/dmd-dataset-distraction-g<G>-<S>.tar.gz   14 archives, 77 GB
    gaze-hands/dmd-dataset-gaze-g<G>-<S>.tar.gz           14 archives, 19 GB
    drowsiness/dmd-dataset-drowsiness-g<G>-<S>.tar.gz     13 archives, 13 GB

Inside every archive the layout is:

    dmd/g<G>/<subject>/s<N>/g<G>_<subject>_s<N>_<timestamp>_rgb_<rest>

where <rest> is face.mp4 / body.mp4 / hands.mp4 / mosaic.avi or an OpenLABEL
annotation file (ann_distraction.json, ann_gaze.json, ann_hands.json,
ann_drowsiness.json). Timestamps contain semicolons ("2019-04-09T10;35;56+02;00")
and never underscores, so a maxsplit filename parse is exact.

gzip has no central directory, so unlike DGW's zips the member lists cannot be
read cheaply: every listing costs a full decompress of the archive. This module
therefore makes ONE streaming pass per archive (cached, resumable) that records
the member inventory to inventory/<archive>.tsv and extracts only the small
annotation JSONs to annotations/, both beside the data on the SSD. Everything
downstream builds from that cache without touching the archives again; deleting
the cache just re-runs the pass.

The annotations are ASAM OpenLABEL: openlabel.actions is a dict of
{type: "level/label", frame_intervals: [{frame_start, frame_end}]}. The parse is
generic over levels, so gaze_zone/*, eyes_state/*, blinks/*, yawning/*, and the
distraction levels all land in one intervals.tsv without hardcoding any label
vocabulary.

DMD has no official split. Like MRL, the partition is created here: the 14
subject IDs are sorted numerically, shuffled with a fixed seed, and sliced
0.8/0.2 (floor, integer arithmetic) into 11 train / 3 test. ONE partition is
shared by all three bundles so no subject is ever train in one signal and test
in another; subject 28 simply contributes no drowsiness rows. Subjects are
sorted into canonical order before the seeded shuffle for the same reason as
MRL: without the sort, per-process hash randomization would change the
partition on every run despite the fixed seed.

License: CC BY-NC-ND 4.0 (Vicomtech). Attribution required, non-commercial use
only, no distribution of transformed material. Only RGB is public: IR, depth,
and simulator recordings were withdrawn upstream, so no DriveSafe feature may
assume DMD IR exists.

Paths are resolved by probing the known SSD mount points, so the script keeps
working after the drive is renamed or mounted under a different username.
"""

import json
import os
import random
import re
import sys
import tarfile
from datetime import datetime

SUFFIX = "mustafa-ssd/datasets/DriveSafe/dmd"
MOUNT_CANDIDATES = [
    f"/media/{os.environ.get('USER', '')}/{SUFFIX}",
    f"/run/media/{os.environ.get('USER', '')}/{SUFFIX}",
    f"/media/mustafa/{SUFFIX}",
    f"/media/mustafaspc/{SUFFIX}",
    f"/run/media/mustafa/{SUFFIX}",
    f"/run/media/mustafaspc/{SUFFIX}",
]

# bundle name -> (folder on the SSD, archive filename prefix)
BUNDLES = {
    "distraction": ("distraction", "dmd-dataset-distraction"),
    "gaze": ("gaze-hands", "dmd-dataset-gaze"),
    "drowsiness": ("drowsiness", "dmd-dataset-drowsiness"),
}

ARCHIVE_RE = re.compile(r"^dmd-dataset-(?:distraction|gaze|drowsiness)-(g[A-Z])-([0-9]+)\.tar\.gz$")

VIDEO_STREAMS = {"face.mp4", "body.mp4", "hands.mp4", "mosaic.avi"}
ANN_RE = re.compile(r"^ann_([a-z]+)\.json$")

SEED = 4316
TRAIN_RATIO = 0.8

CLIP_COLUMNS = ("path", "bundle", "group", "subject", "session", "timestamp", "stream", "split")
INTERVAL_COLUMNS = (
    "bundle", "group", "subject", "session", "timestamp", "annotation",
    "level", "label", "frame_start", "frame_end", "split",
)


def resolve_dmd_dir():
    for cand in MOUNT_CANDIDATES:
        if os.path.isdir(cand):
            return cand
    sys.exit("ERROR: DMD folder not found at any known SSD mount point. Is the SSD mounted?")


def read_sha256sums(path):
    """Return {filename: sha256} from a SHA256SUMS.txt if present."""
    sums = {}
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


def parse_member(name):
    """Parse an archive member path into its fields, or None for directories.

    Returns (group, subject, session, timestamp, rest) and fails loudly on any
    file that does not match the verified layout.
    """
    if name.endswith("/"):
        return None
    parts = name.split("/")
    if len(parts) != 5 or parts[0] != "dmd":
        sys.exit(f"ERROR: unexpected member path {name!r}")
    _, group, subject, session, fname = parts
    fields = fname.split("_", 5)
    if len(fields) != 6:
        sys.exit(f"ERROR: cannot parse member filename {fname!r}")
    f_group, f_subject, f_session, timestamp, rgb, rest = fields
    if (f_group, f_subject, f_session) != (group, subject, session) or rgb != "rgb":
        sys.exit(f"ERROR: filename fields disagree with path in {name!r}")
    if rest not in VIDEO_STREAMS and not ANN_RE.fullmatch(rest):
        sys.exit(f"ERROR: unknown stream/annotation {rest!r} in {name!r}")
    return group, subject, session, timestamp, rest


def discover_archives(dmd_dir):
    """Return {bundle: [(archive filename, group, subject), ...]}, sorted."""
    out = {}
    for bundle, (folder, prefix) in BUNDLES.items():
        bdir = os.path.join(dmd_dir, folder)
        if not os.path.isdir(bdir):
            sys.exit(f"ERROR: missing bundle folder {folder} in {dmd_dir}")
        found = []
        for fname in sorted(os.listdir(bdir)):
            m = ARCHIVE_RE.fullmatch(fname)
            if m and fname.startswith(prefix + "-"):
                found.append((fname, m.group(1), m.group(2)))
        if not found:
            sys.exit(f"ERROR: no {prefix} archives in {bdir}")
        out[bundle] = found
    return out


def scan_archive(archive_path, inv_path, ann_dir):
    """One streaming pass: write the member inventory and extract annotation JSONs.

    Resumable at archive granularity: the inventory file is written to a .partial
    name and renamed only after the whole archive has streamed, so a killed run
    leaves no half-recorded archive behind.
    """
    partial = inv_path + ".partial"
    with tarfile.open(archive_path, "r|gz") as tf, open(partial, "w") as inv:
        inv.write("member\tbytes\n")
        for member in tf:
            if member.isdir():
                continue
            inv.write(f"{member.name}\t{member.size}\n")
            base = member.name.rsplit("/", 1)[-1]
            if ANN_RE.fullmatch(base.split("_", 5)[-1] if len(base.split("_", 5)) == 6 else ""):
                # annotations/ mirrors the member path minus the leading "dmd/"
                dest = os.path.join(ann_dir, *member.name.split("/")[1:])
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                fh = tf.extractfile(member)
                if fh is None:
                    sys.exit(f"ERROR: cannot extract {member.name} from {archive_path}")
                with open(dest, "wb") as out:
                    out.write(fh.read())
    os.replace(partial, inv_path)


def ensure_cache(dmd_dir, archives):
    """Run the streaming pass for every archive whose inventory is missing."""
    inv_dir = os.path.join(dmd_dir, "inventory")
    ann_dir = os.path.join(dmd_dir, "annotations")
    os.makedirs(inv_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)
    todo = []
    for bundle, items in archives.items():
        folder = BUNDLES[bundle][0]
        for fname, _, _ in items:
            inv_path = os.path.join(inv_dir, fname + ".tsv")
            if not os.path.isfile(inv_path):
                todo.append((os.path.join(dmd_dir, folder, fname), inv_path))
    for i, (archive_path, inv_path) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] scanning {os.path.basename(archive_path)} ...", flush=True)
        scan_archive(archive_path, inv_path, ann_dir)
    return inv_dir, ann_dir


def partition(subjects, seed, train_ratio):
    """Split subject IDs into train and test sets, MRL-style."""
    ordered = sorted(subjects, key=int)  # canonical order BEFORE the shuffle
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n_train = (len(ordered) * round(train_ratio * 100)) // 100
    return set(ordered[:n_train]), set(ordered[n_train:])


def load_intervals(ann_path):
    """Parse one OpenLABEL file into (annotation kind, [(level, label, start, end)])."""
    with open(ann_path) as fh:
        ol = json.load(fh)["openlabel"]
    rest = os.path.basename(ann_path).split("_", 5)[-1]
    m = ANN_RE.fullmatch(rest)
    if not m:
        sys.exit(f"ERROR: cannot derive annotation kind from {ann_path!r}")
    kind = m.group(1)
    rows = []
    for action in ol.get("actions", {}).values():
        atype = action["type"]
        level, _, label = atype.partition("/")
        if not label:
            sys.exit(f"ERROR: action type {atype!r} in {ann_path} has no level/label form")
        for iv in action.get("frame_intervals", []):
            rows.append((level, label, int(iv["frame_start"]), int(iv["frame_end"])))
    return kind, rows


def write_tsv(path, columns, rows):
    with open(path, "w") as fh:
        fh.write("\t".join(columns) + "\n")
        for row in rows:
            fh.write("\t".join(str(row[c]) for c in columns) + "\n")


def build():
    dmd_dir = resolve_dmd_dir()
    archives = discover_archives(dmd_dir)
    inv_dir, ann_dir = ensure_cache(dmd_dir, archives)

    splits_dir = os.path.join(dmd_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    # One shared partition across all bundles.
    all_subjects = {subj for items in archives.values() for _, _, subj in items}
    train_subs, test_subs = partition(all_subjects, SEED, TRAIN_RATIO)
    shared = sorted(train_subs & test_subs)
    if shared:
        sys.exit(f"ERROR: subject leakage between train and test: {shared}")
    if (train_subs | test_subs) != all_subjects:
        sys.exit("ERROR: subjects lost by the partition")
    split_of = {s: "train" for s in train_subs}
    split_of.update({s: "test" for s in test_subs})

    clip_rows = []
    interval_rows = []
    per_bundle = {}
    for bundle, items in sorted(archives.items()):
        folder = BUNDLES[bundle][0]
        sums = read_sha256sums(os.path.join(dmd_dir, folder, "SHA256SUMS.txt"))
        bundle_archives = []
        n_videos = n_anns = 0
        level_counts = {}
        for fname, a_group, a_subject in items:
            apath = os.path.join(dmd_dir, folder, fname)
            bundle_archives.append(
                {"name": fname, "bytes": os.path.getsize(apath), "sha256": sums.get(fname)}
            )
            with open(os.path.join(inv_dir, fname + ".tsv")) as fh:
                header = fh.readline()
                if header != "member\tbytes\n":
                    sys.exit(f"ERROR: bad inventory header for {fname}")
                for line in fh:
                    member = line.rstrip("\n").split("\t")[0]
                    parsed = parse_member(member)
                    if parsed is None:
                        continue
                    group, subject, session, timestamp, rest = parsed
                    if group != a_group or subject != a_subject:
                        sys.exit(
                            f"ERROR: member {member!r} disagrees with archive {fname}"
                        )
                    if rest in VIDEO_STREAMS:
                        n_videos += 1
                        clip_rows.append(
                            {
                                "path": member,
                                "bundle": bundle,
                                "group": group,
                                "subject": subject,
                                "session": session,
                                "timestamp": timestamp,
                                "stream": rest.split(".")[0],
                                "split": split_of[subject],
                            }
                        )
                    else:
                        n_anns += 1
                        dest = os.path.join(ann_dir, *member.split("/")[1:])
                        if not os.path.isfile(dest):
                            sys.exit(
                                f"ERROR: annotation {member} listed in inventory but "
                                f"missing from cache; delete {inv_dir} and re-run"
                            )
                        kind, ivs = load_intervals(dest)
                        for level, label, start, end in ivs:
                            level_counts[f"{level}/{label}"] = (
                                level_counts.get(f"{level}/{label}", 0) + 1
                            )
                            interval_rows.append(
                                {
                                    "bundle": bundle,
                                    "group": group,
                                    "subject": subject,
                                    "session": session,
                                    "timestamp": timestamp,
                                    "annotation": kind,
                                    "level": level,
                                    "label": label,
                                    "frame_start": start,
                                    "frame_end": end,
                                    "split": split_of[subject],
                                }
                            )
        missing_hash = [a["name"] for a in bundle_archives if a["sha256"] is None]
        if missing_hash:
            print(
                f"WARNING: {bundle}: no SHA256 recorded for {missing_hash}",
                file=sys.stderr,
            )
        per_bundle[bundle] = {
            "archives": bundle_archives,
            "subjects": sorted({s for _, _, s in items}, key=int),
            "videos": n_videos,
            "annotation_files": n_anns,
            "intervals_per_label": dict(sorted(level_counts.items())),
        }

    clip_rows.sort(key=lambda r: (r["bundle"], int(r["subject"]), r["session"], r["timestamp"], r["stream"]))
    interval_rows.sort(
        key=lambda r: (
            r["bundle"], int(r["subject"]), r["session"], r["timestamp"],
            r["annotation"], r["level"], r["frame_start"], r["frame_end"], r["label"],
        )
    )

    write_tsv(os.path.join(splits_dir, "clips.tsv"), CLIP_COLUMNS, clip_rows)
    write_tsv(os.path.join(splits_dir, "intervals.tsv"), INTERVAL_COLUMNS, interval_rows)

    summary = {
        "dataset": "DMD (Driver Monitoring Dataset)",
        "source": (
            "Vicomtech, https://dmd.vicomtech.org/; distraction, gaze-hands, and "
            "drowsiness bundles for the 14 unrestricted subjects "
            "(1,5,6,7,9,10,13,14,23,28,29,33,36,37); RGB only"
        ),
        "license_note": (
            "CC BY-NC-ND 4.0. Attribution to Vicomtech required, non-commercial use "
            "only, no distribution of transformed material. Cite the DMD papers per "
            "https://dmd.vicomtech.org/. IR, depth, and simulator recordings were "
            "withdrawn upstream; nothing may assume DMD IR exists."
        ),
        "task": (
            "distraction actions, gaze zones, hands-on-wheel, and drowsiness "
            "(eye state, blinks, yawns) from frame-interval annotations"
        ),
        "label_note": (
            "annotations are ASAM OpenLABEL JSON per recording; every action type is "
            "level/label (e.g. eyes_state/close, gaze_zone/left_mirror, "
            "driver_actions/texting_right) with frame intervals. intervals.tsv holds "
            "one row per interval, generically parsed with no hardcoded vocabulary; "
            "clips.tsv holds one row per video file (face/body/hands/mosaic streams). "
            "The drowsiness bundle ships 13 archives: subject 28 has no drowsiness "
            "recording upstream."
        ),
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "subject_independent": True,
        "train_test_shared_subjects": 0,
        "split_policy": {
            "official_split": False,
            "unit": "subject",
            "seed": SEED,
            "train_ratio": TRAIN_RATIO,
            "val_ratio": 0.0,
            "test_ratio": round(1.0 - TRAIN_RATIO, 10),
            "rounding": "n_train = floor(n_subjects * train_ratio), in integer arithmetic",
            "note": (
                "one partition of the 14 unrestricted subjects is shared by all three "
                "bundles, so no subject is train in one signal and test in another. "
                "Subjects are sorted numerically before the seeded shuffle; without "
                "the sort, hash randomization would change the partition every run."
            ),
        },
        "storage_note": (
            "archives stay gzipped on the exFAT SSD (gzip has no central directory, "
            "so member listings cost a full decompress; the one-time streaming pass "
            "caches inventories and annotation JSONs under inventory/ and "
            "annotations/ beside the archives, and rebuilds never touch the archives "
            "again)"
        ),
        "subjects": {
            "all": sorted(all_subjects, key=int),
            "train": sorted(train_subs, key=int),
            "test": sorted(test_subs, key=int),
        },
        "splits": {
            "clips": {
                "total": len(clip_rows),
                "train": sum(1 for r in clip_rows if r["split"] == "train"),
                "test": sum(1 for r in clip_rows if r["split"] == "test"),
            },
            "intervals": {
                "total": len(interval_rows),
                "train": sum(1 for r in interval_rows if r["split"] == "train"),
                "test": sum(1 for r in interval_rows if r["split"] == "test"),
            },
        },
        "bundles": per_bundle,
    }

    summary_path = os.path.join(dmd_dir, "summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")

    # Repo copy for team visibility (without needing the SSD). This module sits at
    # src/drivesafe/data/, so the repo root is four levels up.
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    repo_copy = os.path.join(repo_root, "docs", "data", "dmd-summary.json")
    os.makedirs(os.path.dirname(repo_copy), exist_ok=True)
    with open(repo_copy, "w") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")

    print(f"subjects: {len(all_subjects)} -> train {sorted(train_subs, key=int)} / test {sorted(test_subs, key=int)}")
    print(f"clips: {len(clip_rows)} video files, intervals: {len(interval_rows)}")
    for bundle, info in sorted(per_bundle.items()):
        print(f"  {bundle}: {len(info['archives'])} archives, {info['videos']} videos, {info['annotation_files']} annotation files")
    print(f"seed: {SEED}")
    print(f"wrote {summary_path}")
    print(f"wrote {repo_copy}")
    print(f"wrote manifests in {splits_dir}")


if __name__ == "__main__":
    build()
