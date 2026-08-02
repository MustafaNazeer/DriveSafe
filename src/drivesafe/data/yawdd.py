#!/usr/bin/env python3
"""Build the validation manifest and summary.json for YawDD.

YawDD (Abtahi et al., ACM MMSys 2014) sits extracted on the SSD in four groups:

    Dash/Female, Dash/Male                          29 videos, one per participant
    Mirror/Female_mirror,                          320 videos, one ACTION per video
    Mirror/Male_mirror Avi Videos                       (that directory name really
                                                         does contain spaces)

This is a VALIDATION set for the yawn signal, per the dataset plan: no training
split, no partition, no seed. The manifest is a single clips.tsv inventory of all
349 files; evaluation consumes it whole, and the namespaced subject column exists
so any future use beyond validation can still prove independence from training
subjects.

Layout facts this script is written against, all verified on disk 2026-08-01:

  - Mirror filenames carry the action: Normal, Talking, Yawning, and 13 videos of
    Talking&Yawning (one with a lowercase 'yawning', normalized here). Dash
    filenames carry NO action because each dash video is a single session
    containing normal, talking, and yawning segments; the empty action column on
    dash rows is the design, not missing data.
  - The eyewear/facial-hair token is one of NoGlasses, Glasses, SunGlasses,
    GlassesBeard, Glassesmoustache.
  - Subject numbering restarts per group (Dash F 1-13, Dash M 1-16, Mirror F 1-43,
    Mirror M 1-47) and nothing in the authors' Readme links dash subject N to
    mirror subject N, so subjects are namespaced as e.g. mirror-female-01. 119
    distinct subjects.
  - Filename mess preserved verbatim in paths: three files end in .avi.avi and one
    has a space before .avi ('13-MaleNoGlasses .avi').
  - The authors' Readme claims 322 mirror videos; the distribution contains 320.
    The pristine source archive (user06.tar, containing 'YawDD dataset.rar') was
    listed on 2026-08-01 and holds exactly the same 349 paths as the extracted
    tree, so nothing was lost locally; the official distribution itself is 320.

License: non-commercial research use only, and the Abtahi et al. MMSys 2014 paper
MUST be cited in anything that uses this data. Screenshot permission is PER VIDEO
via a column in Table1.pdf / Table2.pdf beside the data; check it before putting
any YawDD frame in a report or slide. The Kaggle mirror's 'MIT' label is not a
real relicense and is ignored.

Paths are resolved by probing the known SSD mount points, so the script keeps
working after the drive is renamed or mounted under a different username.
"""

import hashlib
import json
import os
import re
import sys
from datetime import datetime

SUFFIX = "mustafa-ssd/datasets/DriveSafe/yawdd"
MOUNT_CANDIDATES = [
    f"/media/{os.environ.get('USER', '')}/{SUFFIX}",
    f"/run/media/{os.environ.get('USER', '')}/{SUFFIX}",
    f"/media/mustafa/{SUFFIX}",
    f"/media/mustafaspc/{SUFFIX}",
    f"/run/media/mustafa/{SUFFIX}",
    f"/run/media/mustafaspc/{SUFFIX}",
]

SOURCE_ARCHIVE = "user06.tar"

# Group directory -> (camera, gender). The Male mirror folder name has spaces.
GROUPS = {
    ("Dash", "Female"): ("dash", "female"),
    ("Dash", "Male"): ("dash", "male"),
    ("Mirror", "Female_mirror"): ("mirror", "female"),
    ("Mirror", "Male_mirror Avi Videos"): ("mirror", "male"),
}

ATTRIBUTES = {"NoGlasses", "Glasses", "SunGlasses", "GlassesBeard", "Glassesmoustache"}
ACTIONS = {"Normal", "Talking", "Yawning", "Talking&Yawning"}
YAWN_ACTIONS = {"Yawning", "Talking&Yawning"}

# Stem: number-GenderAttributes[-Action], tolerating the trailing-space quirk.
STEM_RE = re.compile(
    r"^([0-9]+)-(Male|Female)([A-Za-z]+)(?:-([A-Za-z&]+))?\s?$"
)

COLUMNS = ("path", "subject", "camera", "gender", "attributes", "action", "contains_yawn")


def resolve_yawdd_dir():
    for cand in MOUNT_CANDIDATES:
        if os.path.isdir(cand):
            return cand
    sys.exit("ERROR: YawDD folder not found at any known SSD mount point. Is the SSD mounted?")


def read_sha256sums(yawdd_dir):
    """Return {relative path or filename: sha256} from SHA256SUMS.txt if present."""
    sums = {}
    path = os.path.join(yawdd_dir, "SHA256SUMS.txt")
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


def parse_rows(yawdd_dir):
    """Walk the four group folders into manifest rows.

    Fails loudly rather than skipping: an unparseable filename means the layout is
    not the one this script was verified against, and dropping it would silently
    shrink the validation set.
    """
    rows = []
    for (top, sub), (camera, gender) in GROUPS.items():
        gdir = os.path.join(yawdd_dir, top, sub)
        if not os.path.isdir(gdir):
            sys.exit(f"ERROR: missing group folder {top}/{sub} in {yawdd_dir}")
        for fname in sorted(os.listdir(gdir)):
            stem = fname
            while stem.lower().endswith(".avi"):  # handles the .avi.avi doubles
                stem = stem[: -len(".avi")]
            if stem == fname:
                sys.exit(f"ERROR: non-avi file {top}/{sub}/{fname}")
            m = STEM_RE.fullmatch(stem)
            if not m:
                sys.exit(f"ERROR: cannot parse {top}/{sub}/{fname}")
            num, name_gender, attrs, action = m.groups()

            if name_gender.lower() != gender:
                sys.exit(
                    f"ERROR: filename gender {name_gender!r} disagrees with folder "
                    f"{top}/{sub} for {fname}"
                )
            if attrs not in ATTRIBUTES:
                sys.exit(f"ERROR: unknown attribute token {attrs!r} in {top}/{sub}/{fname}")

            if camera == "mirror":
                if action is None:
                    sys.exit(f"ERROR: mirror video without action label: {top}/{sub}/{fname}")
                # Normalize the one lowercase 'Talking&yawning' quirk.
                canonical = next((a for a in ACTIONS if a.lower() == action.lower()), None)
                if canonical is None:
                    sys.exit(f"ERROR: unknown action {action!r} in {top}/{sub}/{fname}")
                action_val = canonical
                yawn_val = "1" if canonical in YAWN_ACTIONS else "0"
            else:
                if action is not None:
                    sys.exit(f"ERROR: dash video with unexpected action label: {top}/{sub}/{fname}")
                action_val = ""  # dash sessions mix all actions; unlabeled by design
                yawn_val = ""

            rows.append(
                {
                    "path": f"{top}/{sub}/{fname}",
                    "subject": f"{camera}-{gender}-{int(num):02d}",
                    "camera": camera,
                    "gender": gender,
                    "attributes": attrs,
                    "action": action_val,
                    "contains_yawn": yawn_val,
                }
            )
    if not rows:
        sys.exit(f"ERROR: no avi files found under {yawdd_dir}")
    return rows


def verify(rows):
    """Assert the structural properties the summary will claim."""
    by_group = {}
    for row in rows:
        by_group.setdefault((row["camera"], row["gender"]), []).append(row)

    for (camera, gender), grows in sorted(by_group.items()):
        nums = sorted({int(r["subject"].rsplit("-", 1)[1]) for r in grows})
        # Contiguity from 1 was verified on disk; a gap appearing later means files
        # were lost or renamed, which is exactly what should fail the build.
        if nums != list(range(1, len(nums) + 1)):
            sys.exit(f"ERROR: {camera}/{gender} subject numbers not contiguous: {nums}")
        if camera == "dash":
            per_subject = {}
            for r in grows:
                per_subject[r["subject"]] = per_subject.get(r["subject"], 0) + 1
            multi = {s: n for s, n in per_subject.items() if n != 1}
            if multi:
                sys.exit(f"ERROR: dash subjects with more than one session: {multi}")
    return by_group


def write_tsv(path, rows):
    with open(path, "w") as fh:
        fh.write("\t".join(COLUMNS) + "\n")
        for row in rows:
            fh.write("\t".join(row[col] for col in COLUMNS) + "\n")


def build():
    yawdd_dir = resolve_yawdd_dir()
    splits_dir = os.path.join(yawdd_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    rows = parse_rows(yawdd_dir)
    rows.sort(key=lambda r: (r["camera"], r["gender"], int(r["subject"].rsplit("-", 1)[1]), r["path"]))

    by_group = verify(rows)

    sums = read_sha256sums(yawdd_dir)
    unhashed = [r["path"] for r in rows if r["path"] not in sums]
    if unhashed:
        print(
            f"WARNING: {len(unhashed)} of {len(rows)} files have no entry in "
            f"SHA256SUMS.txt; summary.json will carry null provenance for the set",
            file=sys.stderr,
        )

    archive_path = os.path.join(yawdd_dir, SOURCE_ARCHIVE)
    total_bytes = sum(os.path.getsize(os.path.join(yawdd_dir, r["path"])) for r in rows)

    write_tsv(os.path.join(splits_dir, "clips.tsv"), rows)

    per_group = {}
    for (camera, gender), grows in sorted(by_group.items()):
        actions = {}
        for r in grows:
            key = r["action"] or "(unlabeled session)"
            actions[key] = actions.get(key, 0) + 1
        per_group[f"{camera}-{gender}"] = {
            "files": len(grows),
            "subjects": len({r["subject"] for r in grows}),
            "per_action": dict(sorted(actions.items())),
            "yawn_positive": sum(1 for r in grows if r["contains_yawn"] == "1"),
        }

    mirror_rows = [r for r in rows if r["camera"] == "mirror"]
    sums_path = os.path.join(yawdd_dir, "SHA256SUMS.txt")
    sums_hash = sha256_of(sums_path) if (sums and not unhashed) else None

    summary = {
        "dataset": "YawDD (Yawning Detection Dataset)",
        "source": (
            "Abtahi, Omidyeganeh, Shirmohammadi, Hariri, ACM MMSys 2014; obtained via "
            "Kaggle mirror enider/yawdd-dataset. The mirror's 'MIT' label is not a valid "
            "relicense and is ignored; the authors' license governs."
        ),
        "license_note": (
            "non-commercial research use only. The MMSys 2014 paper must be cited in "
            "anything using this data. Screenshot permission is PER VIDEO via a column "
            "in Table1.pdf / Table2.pdf beside the data; check it before using any "
            "YawDD frame in a report, slide, or publication."
        ),
        "task": "yawn detection validation (no training split)",
        "class_names": {"1": "contains yawn", "0": "no yawn", "": "unlabeled dash session"},
        "label_note": (
            "mirror videos carry one action each: Normal, Talking, Yawning, or "
            "Talking&Yawning (13 videos; one lowercase 'yawning' normalized). "
            "contains_yawn = 1 for Yawning and Talking&Yawning. Dash videos are whole "
            "sessions mixing all actions, unlabeled by design, action left empty. "
            "Subject numbering restarts per camera/gender group and the Readme never "
            "links dash and mirror numbering, so subjects are namespaced "
            "(e.g. mirror-female-01); 119 distinct subjects."
        ),
        "official_count_note": (
            "the authors' Readme claims 322 mirror videos; the distribution contains "
            "320. The pristine source archive (user06.tar -> 'YawDD dataset.rar') was "
            "listed 2026-08-01 and holds exactly the same 349 paths as the extracted "
            "tree, so the gap is upstream, not local."
        ),
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "subject_independent": True,
        "split_policy": {
            "official_split": False,
            "validation_only": True,
            "unit": "subject",
            "seed": None,
            "protocol": (
                "the entire dataset is a held-out validation set for the yawn signal; "
                "nothing here is ever trained on. The namespaced subject column exists "
                "so any future training use can still prove subject independence."
            ),
        },
        "storage_note": (
            "extracted loose on the exFAT SSD (videos are large, so cluster slack is "
            "negligible); the pristine source archive user06.tar is kept alongside and "
            "its contents were verified identical to the extracted tree"
        ),
        "archives": [
            {
                "name": SOURCE_ARCHIVE,
                "bytes": os.path.getsize(archive_path) if os.path.isfile(archive_path) else None,
                "sha256": sums.get(SOURCE_ARCHIVE),
            }
        ],
        "files": {
            "count": len(rows),
            "total_bytes": total_bytes,
            "per_file_sha256": "SHA256SUMS.txt beside the data on the SSD",
            "sha256sums_sha256": sums_hash,
        },
        "splits": {
            "validation": {
                "files": len(rows),
                "subjects": len({r["subject"] for r in rows}),
                "mirror_labeled": len(mirror_rows),
                "mirror_yawn_positive": sum(1 for r in mirror_rows if r["contains_yawn"] == "1"),
                "dash_unlabeled_sessions": len(rows) - len(mirror_rows),
                "groups": per_group,
            }
        },
    }

    summary_path = os.path.join(yawdd_dir, "summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")

    # Repo copy for team visibility (without needing the SSD). This module sits at
    # src/drivesafe/data/, so the repo root is four levels up.
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    repo_copy = os.path.join(repo_root, "docs", "data", "yawdd-summary.json")
    os.makedirs(os.path.dirname(repo_copy), exist_ok=True)
    with open(repo_copy, "w") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")

    n_yawn = sum(1 for r in rows if r["contains_yawn"] == "1")
    print(f"{len(rows)} files, {len({r['subject'] for r in rows})} subjects across 4 groups")
    print(f"mirror labeled: {len(mirror_rows)} ({n_yawn} yawn-positive), dash sessions: {len(rows) - len(mirror_rows)}")
    print(f"provenance: {'complete' if sums_hash else 'INCOMPLETE'}")
    print(f"wrote {summary_path}")
    print(f"wrote {repo_copy}")
    print(f"wrote manifest {os.path.join(splits_dir, 'clips.tsv')}")


if __name__ == "__main__":
    build()
