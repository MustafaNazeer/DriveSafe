#!/usr/bin/env python3
"""Entry point for building subject independent split manifests.

    python -m drivesafe.data.build_splits --dataset mrl

This module holds no dataset logic. Each dataset lives in its own module because the
four layouts have very little in common: MRL encodes its label in the filename,
UTA-RLDD is video clips nested in fold folders, YawDD is a validation manifest with no
partition at all, and DMD ships OpenLABEL annotation files. A shared abstraction over
those would be mostly branches.

There is no --root or --out. Each module resolves its own paths by probing the known
SSD mount points, so the tooling keeps working after the drive is remounted under a
different username or the folder is renamed.
"""

import argparse

from drivesafe.data import dmd, mrl, rldd, yawdd

# Dataset name -> builder. Add a dataset by writing its module and one line here.
BUILDERS = {
    "dmd": dmd.build,
    "mrl": mrl.build,
    "uta-rldd": rldd.build,
    "yawdd": yawdd.build,
}


def main():
    parser = argparse.ArgumentParser(
        prog="python -m drivesafe.data.build_splits",
        description="Build subject independent split manifests and summary.json.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(BUILDERS),
        help="which dataset to build splits for",
    )
    args = parser.parse_args()
    BUILDERS[args.dataset]()


if __name__ == "__main__":
    main()
