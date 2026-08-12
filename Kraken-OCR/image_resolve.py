"""Shared image-path resolution for train.py and inference.py.

The naive version of this (``os.path.join(base_dir, f"{record_id}.jpg")``)
silently fails whenever:
  1. pandas has coerced the ID column to float (NaNs anywhere in the
     column upcast the whole column, turning "1054" into "1054.0"),
  2. the images archive had its own top-level folder, so unzipping it
     into `images/` produced a nested `images/images/*.jpg`, or
  3. the actual extension isn't `.jpg`.

build_image_path() tries the fast exact-match path first (no behavior
change, no perf hit, for correctly-named data), then falls back
through the three cases above. It prints a one-time diagnostic per
base_dir the first time nothing matches, so the actual cause shows up
in the logs instead of a silent 100%-missing count.
"""

import glob
import os

_warned_dirs = set()


def _glob_any_ext(directory, stem):
    if not os.path.isdir(directory):
        return None
    matches = glob.glob(os.path.join(directory, f"{stem}.*"))
    if not matches:
        # case-insensitive fallback (e.g. .JPG, .Png)
        matches = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if os.path.splitext(f)[0].lower() == stem.lower()
        ]
    return matches[0] if matches else None


def build_image_path(base_dir, record_id):
    record_id = str(record_id)

    exact = os.path.join(base_dir, f"{record_id}.jpg")
    if os.path.exists(exact):
        return exact

    stems = [record_id]
    if record_id.endswith(".0"):
        stems.append(record_id[:-2])  # undo pandas float coercion

    search_dirs = [base_dir]
    nested = os.path.join(base_dir, os.path.basename(os.path.normpath(base_dir)))
    if os.path.isdir(nested):
        search_dirs.append(nested)  # unzip produced base_dir/base_dir/*

    for directory in search_dirs:
        for stem in stems:
            match = _glob_any_ext(directory, stem)
            if match:
                return match

    if base_dir not in _warned_dirs:
        _warned_dirs.add(base_dir)
        sample = os.listdir(base_dir)[:5] if os.path.isdir(base_dir) else []
        print(
            f"[WARN] no image match under {base_dir!r} for id={record_id!r} "
            f"(sample dir contents: {sample}) - "
            f"check ID dtype in your CSV and the unzip layout"
        )

    # preserve prior behavior: caller checks os.path.exists() and skips
    return exact
