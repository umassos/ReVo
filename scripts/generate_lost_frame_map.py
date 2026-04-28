# Generates a 1-D frame-corruption mask (.npy) from each receiver log file.
#
# Mask values:
#   0 — clean frame
#   1 — corrupted P-frame (or tail of a GoP whose P-frame was lost)
#   2 — corrupted I-frame (entire GoP is marked 2, since all P-frames depend on it)

import os
import re
import numpy as np
import glob

# ── Configuration ──────────────────────────────────────────────────────────────
network  = "eth"
LOG_DIR  = f"/path/to/receiver/output/{network}/logs"
SAVE_DIR = f"/path/to/receiver/output/{network}/frame_masks"
GOP_SIZE = 30
# ───────────────────────────────────────────────────────────────────────────────


def get_frame_id_from_line(line):
    """Return the integer frame ID embedded in a log line, or None."""
    match = re.search(r'(?:[Ff]rame|P-frame|Dropping\.|none)\s+(\d+)', line)
    return int(match.group(1)) if match else None


def parse_log_file(filepath):
    """
    Parse one receiver log file and return (filename, corruption_tensor).

    GoP-level propagation rules:
      - A bad I-frame (gop_start) poisons the whole GoP → mark all as 2.
      - The first bad P-frame within a GoP poisons that frame through GoP end → mark as 1.
    """

    # Log patterns that identify a dropped or incomplete I-frame
    I_FRAME_ERROR = "I-frame not fully ready. Dropping"

    # Log patterns that identify a dropped or incomplete P-frame
    P_FRAME_ERRORS = [
        "Could not build best-effort payload for P-frame",
        "Built best-effort payload for P-frame",
        "of RGB frame",                        # "chunk X of RGB frame Y is missing"
        "First chunk did not arrive within deadline",
    ]

    # Log patterns that can apply to either frame type; disambiguate by fid % GOP_SIZE
    GENERIC_ERRORS = [
        "Whole frame missing",
        "frame_rgb none",
        "content is None",                     # "Frame {fid} content is None. Nothing arrived within time!"
    ]

    corrupted_i_frames = set()
    corrupted_p_frames = set()
    max_frame_found = -1

    with open(filepath) as f:
        for line in f:
            fid = get_frame_id_from_line(line)
            if fid is None:
                continue

            max_frame_found = max(max_frame_found, fid)

            if I_FRAME_ERROR in line:
                corrupted_i_frames.add(fid)
            elif any(pat in line for pat in P_FRAME_ERRORS):
                corrupted_p_frames.add(fid)
            elif any(pat in line for pat in GENERIC_ERRORS):
                target = corrupted_i_frames if fid % GOP_SIZE == 0 else corrupted_p_frames
                target.add(fid)

    if max_frame_found == -1:
        return os.path.basename(filepath), np.array([])

    tensor = np.zeros(max_frame_found + 1, dtype=int)

    for gop_start in range(0, len(tensor), GOP_SIZE):
        gop_end = min(gop_start + GOP_SIZE, len(tensor))

        if gop_start in corrupted_i_frames:
            # Bad I-frame: all P-frames in this GoP are undecodable regardless
            tensor[gop_start:gop_end] = 2
            continue

        # Find the earliest P-frame corruption; everything after it in this GoP
        # is undecodable because each P-frame depends on its predecessor.
        first_bad = next(
            (i for i in range(gop_start + 1, gop_end) if i in corrupted_p_frames),
            None,
        )
        if first_bad is not None:
            tensor[first_bad:gop_end] = 1

    return os.path.basename(filepath), tensor


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    log_files = sorted(glob.glob(os.path.join(LOG_DIR, "*.log")))
    print(f"Found {len(log_files)} log files. Processing → {SAVE_DIR}")
    print("-" * 60)

    for log_path in log_files:
        name, tensor = parse_log_file(log_path)

        out_name = name.replace(".log", "_frame_mask.npy")
        np.save(os.path.join(SAVE_DIR, out_name), tensor)

        unique, counts = np.unique(tensor, return_counts=True)
        stats = dict(zip(unique, counts))

        print(f"File: {name}")
        print(f"  Saved:         {out_name}")
        print(f"  Total frames:  {len(tensor)}")
        print(f"  [0] Good:      {stats.get(0, 0)}")
        print(f"  [1] P-corrupt: {stats.get(1, 0)}")
        print(f"  [2] I-corrupt: {stats.get(2, 0)}")

        bad = np.where(tensor != 0)[0]
        if len(bad):
            print(f"  First corruption: frame {bad[0]} (value {tensor[bad[0]]})")

        print("-" * 60)


if __name__ == "__main__":
    main()
