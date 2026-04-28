"""
generate_frame_masks.py — Parse receiver logs to produce per-video frame corruption masks.

Frame mask encoding:
    0  clean frame
    1  corrupted P-frame (partial or full packet loss)
    2  corrupted I-frame (or entire GoP rendered undecodable by a missing I-frame)

The masks are saved as 1-D NumPy arrays (<video>_frame_mask.npy) and consumed
by the LossRec inference pipeline to identify which frames need reconstruction.

GoP propagation rule:
    - If the I-frame of a GoP is lost, every frame in that GoP is marked 2
      (an I-frame loss makes all subsequent P-frames in the same GoP undecodable).
    - If one or more P-frames in a GoP are lost, frames from the first lost
      P-frame to the end of that GoP are marked 1 (P-frame dependency chain).
    - Frames before the first loss in a GoP remain 0.

Usage:
    Edit LOG_DIR, SAVE_DIR, and GOP_SIZE below, then run:
        python scripts/generate_frame_masks.py
"""

import os
import re
import glob
import numpy as np


# ---------------------------------------------------------------------------
# Configuration — edit these paths before running
# ---------------------------------------------------------------------------
NETWORK  = "wifi"
CODEC	 = "h265"
LOG_DIR  = f"./output/{CODEC}/receiver_logs/{NETWORK}/logs"
SAVE_DIR = f"./output/{CODEC}/receiver_logs/{NETWORK}/frame_masks"
GOP_SIZE = 30   # frames per Group of Pictures; must match the encoder setting
# ---------------------------------------------------------------------------

# Compiled regex to strip ANSI terminal colour codes from log lines.
# Receiver threads may interleave colour-coded output on the same line,
# so we split on the reset code (\033[0m) before stripping remaining codes.
ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')


def get_frame_id_from_line(line: str):
    """Extract the frame ID embedded by the receiver using the [FID: X] tag.

    Returns the integer frame ID, or None if the tag is absent.
    The tag format is strict so unrelated numeric strings in the log are ignored.
    """
    match = re.search(r'\[FID:\s*(\d+)\]', line)
    return int(match.group(1)) if match else None


def parse_log_file(filepath: str):
    """Parse a single receiver log file and return a frame corruption mask.

    The parser scans each log line for the [FID: X] tag, classifies the
    associated error, then applies the GoP propagation rule to produce
    the final per-frame labels.

    Args:
        filepath: Absolute path to the .log file.

    Returns:
        (filename, mask) where filename is the basename of filepath and
        mask is a 1-D NumPy int array of length (max_frame_id + 1).
        Returns an empty array if no frames were found in the log.
    """

    # -----------------------------------------------------------------------
    # Error pattern lists — matched against cleaned log lines
    # -----------------------------------------------------------------------

    # Patterns that indicate an I-frame was dropped → value 2
    i_frame_errors = [
        "I-frame not fully ready. Dropping",
    ]

    # Patterns that indicate a P-frame was dropped or only partially received → value 1
    p_frame_errors = [
        "Could not build best-effort payload",
        "Built best-effort payload",
        "of RGB frame",
        "of depth frame",
        "First chunk did not arrive",
        "First chunk (depth) did not arrive",
    ]

    # Patterns that indicate a full frame is missing but type is unknown;
    # classification falls back to frame_id % GOP_SIZE to infer I vs P.
    generic_errors = [
        "Whole frame missing",
        "frame_rgb none",
        "frame_depth none",
        "content is None",
    ]

    # -----------------------------------------------------------------------
    # Pass 1: scan the log and collect corrupted frame IDs
    # -----------------------------------------------------------------------
    corrupted_i_frames = set()
    corrupted_p_frames = set()
    max_frame_found    = -1

    with open(filepath, 'r') as f:
        for raw_line in f:
            # Receiver threads often jam multiple colour-coded prints onto one
            # physical line.  Split on the ANSI reset code to separate them.
            for sub_line in raw_line.split('\033[0m'):
                if not sub_line.strip():
                    continue

                line = ANSI_ESCAPE.sub('', sub_line)
                fid  = get_frame_id_from_line(line)
                if fid is None:
                    continue

                max_frame_found = max(max_frame_found, fid)

                if any(pat in line for pat in i_frame_errors):
                    corrupted_i_frames.add(fid)

                elif any(pat in line for pat in p_frame_errors):
                    corrupted_p_frames.add(fid)

                elif any(pat in line for pat in generic_errors):
                    # Infer frame type from position within the GoP
                    if fid % GOP_SIZE == 0:
                        corrupted_i_frames.add(fid)
                    else:
                        corrupted_p_frames.add(fid)

    if max_frame_found == -1:
        return os.path.basename(filepath), np.array([])

    # -----------------------------------------------------------------------
    # Pass 2: apply GoP propagation rules
    # -----------------------------------------------------------------------
    mask = np.zeros(max_frame_found + 1, dtype=int)

    for gop_start in range(0, len(mask), GOP_SIZE):
        gop_end = min(gop_start + GOP_SIZE, len(mask))

        if gop_start in corrupted_i_frames:
            # Lost I-frame → entire GoP is undecodable
            mask[gop_start:gop_end] = 2
            continue

        # Find the first corrupted P-frame within this GoP (if any)
        first_bad = next(
            (i for i in range(gop_start + 1, gop_end) if i in corrupted_p_frames),
            None
        )
        if first_bad is not None:
            # Frames from the first corruption to the GoP boundary are tainted
            mask[first_bad:gop_end] = 1

    return os.path.basename(filepath), mask


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    log_files = sorted(glob.glob(os.path.join(LOG_DIR, "*.log")))
    print(f"Found {len(log_files)} log files in {LOG_DIR}")
    print(f"Saving masks to {SAVE_DIR}")
    print("-" * 60)

    for log_path in log_files:
        name, mask = parse_log_file(log_path)

        output_filename = name.replace(".log", "_frame_mask.npy")
        np.save(os.path.join(SAVE_DIR, output_filename), mask)

        stats = {v: int(c) for v, c in zip(*np.unique(mask, return_counts=True))}
        print(f"File: {name}")
        print(f"  → {output_filename}")
        print(f"  Total frames : {len(mask)}")
        print(f"  [0] clean    : {stats.get(0, 0)}")
        print(f"  [1] P-corrupt: {stats.get(1, 0)}")
        print(f"  [2] I-corrupt: {stats.get(2, 0)}")

        bad = np.where(mask != 0)[0]
        if len(bad) > 0:
            print(f"  First corruption at frame {bad[0]} (value={mask[bad[0]]})")
        print("-" * 60)


if __name__ == "__main__":
    main()
