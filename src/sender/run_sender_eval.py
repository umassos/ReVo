"""
run_sender_eval.py  —  ReVo

Batch evaluation script: iterates over every (trace, video) pair defined in
trace_map.txt, runs sender-3d.py for each combination, and logs the output.

Before each run the script syncs with the receiver over a TCP control socket
so both sides start at the same time.  After each run the network rules are
cleaned up and a cooldown period is observed to let the receiver finish saving
its output.

Trace map file format (trace_map.txt, tab-separated):
    <category>  <video_stem>  <trace_path>

Example:
    wifi    scene_01    /path/to/traces/wifi/trace_03.log
"""

import os
import glob
import subprocess
import time
import sys
import socket

# ──────────────────────────────────────────────────────────────────────────────
# Configuration — edit these before running
# ──────────────────────────────────────────────────────────────────────────────

NETWORK_INTERFACE  = "enp130s0"
SERVER_IP          = "10.0.0.5"   # signaling server IP
RECEIVER_IP        = "10.0.0.6"   # used for the control-plane sync socket
CONTROL_PORT       = 6000         # TCP port the receiver listens on for sync
CODEC              = "h264"       # "h265", "h264", or "dcvcrt"
DEPTH_SUFFIX       = "_vis"       # depth filename = <rgb_stem><DEPTH_SUFFIX>.mp4
POST_RUN_COOLDOWN  = 180          # seconds to wait after each run (receiver save time)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Trace categories to evaluate (uncomment to enable)
TRACE_DIRS = [
    os.path.join(BASE_DIR, "./traces/wifi"),
    # os.path.join(BASE_DIR, "./traces/cell"),
    # os.path.join(BASE_DIR, "./traces/eth"),
]

RGB_VIDEO_SOURCE_DIR   = os.path.join(BASE_DIR, "../data/gt_rgb_looped")
DEPTH_VIDEO_SOURCE_DIR = os.path.join(BASE_DIR, "../data/gt_depth_looped")

OUTPUT_ROOT  = os.path.join(BASE_DIR, f"../output/{CODEC}/sender_logs")
SENDER_SCRIPT = os.path.join(BASE_DIR, "sender-3d.py")
TC_SCRIPT     = os.path.join(BASE_DIR, "run_loss_trace.py")
TRACE_MAP_TXT = os.path.join(BASE_DIR, "./trace_map.txt")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def cleanup_network():
    """Remove any leftover tc rules from a previous run."""
    subprocess.run(f"tc qdisc del dev {NETWORK_INTERFACE} root",
                   shell=True, stderr=subprocess.DEVNULL)


def sync_with_receiver(run_id: str) -> bool:
    """
    Connect to the receiver's control socket and exchange a handshake.
    The sender sends the run_id string; the receiver replies b"READY" when it
    is ready to accept a new stream.

    Returns True if the handshake succeeds, False otherwise.
    """
    print(f" -> Syncing with receiver for run '{run_id}'...", end="", flush=True)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5)
        s.connect((RECEIVER_IP, CONTROL_PORT))
        s.sendall(run_id.encode())
        resp = s.recv(1024)
        s.close()
        if resp == b"READY":
            print(" [OK]")
            return True
        print(f" [FAIL] unexpected response: {resp}")
        return False
    except Exception as e:
        print(f" [ERROR] {e}")
        return False


def load_trace_map(path: str) -> dict:
    """
    Parse trace_map.txt into a dict:  (category, video_stem) -> trace_path
    Lines beginning with '#' and blank lines are ignored.
    """
    if not os.path.exists(path):
        raise RuntimeError(f"Trace map not found: {path}")
    mapping = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            category, stem, trace_path = line.split("\t")
            mapping[(category, stem)] = trace_path
    return mapping


# ──────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

def run_evaluation():
    output_dir = os.path.join(OUTPUT_ROOT, CODEC)
    os.makedirs(output_dir, exist_ok=True)

    cleanup_network()
    trace_map  = load_trace_map(TRACE_MAP_TXT)
    rgb_videos = sorted(glob.glob(os.path.join(RGB_VIDEO_SOURCE_DIR, "*.mp4")))

    if not rgb_videos:
        print("Error: no RGB videos found.")
        return

    for trace_dir_path in TRACE_DIRS:
        if not os.path.exists(trace_dir_path):
            print(f"Warning: trace directory not found: {trace_dir_path}")
            continue

        category_traces = sorted(glob.glob(os.path.join(trace_dir_path, "*.log")))
        if not category_traces:
            print(f"Warning: no trace files in {trace_dir_path}")
            continue

        category_name = os.path.basename(trace_dir_path)  # e.g. "wifi", "cell"
        print(f"\n{'='*50}")
        print(f" CATEGORY: {category_name.upper()} — {len(category_traces)} traces, {len(rgb_videos)} videos")
        print(f"{'='*50}\n")

        for rgb_video_path in rgb_videos:
            rgb_stem       = os.path.splitext(os.path.basename(rgb_video_path))[0]
            depth_video_path = os.path.join(DEPTH_VIDEO_SOURCE_DIR, f"{rgb_stem}{DEPTH_SUFFIX}.mp4")

            if not os.path.exists(depth_video_path):
                print(f"[SKIP] Depth file missing for {rgb_stem}")
                continue

            key = (category_name, rgb_stem)
            if key not in trace_map:
                print(f"[SKIP] No trace mapping for {key}")
                continue

            trace_path = trace_map[key]
            if not os.path.exists(trace_path):
                print(f"[SKIP] Trace file not found: {trace_path}")
                continue

            trace_stem = os.path.splitext(os.path.basename(trace_path))[0]
            # run_id is shared with the receiver so it can name its output files
            run_id       = f"{rgb_stem}_{category_name}_{trace_stem}"
            log_file_path = os.path.join(output_dir, f"{run_id}.log")

            print(f"{'─'*50}")
            print(f"RUN: {run_id}")

            if not sync_with_receiver(run_id):
                print("[SKIP] Sync failed.")
                continue

            print(f" -> trace: {trace_stem} ({category_name})")
            with open(log_file_path, "wb") as log_file:
                subprocess.run([
                    sys.executable, "-u", SENDER_SCRIPT,
                    "--file",       rgb_video_path,
                    "--depth_file", depth_video_path,
                    "--server_ip",  SERVER_IP,
                    "--codec",      CODEC,
                    "--trace_path", trace_path,
                    "--interface",  NETWORK_INTERFACE,
                    "--tc_script",  TC_SCRIPT,
                ], stdout=log_file, stderr=log_file)

            cleanup_network()
            print(f" -> network cleaned. Cooling down for {POST_RUN_COOLDOWN}s...")
            time.sleep(POST_RUN_COOLDOWN)


if __name__ == "__main__":
    try:
        run_evaluation()
    except KeyboardInterrupt:
        cleanup_network()
