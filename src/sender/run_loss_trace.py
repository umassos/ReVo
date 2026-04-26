"""
run_loss_trace.py  —  ReVo

Applies time-varying bandwidth and packet loss to a network interface using
Linux tc (traffic control) / netem, replaying a 4-column trace file.

Trace file format (whitespace-separated, one row per time step):
    <timestamp_s>  <bandwidth_mbps>  <rtt_ms>  <loss_0_to_1>

RTT is read from the trace but NOT applied — delay is fixed at FIXED_DELAY_MS.
This is intentional: in our lab setup the propagation delay is controlled
separately, so only bandwidth and loss are emulated here.

The trace loops automatically when it reaches the end.

Usage (standalone):
    sudo python run_loss_trace.py --trace path/to/trace.log --interface eth0

Normally launched as a subprocess by sender-3d.py via --trace_path.
"""

import time
import subprocess
import sys
import argparse
import os

# One-way delay applied to all packets.  RTT column in the trace is ignored.
FIXED_DELAY_MS = 40


def parse_args():
    parser = argparse.ArgumentParser(description="ReVo network trace player")
    parser.add_argument("--interface", default="enp130s0", help="Network interface (e.g. eth0)")
    parser.add_argument("--trace",     required=True,      help="Path to the trace file")
    return parser.parse_args()


def _run(cmd: str):
    """Run a shell command silently; ignore errors (tc may warn on first delete)."""
    subprocess.run(cmd, shell=True, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def setup_tc(interface: str):
    """
    Build the tc hierarchy on the interface:
      root HTB qdisc → HTB class (rate updated per step) → netem child (loss + delay)
    Any existing root qdisc is deleted first.
    """
    print(f"[tc] Setting up on {interface}")
    _run(f"tc qdisc del dev {interface} root")
    _run(f"tc qdisc  add dev {interface} root handle 1: htb default 10")
    _run(f"tc class  add dev {interface} parent 1: classid 1:10 htb rate 1000Mbit")
    _run(f"tc qdisc  add dev {interface} parent 1:10 handle 10: netem delay 0ms loss 0%")


def update_tc(interface: str, bandwidth_mbps: float, loss_ratio: float):
    """
    Apply one trace step: update bandwidth (HTB class) and loss+delay (netem).
    loss_ratio is in [0, 1]; converted to percent for netem.
    """
    loss_pct = max(0.0, loss_ratio * 100.0)
    _run(f"tc class change dev {interface} parent 1: classid 1:10 htb rate {bandwidth_mbps}Mbit burst 150k")
    _run(f"tc qdisc change dev {interface} parent 1:10 handle 10: netem delay {FIXED_DELAY_MS}ms loss {loss_pct}%")
    sys.stdout.write(f"\r[trace] BW: {bandwidth_mbps:.1f} Mbps | Loss: {loss_pct:.1f}% | t={time.perf_counter():.2f}s")
    sys.stdout.flush()


def cleanup_tc(interface: str):
    """Remove all tc rules from the interface."""
    print(f"\n[tc] Cleaning up on {interface}")
    _run(f"tc qdisc del dev {interface} root")


def main():
    args = parse_args()

    if not os.path.exists(args.trace):
        print(f"Error: trace file not found: {args.trace}")
        return

    setup_tc(args.interface)

    with open(args.trace, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if not lines:
        print("Error: trace file is empty.")
        cleanup_tc(args.interface)
        return

    # Total duration of one pass — used to offset timestamps on loop
    trace_duration = float(lines[-1].split()[0])
    num_lines      = len(lines)
    line_idx       = 0
    loop_offset    = 0.0
    start_time     = time.perf_counter()

    try:
        while True:
            current_idx = line_idx % num_lines

            # Advance the time offset each time the trace wraps around
            if line_idx > 0 and current_idx == 0:
                loop_offset += trace_duration

            parts = lines[current_idx].split()
            if len(parts) < 4:
                line_idx += 1
                continue

            # Columns: timestamp  bandwidth_mbps  rtt_ms  loss_fraction
            target_time    = float(parts[0]) + loop_offset
            bandwidth_mbps = float(parts[1])
            # parts[2] is rtt_ms — read but not applied (delay is fixed)
            loss_ratio     = max(0.0, float(parts[3]))

            # Wait until wall time matches the trace timestamp
            sleep_s = target_time - (time.perf_counter() - start_time)
            if sleep_s > 0:
                time.sleep(sleep_s)

            update_tc(args.interface, bandwidth_mbps, loss_ratio)
            line_idx += 1

    except KeyboardInterrupt:
        print("\n[tc] Interrupted.")
    except Exception as e:
        print(f"\n[tc] Error: {e}")
    finally:
        cleanup_tc(args.interface)


if __name__ == "__main__":
    main()
