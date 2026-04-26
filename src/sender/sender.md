# Sender — RGB + Depth Video over WebRTC  (ReVo)

`sender-3d.py` is the transmitting end of the ReVo streaming pipeline.
It reads pre-recorded RGB and depth video files, compresses each frame with the
chosen codec, and streams both over separate WebRTC DataChannels to a waiting
receiver (`receiver-3d.py`).

For batch evaluation over many (trace, video) pairs use `run_sender_eval.py`
instead of calling `sender-3d.py` directly.

---

## Architecture

```
VideoDecoder (torchcodec) ── RGB file
VideoDecoder (torchcodec) ── depth file
        │  raw tensors (C,H,W)
        ▼
codec.compress_stream()          (run concurrently in asyncio thread pool)
        │  compressed payload bytes
        ▼
I-frame? ──► _make_iframe_chunks()   Reed-Solomon FEC (zfec k-of-n)
P-frame? ──► slice into chunk_size   plain chunks, no FEC
        │
        ▼
DataChannel send()   rgb_payload  ──► receiver
                     depth_payload ──► receiver
        │
        │  chunks interleaved (RGB chunk → depth chunk) and paced
        │  evenly across each frame's time slot
        ▼
"bye" message → receiver saves output and exits
```

### DataChannels

| Channel | Label | Mode |
|---------|-------|------|
| RGB   | `rgb_payload`   | unreliable, unordered (SCTP maxRetransmits=0) |
| Depth | `depth_payload` | unreliable, unordered (SCTP maxRetransmits=0) |

Late packets are useless for real-time video, so SCTP-level retransmission is
disabled entirely.  Loss recovery is handled at the application layer (FEC for
I-frames, best-effort zero-padding for P-frames on the receiver side).

### FEC strategy

| Frame type | Encoding | Overhead |
|------------|----------|----------|
| I-frame | zfec k-of-n (≈ 50% parity) | `n = k + ⌈k/2⌉` |
| P-frame | no FEC; chunk 0 retransmitted once | ~1 extra packet |

I-frames are critical: losing one stalls the whole GOP.  FEC lets the receiver
reconstruct the full frame from any `k` of the `n` transmitted shards,
tolerating up to `n − k` shard losses.

P-frames carry less risk because the receiver can zero-pad missing chunks and
still attempt a decode.  Retransmitting chunk 0 protects the slice header,
which is the part the codec most needs to begin decoding.

### Send pacing

Each frame's available send window is split evenly across all chunks
(RGB + depth combined).  This prevents bursts that would overwhelm the WebRTC
congestion controller and spread any packet loss uniformly over time.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `aiortc` | WebRTC peer connection and DataChannel |
| `aiohttp` | async WebSocket client (signaling) |
| `torchcodec` | GPU/CPU video file decoder |
| `torch` | tensor operations |
| `zfec` | Reed-Solomon FEC encoder |
| `DCVCRT_wrapper` | DCVC-RT neural codec wrapper (local) |
| `H265_wrapper` | H.265 / HEVC codec wrapper (local) |
| `H264_wrapper` | H.264 / AVC codec wrapper (local) |

Install public packages:

```bash
pip install aiortc aiohttp torchcodec torch zfec
```

The `*_wrapper` modules are project-local — run the script from the directory
that contains them, or add that directory to `PYTHONPATH`.

---

## Running the sender

### Single run

**Minimal:**
```bash
python sender-3d.py \
    --file       /path/to/rgb.mp4 \
    --depth_file /path/to/depth.mp4 \
    --server_ip  192.168.1.10 \
    --codec      h265
```

**With network trace emulation:**
```bash
python sender-3d.py \
    --file       /path/to/rgb.mp4 \
    --depth_file /path/to/depth.mp4 \
    --server_ip  192.168.1.10 \
    --codec      h265 \
    --trace_path /path/to/trace.log \
    --interface  eth0 \
    --tc_script  run_loss_trace.py
```

### Command-line arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--file` | **yes** | — | Path to the RGB video file |
| `--depth_file` | **yes** | — | Path to the depth video file |
| `--server_ip` | **yes** | — | IP address of the signaling server |
| `--codec` | **yes** | — | Codec: `h265`, `h264`, `dcvcrt` |
| `--stun_url` | no | `stun:stun.l.google.com:19302` | STUN server URL |
| `--trace_path` | no | `None` | Network trace file for loss/BW emulation |
| `--interface` | no | `enp130s0` | Network interface for `tc` rules |
| `--tc_script` | no | `run_loss_trace.py` | Path to the TC control script |

---

## Batch evaluation (`run_sender_eval.py`)

`run_sender_eval.py` automates running every (video, trace) pair defined in
`trace_map.txt`.  It syncs with the receiver before each run via a TCP
control socket, launches `sender-3d.py` as a subprocess, and waits for a
cooldown period after each run so the receiver can finish saving its output.

### Setup

**1. Edit the constants** at the top of `run_sender_eval.py`:

| Constant | Description |
|----------|-------------|
| `CODEC` | Codec to use for all runs (`h265`, `h264`, `dcvcrt`) |
| `SERVER_IP` | Signaling server IP |
| `RECEIVER_IP` | Receiver machine IP (for the control-plane handshake) |
| `CONTROL_PORT` | TCP port the receiver listens on (must match receiver side) |
| `TRACE_DIRS` | List of trace category directories to evaluate |
| `RGB_VIDEO_SOURCE_DIR` | Directory containing RGB video files (`*.mp4`) |
| `DEPTH_VIDEO_SOURCE_DIR` | Directory containing depth video files |
| `DEPTH_SUFFIX` | Suffix that maps an RGB stem to its depth filename |
| `OUTPUT_ROOT` | Root directory for sender log files |
| `POST_RUN_COOLDOWN` | Seconds to wait after each run (receiver save time) |

**2. Prepare `trace_map.txt`** — one tab-separated line per run:
```
<category>  <video_stem>  <trace_path>
```
Example:
```
wifi    scene_01    /data/traces/wifi/trace_03.log
cell    scene_01    /data/traces/cell/trace_07.log
```

**3. Start the receiver eval service first** (see `receiver.md`), then run:
```bash
python run_sender_eval.py
```

### Output

Sender logs are saved to:
```
<OUTPUT_ROOT>/<CODEC>/<run_id>.log
```
where `run_id = <video_stem>_<category>_<trace_stem>`.

---

## Startup sequence (single run)

1. Sender connects to `ws://<server_ip>:8080/ws/demo` with `role: offer`.
2. Creates an SDP offer and sends it to the signaling server.
3. Waits for the receiver's SDP answer.
4. Once both DataChannels are open, a 1-second probe burst is sent on the RGB
   channel to warm up the congestion controller's bandwidth estimate.
5. After a 2-second hold, `stream_video()` begins.
6. When the last frame is sent, a `{"type": "bye"}` message is forwarded to the
   receiver, which then drains its buffers and writes the output files.

---

## Network trace (`run_loss_trace.py`)

Pass `--trace_path` to emulate real-world network conditions.  The trace script
applies time-varying bandwidth and packet loss via Linux `tc` / `netem`.

**Trace file format** (whitespace-separated):
```
<timestamp_s>  <bandwidth_mbps>  <rtt_ms>  <loss_0_to_1>
```

The RTT column is parsed but **not applied** — one-way delay is fixed at 40 ms
(`FIXED_DELAY_MS` in `run_loss_trace.py`).  The trace loops automatically.

> **Note:** `tc` rules require `sudo`.  The trace is always cleaned up on exit,
> even on crash, so the interface is left in a usable state.

---

## Session summary

At the end of each run the sender logs a breakdown:

```
[Sender Summary]
Frames sent:  300 RGB, 300 depth
Total bytes:  12.34 MB RGB, 8.21 MB depth
P-bytes RGB:  8.10 MB
I-bytes RGB (total):  4.24 MB  (data 2.83 MB + parity 1.41 MB)
...
```

- **I-bytes** — I-frame bytes (data shards + FEC parity shards)
- **P-bytes** — P-frame bytes (including the retransmitted first chunk)
- **Headers** — `DESC` / `INIT` overhead (already included in I/P totals)

---

## Troubleshooting

**Receiver never gets an offer**
- Confirm the signaling server is running at `ws://<server_ip>:8080/ws/demo`.
- Start the receiver before the sender sends the offer.

**`ModuleNotFoundError` for a wrapper**
- Run the script from the directory containing `H265_wrapper.py` etc., or set `PYTHONPATH`.

**Channel buffer exceeds hard limit (frames dropped)**
- The network is too slow for the configured bitrate.  Try a lower-quality codec
  or a smaller resolution.

**Batch run skips every video (`sync failed`)**
- The receiver eval service is not running, or `RECEIVER_IP` / `CONTROL_PORT`
  do not match between sender and receiver eval scripts.

**`sudo` prompt from `tc` cleanup on exit**
- Pre-authorize `tc` via `/etc/sudoers`, or run as root in test environments.
