# Receiver — RGB + Depth Video over WebRTC  (ReVo)

`receiver-3d.py` is the receiving end of the ReVo streaming pipeline.
It connects to a signaling server, negotiates a WebRTC peer connection with the
sender, reassembles chunked / FEC-protected frames over a DataChannel, decodes
them with the chosen video codec, and saves both the RGB and depth streams to
MP4 files.

For batch evaluation over many (trace, video) pairs use `run_receiver_eval.py`
instead of calling `receiver-3d.py` directly.

---

## Architecture

```
Signaling server (WebSocket)
        │  offer / answer
        ▼
RTCPeerConnection (WebRTC)
        │  DataChannel messages
        ▼
on_message()               ← asyncio / network thread
        │  writes shards into frame_content[]
        ▼
_decode_worker_thread      ← background thread
  • waits for frame assembly deadline
  • FEC-reconstructs I-frames (zfec k-of-n)
  • best-effort zero-pads incomplete P-frames
  • runs codec decompressor → numpy frame
        │  writes into display_buf[]
        ▼
_display_worker_thread     ← background thread
  • paces output to wall-clock time
  • freeze-frames on loss
  • appends to saved_frames[]
        │
        ▼
write_video_pyav()         ← called at end of session
  • writes saved_frames[]       → <out>.mp4
  • writes saved_frames_depth[] → <out>_depth.mp4
```

**Frame types**

| Symbol | Value | Meaning |
|--------|-------|---------|
| `FRAME_TYPE_RGB` | 3 | RGB color frame |
| `FRAME_TYPE_DEPTH` | 4 | Depth map frame |

**I-frames** use Reed-Solomon FEC (via `zfec`): the sender transmits *n* shards
and the receiver reconstructs from any *k* of them.
**P-frames** have no FEC; missing chunks are zero-padded for a best-effort decode.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `aiortc` | WebRTC peer connection and DataChannel |
| `aiohttp` | async WebSocket client (signaling) |
| `av` (PyAV) | MP4 muxing with libx264 |
| `torch` | GPU memory management around codec calls |
| `numpy` | frame buffer manipulation |
| `opencv-python` (`cv2`) | window cleanup |
| `zfec` | Reed-Solomon FEC decoder |
| `DCVCRT_wrapper` | DCVC-RT neural codec wrapper (local) |
| `H265_wrapper` | H.265 / HEVC codec wrapper (local) |
| `H264_wrapper` | H.264 / AVC codec wrapper (local) |

Install public packages:

```bash
pip install aiortc aiohttp av torch numpy opencv-python zfec
```

The `*_wrapper` modules are project-local — run the script from the directory
that contains them, or add that directory to `PYTHONPATH`.

---

## Running the receiver

### Single run

**Minimal:**
```bash
python receiver-3d.py \
    --server_ip 192.168.1.10 \
    --codec h265
```
Outputs: `out_video.mp4` (RGB) and `out_video_depth.mp4` (depth).

**With explicit output paths:**
```bash
python receiver-3d.py \
    --server_ip  192.168.1.10 \
    --codec      h265 \
    --out        /path/to/rgb_output.mp4 \
    --out_depth  /path/to/depth_output.mp4 \
    --stun       stun:stun.l.google.com:19302
```

### Command-line arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--server_ip` | **yes** | — | IP address of the signaling server |
| `--codec` | **yes** | — | Codec for both streams: `h265`, `h264`, `dcvcrt` |
| `--out` | no | `out_video.mp4` | Output path for the RGB video |
| `--out_depth` | no | `<out>_depth.mp4` | Output path for the depth video |
| `--stun` | no | `stun:stun.l.google.com:19302` | STUN server URL for NAT traversal |

---

## Batch evaluation (`run_receiver_eval.py`)

`run_receiver_eval.py` runs as a persistent service on the receiver machine.
It listens on a TCP control socket, waits for the sender eval script to connect
and send a `run_id`, launches `receiver-3d.py` with the correct output paths,
signals `READY` to the sender, and waits for the run to finish before accepting
the next connection.

Output is organized automatically under `OUTPUT_ROOT`:
```
<OUTPUT_ROOT>/
  <category>/
    rgb/    <video_stem>.mp4
    depth/  <video_stem>_vis.mp4
    logs/   <video_stem>.log
```

### Setup

**1. Edit the constants** at the top of `run_receiver_eval.py`:

| Constant | Description |
|----------|-------------|
| `CODEC` | Codec to use for all runs (must match sender) |
| `CONTROL_PORT` | TCP port to listen on (must match `run_sender_eval.py`) |
| `OUTPUT_ROOT` | Root directory for all output files and logs |
| `CATEGORIES` | List of network category names embedded in run IDs |

**2. Start the service** — run this before starting the sender eval:
```bash
python run_receiver_eval.py
```

The service loops indefinitely, handling one run at a time.  Stop it with
`Ctrl-C` when all evaluations are complete.

### Run ID convention

The sender constructs a `run_id` as:
```
<video_stem>_<category>_<trace_stem>
```
Example: `scene_01_wifi_trace_03`

The receiver service parses `_<category>_` from the run ID to route output
into the correct category subdirectory.  The video stem (everything before the
category tag) becomes the output filename.

### Output layout example

```
output/
  wifi/
    rgb/    scene_01.mp4
    depth/  scene_01_vis.mp4
    logs/   scene_01.log
  cell/
    rgb/    scene_01.mp4
    ...
```

---

## Signaling server

The receiver connects to:
```
ws://<server_ip>:8080/ws/demo
```
It joins as `role: answer` and waits for an SDP offer from the sender.
The sender terminates the session with `{"type": "bye"}`, after which the
receiver drains in-flight frames, saves the videos, and exits.

---

## Output files

| File | Content |
|------|---------|
| `<out>.mp4` | RGB frames, encoded losslessly with libx264 (CRF 0, preset veryslow) |
| `<out_depth>.mp4` | Depth frames, same encoding settings |

Both files use the same frame rate and GOP structure as the incoming stream.

---

## Quality masks

After each session the receiver holds two boolean lists per stream for
downstream quality analysis or neural loss concealment:

| List | Meaning |
|------|---------|
| `corrupted_frame_list_rgb/depth` | `True` if a frame received bad data (propagates through the GOP) |
| `frozen_mask_list_rgb/depth` | `True` if a frame was totally lost and the display froze on the previous frame |

---

## Troubleshooting

**Receiver connects but no frames arrive**
- Check that the sender is running and connected to the same signaling server.
- Verify the STUN server is reachable (needed for NAT traversal).

**`ModuleNotFoundError` for a wrapper**
- Run from the directory containing `H265_wrapper.py` etc., or set `PYTHONPATH`.

**All frames marked as lost**
- Network loss is too high for P-frame recovery.  Try a lower bitrate or a
  codec with stronger FEC on the sender side.

**GPU out of memory**
- Reduce resolution or frame rate on the sender, or free GPU memory before
  starting the receiver.

**Batch run: receiver exits before sender finishes**
- Increase `POST_RUN_COOLDOWN` in `run_sender_eval.py` to give the receiver
  more time to save its output before the next run begins.

**Batch run: output goes to `mixed/` instead of the right category**
- Ensure `CATEGORIES` in `run_receiver_eval.py` contains all category names
  used in your `trace_map.txt` (e.g. `wifi`, `cell`, `eth`).
