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


Please Refer to [README.md](../../README.md) for installation details. 

The `*_wrapper` modules are project-local — run the script from the directory
that contains them.

---

## Running the receiver (`run_receiver_eval.py`)

`run_receiver_eval.py` runs as a persistent service on the receiver machine.
It listens on a TCP control socket, waits for the sender eval script to connect
and send a `run_id`, launches `receiver-3d.py` with the correct output paths,
signals `READY` to the sender, and waits for the run to finish before accepting
the next connection.

Output is organized automatically under `OUTPUT_ROOT`:
```
<OUTPUT_ROOT>/
  <category>/
    rgb/    <videoName>.mp4
    depth/  <videoName>_vis.mp4
    logs/   <videoName>.log
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
Example: `videoName_wifi_trace_03`

The receiver service parses `_<category>_` from the run ID to route output
into the correct category subdirectory.  The video stem (everything before the
category tag) becomes the output filename.

### Output layout example

```
output/
  wifi/
    rgb/    videoName.mp4
    depth/  videoName_vis.mp4
    logs/   videoName.log
  cell/
    rgb/    videoName.mp4
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

## Troubleshooting

See [FAQ.md](../../FAQ.md) for known issues and workarounds.
