# ReVo — Real-Time Volumetric Video Streaming over WebRTC

ReVo streams synchronized **RGB + depth** video between two machines in
real time using WebRTC DataChannels.  It supports traditional codecs (H.264,
H.265) and the neural DCVC-RT codec, with forward error correction on I-frames
and deadline-based frame assembly on the receiver.

---

## Repository layout

```
src/
├── signalling_server.py      WebRTC signaling server (run on a third machine or either peer)
├── sender/
│   ├── sender-3d.py          Sender — reads video files, encodes, streams over WebRTC
│   ├── run_sender_eval.py    Batch evaluation script (iterates over traces × videos)
│   ├── run_loss_trace.py     Network emulator — applies tc/netem from a trace file
│   ├── H264_wrapper.py       H.264 codec interface
│   ├── H265_wrapper.py       H.265 / HEVC codec interface
│   ├── DCVCRT_wrapper.py     DCVC-RT neural codec interface
│   ├── trace_map.txt         Maps (category, video) → trace file for batch eval
│   ├── traces/               Network trace files organized by category (wifi/, cell/, …)
│   └── sender.md             Detailed sender documentation
└── receiver/
    ├── receiver-3d.py        Receiver — reassembles frames, decodes, saves MP4
    ├── run_receiver_eval.py  Batch evaluation service (pairs with run_sender_eval.py)
    ├── H264_wrapper.py       H.264 codec interface
    ├── H265_wrapper.py       H.265 / HEVC codec interface
    ├── DCVCRT_wrapper.py     DCVC-RT neural codec interface
    └── receiver.md           Detailed receiver documentation
```

---

## How it works

```
  Sender machine                Signaling server            Receiver machine
  ─────────────                 ────────────────            ────────────────
  sender-3d.py  ──── SDP ────►  signalling_server.py  ◄──── receiver-3d.py
                ◄─── SDP ────                          ────►
                                                              │
  [WebRTC DataChannels established directly peer-to-peer]     │
  sender-3d.py ────── rgb_payload / depth_payload ──────────► receiver-3d.py
                                                              │
                                                         write_video_pyav()
                                                         saves rgb.mp4 + depth.mp4
```

1. **Signaling server** relays SDP offer/answer between sender and receiver.
2. **Sender** reads RGB and depth MP4 files, compresses each frame with the
   chosen codec, and transmits both streams over two unreliable WebRTC
   DataChannels.
3. **Receiver** assembles incoming chunks into frames, decodes them, and writes
   the output to two MP4 files.

### Key design choices

| Design | Detail |
|--------|--------|
| Transport | WebRTC DataChannels — unreliable, unordered (no SCTP retransmit) |
| I-frame protection | Reed-Solomon FEC (zfec k-of-n, ≈50% parity overhead) |
| P-frame recovery | Best-effort: missing chunks are zero-padded |
| Chunk interleaving | RGB and depth chunks alternate within each frame to equalize loss impact |
| Send pacing | Chunks spread evenly across the frame's time slot |
| Deadline clock | Receiver starts a wall-clock deadline on the first decoded I-frame |
| Freeze strategy | Lost frames repeat the last good frame |

---

## Quick start

### 1. Install dependencies

```bash
pip install aiortc aiohttp av torchcodec torch numpy opencv-python zfec
```

Also ensure the codec wrapper modules (`H265_wrapper.py`, `H264_wrapper.py`,
`DCVCRT_wrapper.py`) are present in both `src/sender/` and `src/receiver/`.

### 2. Start the signaling server

Run this on any machine reachable by both sender and receiver:

```bash
python src/signalling_server.py
```

Listens on `0.0.0.0:8080`.

### 3. Start the receiver

```bash
python src/receiver/receiver-3d.py \
    --server_ip <signaling_server_ip> \
    --codec h265 \
    --out rgb_out.mp4 \
    --out_depth depth_out.mp4
```

### 4. Start the sender

```bash
python src/sender/sender-3d.py \
    --file       /path/to/rgb.mp4 \
    --depth_file /path/to/depth.mp4 \
    --server_ip  <signaling_server_ip> \
    --codec      h265
```

The sender streams all frames, sends a `bye` message, and exits.  The receiver
saves its output files and exits automatically.

---

## Batch evaluation

For evaluating across many (trace, video) combinations:

1. Edit `src/sender/run_sender_eval.py` and `src/receiver/run_receiver_eval.py`
   with your machine IPs, codec, and directory paths.
2. Populate `src/sender/trace_map.txt` with `(category, video_stem, trace_path)` entries.
3. On the receiver machine: `python src/receiver/run_receiver_eval.py`
4. On the sender machine: `python src/sender/run_sender_eval.py`

See [sender.md](src/sender/sender.md) and [receiver.md](src/receiver/receiver.md)
for full configuration details.

---

## Supported codecs

| Flag | Codec |
|------|-------|
| `h265` | H.265 / HEVC (default) |
| `h264` | H.264 / AVC |
| `dcvcrt` | DCVC-RT (neural video codec) |

---

## Network emulation

Pass `--trace_path` to `sender-3d.py` to replay a bandwidth/loss trace using
Linux `tc` / `netem`.  Trace files are whitespace-separated with columns:

```
<timestamp_s>  <bandwidth_mbps>  <rtt_ms>  <loss_0_to_1>
```

RTT is fixed at 40 ms in the current setup; only bandwidth and loss are applied
from the trace.  Requires `sudo` for `tc` commands.
