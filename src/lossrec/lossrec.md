# LossRec — Video Packet Loss Recovery

LossRec uses a masked-autoencoder-style Video Vision Transformer (VideoMAE) to
reconstruct frames that were corrupted or dropped by packet loss during video
transmission.  Separate models are trained for **RGB** and **Depth** streams.

Given a clip of `num_frames` frames — the last one being corrupted — the model
reconstructs the target frame by attending to clean temporal context from the
preceding frames.

> **Attribution** — the ViT backbone (`modeling_finetune.py`,
> `modeling_pretrain_0820.py`) is adapted from
> [VideoMAE](https://github.com/MCG-NJU/VideoMAE):
> Wang et al., *"VideoMAE: Masked Autoencoders are Data-Efficient Learners for
> Self-Supervised Video Pre-Training"*, NeurIPS 2022.

---

## Directory layout

```
src/lossrec/
├── rgb/
│   ├── modeling_finetune.py              # ViT building blocks (shared)
│   ├── modeling_pretrain_0820.py         # Encoder-decoder MAE model
│   ├── dataloader_finetune_inference.py  # RGB dataset
│   └── test_rgb_stage2.py               # RGB inference entry point
└── depth/
    ├── modeling_finetune.py              # ViT building blocks (shared)
    ├── modeling_pretrain_0820.py         # Encoder-decoder MAE model
    ├── dataloader_finetune_inference_depth.py  # Depth dataset
    └── test_depth_stage2.py             # Depth inference entry point
```

---

## Prerequisites

```bash
pip install torch torchvision timm einops pytorch-msssim torchcodec
```

The scripts must be run from `src/lossrec/` so that Python can resolve the
intra-package imports (`from modeling_pretrain_0820 import ...`).

---

## Input data layout

| Path | Contents |
|------|----------|
| `data/gt_rgb_looped/` | Ground-truth RGB videos (`.mp4` / `.avi`) |
| `data/gt_depth_looped/` | Ground-truth depth videos |
| `output/.../rgb/` | Codec-corrupted RGB videos (same filenames as GT) |
| `output/.../depth/` | Codec-corrupted depth videos |
| `output/.../frame_masks/` | Per-video frame masks (`<stem>_frame_mask.npy`) |

**Frame mask format** — a 1-D NumPy array of length `T` (number of frames).
`0` = clean frame, `> 0` = corrupted / lost frame.

---

## RGB inference

```bash
cd src/lossrec/

python rgb/test_rgb_stage2.py \
  --checkpoint     ../../.checkpoints/h264/h264_rgb.pth \
  --clean_path     ../../data/gt_rgb_looped/ \
  --corrupted_path ../../output/h265/receiver_logs/{network}/rgb \
  --mask_path      ../../output/h265/receiver_logs/{network}/frame_masks \
  --save_path      ../../output/h265/cell/rgb/
```

### All flags

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | *(required)* | Path to trained `.pth` checkpoint |
| `--clean_path` | *(required)* | Directory of ground-truth RGB videos |
| `--corrupted_path` | *(required)* | Directory of corrupted RGB videos |
| `--mask_path` | *(required)* | Directory of `_frame_mask.npy` files |
| `--save_path` | `output/v2` | Root output directory |
| `--input_size` | `512` | Spatial resolution the model was trained on |
| `--num_frames` | `6` | Clip length (5 clean context + 1 corrupted target) |
| `--tubelet_size` | `2` | Temporal depth of each 3-D patch token |
| `--patch_size` | `32` | Spatial patch size (square) 32 x 32|
| `--batch_size` | `1` | DataLoader batch size |
| `--num_workers` | `0` | DataLoader worker processes |
| `--device` | auto | `cuda` or `cpu` |

---

## Depth inference

```bash
cd src/lossrec/

python depth/test_depth_stage2.py \
  --checkpoint     ../../.checkpoints/h264/h264_depth.pth \
  --clean_path     ../../data/gt_depth_looped/ \
  --corrupted_path ../../output/h265/receiver_logs/{network}/depth \
  --mask_path      ../../output/h265/receiver_logs/{network}/frame_masks \
  --save_path      ../../output/h265/lossrec/cell/depth/
```

The depth script accepts the same flags as the RGB script.

> **Depth video naming** — depth videos may carry a `_vis` suffix
> (e.g. `scene_vis.mp4`).  The dataloader automatically strips `_vis` when
> looking up the corresponding mask file (`scene_frame_mask.npy`).

---

## Output structure

```
<save_path>/
├── <video_name>/
│   ├── final/
│   │   ├── frame_0000.png   ← reconstructed corrupted frames
│   │   └── frame_XXXX.png
├── csv_logs/
│   └── <video_name>_metrics.csv   ← per-frame PSNR and SSIM
└── summary_metrics.csv            ← per-video averages + global average
```

**NOTE**: The `final/` directory contains only the **reconstructed** corrupted frames.


---

## Checkpoints

Pre-trained checkpoints are hosted on Hugging Face at
[`umass-lass/ReVo`](https://huggingface.co/umass-lass/ReVo).

| Checkpoint | Stream | Codec |
|------------|--------|-------|
| `.checkpoints/h264/h264_rgb.pth` | RGB | H.264 |
| `.checkpoints/h264/h264_depth.pth` | Depth | H.264 |
| `.checkpoints/h265/h265_rgb.pth` | RGB | H.265 |
| `.checkpoints/h265/h265_depth.pth` | Depth | H.265 |
| `.checkpoints/dcvcrt/dcvcrt_rgb.pth` | RGB | DCVC-RT |
| `.checkpoints/dcvcrt/dcvcrt_depth.pth` | Depth | DCVC-RT |

### Downloading checkpoints

Install the Hugging Face hub client if needed:

```bash
pip install huggingface_hub
```

From the **repository root**, run:

```bash
bash scripts/download_checkpoints.sh
```

This saves all six checkpoints into `.checkpoints/` at the repository root,
preserving the codec subdirectory structure expected by the inference scripts.
`.checkpoints/` is **not** committed to git (add it to `.gitignore`).
