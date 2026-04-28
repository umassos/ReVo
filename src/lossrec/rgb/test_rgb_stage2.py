"""
RGB packet loss recovery inference script.

For each corrupted frame in the test set, the model receives a clip of
(num_frames - 1) clean reference frames followed by the corrupted target frame,
and reconstructs the target frame using temporal context.

Outputs:
  <save_path>/<video_name>/final/frame_XXXX.png  — per-frame reconstructions
  <save_path>/csv_logs/<video_name>_metrics.csv  — per-frame PSNR / SSIM
  <save_path>/summary_metrics.csv               — per-video and global averages

Usage:
  cd src/lossrec/
  python rgb/test_rgb_stage2.py \\
      --checkpoint  ../../.checkpoints/h265/h265_rgb.pth \\
      --clean_path  ../../data/gt_rgb_looped/ \\
      --corrupted_path ../../output/h265/receiver_logs/{network}/rgb \\
      --mask_path   ../../output/h265/receiver_logs/{network}/frame_masks \\
      --save_path   ../../output/h265/cell/rgb/
"""

import argparse
import os
import csv
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import OrderedDict, defaultdict
from einops import rearrange
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
import torchvision.transforms.v2 as transforms_v2
from torchcodec.decoders import VideoDecoder
from pytorch_msssim import MS_SSIM

from dataloader_finetune_inference import LossRecoveryData
from modeling_pretrain import PretrainVisionTransformer


to_pil = ToPILImage()


def _save_frame(t_chw: torch.Tensor, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    to_pil(t_chw.clamp(0, 1)).save(path)


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate VideoMAE-based model for RGB packet loss recovery")

    # Model architecture
    parser.add_argument("--checkpoint",    type=str, required=True,
                        help="Path to the trained model checkpoint (.pth)")
    parser.add_argument("--input_size",    type=int, default=512,
                        help="Spatial resolution (H=W) the model was trained with")
    parser.add_argument("--num_frames",    type=int, default=6,
                        help="Clip length: (num_frames-1) clean context + 1 corrupted target")
    parser.add_argument("--tubelet_size",  type=int, default=2,
                        help="Temporal depth of each 3-D patch token")
    parser.add_argument("--patch_size",    type=int, default=32,
                        help="Spatial size of each patch token (square)")

    # Data paths
    parser.add_argument("--clean_path",     type=str, required=True,
                        help="Directory of ground-truth (clean) videos")
    parser.add_argument("--corrupted_path", type=str, required=True,
                        help="Directory of codec-corrupted videos")
    parser.add_argument("--mask_path",      type=str, required=True,
                        help="Directory of per-video frame mask .npy files")

    # DataLoader
    parser.add_argument("--batch_size",   type=int, default=1)
    parser.add_argument("--num_workers",  type=int, default=0)
    parser.add_argument("--device",       type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    # Output
    parser.add_argument("--save_path",  type=str, default="output/v2",
                        help="Root directory for all outputs")
    parser.add_argument("--save_debug", action="store_true",
                        help="Also save orig / corrupted / recon triplets for each frame")

    return parser.parse_args()


def load_model(args, device):
    model = PretrainVisionTransformer(
        img_size=args.input_size,
        patch_size=args.patch_size,
        num_frames=args.num_frames,
        tubelet_size=args.tubelet_size,
        encoder_in_chans=3,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        decoder_embed_dim=384,
        decoder_depth=4,
        decoder_num_heads=6,
        mlp_ratio=4,
        norm_layer=torch.nn.LayerNorm,
    )

    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt

    # Strip DataParallel "module." prefix if present
    clean_sd = OrderedDict()
    for k, v in state_dict.items():
        clean_sd[k[7:] if k.startswith("module.") else k] = v

    model.load_state_dict(clean_sd, strict=False)
    model.to(device).eval()
    print(f"Loaded checkpoint: {args.checkpoint}")
    return model


def get_dataloader(args):
    dataset = LossRecoveryData(
        clean_dir=args.clean_path,
        corrupted_dir=args.corrupted_path,
        mask_dir=args.mask_path,
        num_frames=args.num_frames,
        input_size=args.input_size,
    )
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                      num_workers=args.num_workers, pin_memory=True)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, args):
    ssim_fn = MS_SSIM(data_range=1.0, size_average=True, channel=3).to(device)

    # video_results[name] = [{'frame': int, 'psnr': float, 'ssim': float}, ...]
    video_results = defaultdict(list)

    p_t = args.tubelet_size
    p_s = args.patch_size
    n_h = args.input_size // p_s  # number of spatial patch rows
    n_w = args.input_size // p_s  # number of spatial patch cols

    print(f"Running inference on {len(dataloader)} clips...")

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------
    for batch_idx, ((corrupted, mask), clean, meta, frame_indices) in enumerate(dataloader):
        corrupted = corrupted.to(device, non_blocking=True)
        mask      = mask.to(device, non_blocking=True)
        clean     = clean.to(device, non_blocking=True)

        video_name = meta["video_name"][0]
        target_idx = int(meta["start_frame"][0])  # global frame index of the corrupted target

        vroot     = Path(args.save_path) / video_name
        final_dir = vroot / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        if args.save_debug:
            for sub in ("orig", "corrupted", "recon"):
                (vroot / sub).mkdir(parents=True, exist_ok=True)

        # Forward pass: model predicts normalised patch pixel vectors for all tokens
        output = model(corrupted, mask=mask)

        # ------------------------------------------------------------------
        # Unpatchify: denormalise predictions using the per-patch mean/std of
        # the corrupted input, then reshape back to image space.
        # ------------------------------------------------------------------
        img_squeeze = rearrange(
            corrupted, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c',
            p0=p_t, p1=p_s, p2=p_s)

        rec_img = rearrange(output, 'b n (p c) -> b n p c', c=3)
        # Denormalise: pred = pred * std(patch) + mean(patch)
        rec_img = (
            rec_img * (img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
            + img_squeeze.mean(dim=-2, keepdim=True)
        )
        rec_img = rearrange(
            rec_img, 'b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2)',
            p0=p_t, p1=p_s, p2=p_s, h=n_h, w=n_w)

        # Evaluate on the last (target) frame only
        pred_last = rec_img[:, :, -1, :, :]
        gt_last   = clean[:, :, -1, :, :]

        mse      = criterion(pred_last, gt_last)
        ssim_val = ssim_fn(pred_last, gt_last)
        psnr     = 10 * torch.log10(1.0 / (mse + 1e-12))

        video_results[video_name].append({
            'frame': target_idx,
            'psnr':  psnr.item(),
            'ssim':  ssim_val.item(),
        })
        print(f"[{batch_idx}/{len(dataloader)}] {video_name} frame {target_idx} "
              f"| PSNR {psnr.item():.2f} dB  SSIM {ssim_val.item():.4f}")

        pred_cpu = pred_last.detach().cpu()[0]
        _save_frame(pred_cpu, final_dir / f"frame_{target_idx:04d}.png")

        if args.save_debug:
            _save_frame(gt_last.detach().cpu()[0],
                        vroot / "orig"      / f"frame_{target_idx:04d}.png")
            _save_frame(corrupted.detach().cpu()[0, :, -1],
                        vroot / "corrupted" / f"frame_{target_idx:04d}.png")
            _save_frame(pred_cpu,
                        vroot / "recon"     / f"frame_{target_idx:04d}.png")

    # ------------------------------------------------------------------
    # Write CSV reports
    # ------------------------------------------------------------------
    print("\nWriting CSV reports...")
    csv_dir = Path(args.save_path) / "csv_logs"
    csv_dir.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.save_path) / "summary_metrics.csv"

    global_psnr = global_ssim = 0.0
    total_videos = 0

    with open(summary_path, mode='w', newline='') as sf:
        sw = csv.writer(sf)
        sw.writerow(["Video Name", "Average PSNR", "Average SSIM", "Frame Count"])

        for vid in sorted(video_results):
            frames = sorted(video_results[vid], key=lambda x: x['frame'])
            vid_psnr = vid_ssim = 0.0

            with open(csv_dir / f"{vid}_metrics.csv", mode='w', newline='') as vf:
                vw = csv.writer(vf)
                vw.writerow(["Frame Index", "PSNR", "SSIM"])
                for fd in frames:
                    vw.writerow([fd['frame'], f"{fd['psnr']:.4f}", f"{fd['ssim']:.4f}"])
                    vid_psnr += fd['psnr']
                    vid_ssim += fd['ssim']

            n = len(frames)
            if n > 0:
                avg_p, avg_s = vid_psnr / n, vid_ssim / n
                sw.writerow([vid, f"{avg_p:.4f}", f"{avg_s:.4f}", n])
                global_psnr += avg_p
                global_ssim += avg_s
                total_videos += 1
            else:
                sw.writerow([vid, "N/A", "N/A", 0])

        if total_videos > 0:
            fp = global_psnr / total_videos
            fs = global_ssim / total_videos
            sw.writerow([])
            sw.writerow(["GLOBAL AVERAGE", f"{fp:.4f}", f"{fs:.4f}", total_videos])
            print(f"Global Average PSNR: {fp:.4f} dB")
            print(f"Global Average SSIM: {fs:.4f}")

    return global_psnr / max(1, total_videos), global_ssim / max(1, total_videos)


def fill_clean_frames(args):
    """Optional post-processing: copy clean (non-corrupted) frames into final/.

    After inference, this fills the gaps in the final/ output directory with
    the original clean frames, producing a complete reconstructed video sequence.
    Call this manually after `evaluate()` if a full frame sequence is needed.
    """
    resize = transforms_v2.Compose([
        transforms_v2.ToDtype(torch.float32, scale=True),
        transforms_v2.Resize(size=(args.input_size, args.input_size), antialias=True),
    ])

    corrupted_files = [f for f in os.listdir(args.corrupted_path)
                       if f.endswith((".mp4", ".avi"))]
    # Build a map from gt_stem → corrupted_filename (handles optional _trace suffix)
    gt_to_corr = {}
    for cf in corrupted_files:
        stem = cf.split("_trace")[0] if "_trace" in cf else os.path.splitext(cf)[0]
        gt_to_corr[stem] = cf

    for vf in os.listdir(args.clean_path):
        if not vf.endswith((".mp4", ".avi")):
            continue
        video_name    = os.path.splitext(vf)[0]
        clean_path    = os.path.join(args.clean_path, vf)
        corr_filename = gt_to_corr.get(video_name)
        if corr_filename is None:
            continue

        mask_path = os.path.join(
            args.corrupted_path, os.path.splitext(corr_filename)[0] + "_frame_mask.npy")
        if not os.path.exists(mask_path):
            continue

        frame_mask = np.load(mask_path, allow_pickle=True).astype(bool)
        dec_clean  = VideoDecoder(clean_path, device="cpu")
        limit      = min(len(dec_clean), len(frame_mask))

        corr_path = os.path.join(args.corrupted_path, corr_filename)
        if os.path.exists(corr_path):
            try:
                limit = min(limit, len(VideoDecoder(corr_path, device="cpu")))
            except Exception:
                pass

        frame_mask   = frame_mask[:limit]
        clean_indices = np.where(~frame_mask)[0]
        final_dir = Path(args.save_path) / video_name / "final"
        final_dir.mkdir(parents=True, exist_ok=True)

        for i in clean_indices:
            out = final_dir / f"frame_{int(i):04d}.png"
            if out.exists():
                continue
            _save_frame(resize(dec_clean[int(i)]), out)

    print("Done filling clean frames into final/.")


if __name__ == "__main__":
    args   = get_args()
    device = torch.device(args.device)
    model  = load_model(args, device)
    dl     = get_dataloader(args)
    evaluate(model, dl, nn.MSELoss(), device, args)

    # Uncomment to copy clean frames into final/ for a complete sequence:
    # fill_clean_frames(args)
