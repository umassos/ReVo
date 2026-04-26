import time
import torch
import numpy as np
import gc

from dcvc.layers.cuda_inference import replicate_pad
from dcvc.models.video_model import DMC
from dcvc.models.image_model import DMCI
from dcvc.utils.common import str2bool, get_state_dict, set_torch_env
from dcvc.utils.stream_helper import SPSHelper
from dcvc.utils.metrics import calc_psnr, calc_msssim, calc_msssim_rgb
from dcvc.utils.transforms import rgb2ycbcr, ycbcr2rgb
from torch import Tensor

# ITU-R BT.709 weights (same as dcvc.utils.transforms.YCBCR_WEIGHTS["ITU-R_BT.709"])
@torch.jit.script
def _dcvc_ycbcr2rgb_bt709(ycbcr: Tensor) -> Tensor:
    """
    Minimal TorchScript-compatible YCbCr -> RGB (BT.709) converter.
    ycbcr: (N, 3, H, W), values in [0,1].

    Returns: (N, 3, H, W), float32 in [0,1].
    """
    _DCVC_KR: float = 0.2126
    _DCVC_KB: float = 0.0722
    _DCVC_KG: float = 1.0 - _DCVC_KR - _DCVC_KB  # 0.7152
    y, cb, cr = ycbcr.chunk(3, dim=1)

    # r, b
    r = y + (2.0 - 2.0 * _DCVC_KR) * (cr - 0.5)
    b = y + (2.0 - 2.0 * _DCVC_KB) * (cb - 0.5)

    # g (using Kr + Kg + Kb = 1)
    g = (y - _DCVC_KR * r - _DCVC_KB * b) / _DCVC_KG

    rgb = torch.cat([r, g, b], dim=1)
    return rgb


@torch.jit.script
def dcvc_postprocess_ycbcr(
    x_hat_yuv: Tensor,
    pic_height: int,
    pic_width: int,
) -> Tensor:
    """
    Crop + YCbCr->RGB + clamp + scale + uint8, all TorchScript-compatible.

    x_hat_yuv: (1, 3, H', W') float/half on GPU
    returns:   (1, 3, H, W) uint8 on GPU
    """
    x_hat_yuv = x_hat_yuv[:, :, :pic_height, :pic_width]
    x_hat_rgb = _dcvc_ycbcr2rgb_bt709(x_hat_yuv)
    x_hat_rgb = x_hat_rgb.clamp(0.0, 1.0).mul(255.0).to(torch.uint8)
    return x_hat_rgb

class DCVCVideoCodec():
    def __init__(self, intra_period=30):

        # settings and configuration
        # [DM] Later, read from a config.json file
        set_torch_env()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path_i = "../../.checkpoints/cvpr2025_image.pth.tar"
        self.model_path_p = "../../.checkpoints/cvpr2025_video.pth.tar"
        self.force_zero_thres = 0.12
        self.reset_interval = 64
        self.force_intra = False
        self.intra_period = intra_period
        self.qp_i = 31
        self.qp_p = 63

        self.host_frame_t = None  # torch pinned tensor (H, W, 3, uint8, cpu, pin_memory=True)
        self.host_frame = None    # numpy view on host_frame_t

        # I-frame codec
        i_frame_net = DMCI()
        i_state_dict = get_state_dict(self.model_path_i)
        i_frame_net.load_state_dict(i_state_dict)
        i_frame_net = i_frame_net.to(self.device)
        i_frame_net.eval()
        i_frame_net.update(self.force_zero_thres)
        i_frame_net.half()
        self.i_frame_net = i_frame_net

        # P-frame codec
        p_frame_net = DMC()
        if not self.force_intra:
            p_state_dict = get_state_dict(self.model_path_p)
            p_frame_net.load_state_dict(p_state_dict)
            p_frame_net = p_frame_net.to(self.device)
            p_frame_net.eval()
            p_frame_net.update(self.force_zero_thres)
            p_frame_net.half()
        self.p_frame_net = p_frame_net

        # --- Initialize state ---
        self.p_frame_net.set_curr_poc(0)
        self.p_frame_net.clear_dpb()
        self.sps_helper = None
        self._sps = {
            "sps_id": -1,
            "height": 0,
            "width": 0,
            "ec_part": 0,
            "use_ada_i": 0,
        }
        self.last_qp = 0
        self.index_map = [0, 1, 0, 2, 0, 2, 0, 2]

    def compress_stream(self, frames: torch.Tensor, frame_id: int, fps: int = 30):
        """
        Frame-by-frame streaming encoder for DCVC-RT.
        Yields one compressed frame at a time as a dictionary:
        {
            "frame_id": int,
            "is_key": bool,
            "pts_ms": int,
            "payload": bytes
        }
        """

        N, T, C, H, W = frames.shape
        pic_height, pic_width = H, W

        # --- Set padding and entropy coder mode ---
        padding_r, padding_b = DMCI.get_padding_size(pic_height, pic_width, 16)
        use_two_entropy_coders = pic_height * pic_width > 1280 * 720
        self.i_frame_net.set_use_two_entropy_coders(use_two_entropy_coders)
        self.p_frame_net.set_use_two_entropy_coders(use_two_entropy_coders)

        # --- Initialize SPS helper and DPB ---

        
        t0 = time.perf_counter()

        with torch.no_grad():
            if (frame_id == 0):
                self.p_frame_net.set_curr_poc(0)
                self.sps_helper = SPSHelper()

            x = frames[:, 0, :, :, :].to(self.device)
            x = rgb2ycbcr(x).to(torch.float16)
            x_padded = replicate_pad(x, padding_b, padding_r)

            # --- Determine I/P frame ---
            is_i_frame = (frame_id == 0 or 
                        (self.intra_period > 0 and frame_id % self.intra_period == 0))

            if is_i_frame:
                curr_qp = self.qp_i
                encoded = self.i_frame_net.compress(x_padded, curr_qp)
                self.p_frame_net.clear_dpb()
                self.p_frame_net.add_ref_frame(None, encoded["x_hat"])
                use_ada_i = 0
            else:
                fa_idx = self.index_map[frame_id % 8]
                if self.reset_interval > 0 and frame_id % self.reset_interval == 1:
                    use_ada_i = 1
                    self.p_frame_net.prepare_feature_adaptor_i(self.last_qp)
                else:
                    use_ada_i = 0
                curr_qp = self.qp_p # [DM] fixing p-frame QP
                # curr_qp = self.p_frame_net.shift_qp(self.qp_p, fa_idx)
                encoded = self.p_frame_net.compress(x_padded, curr_qp)
                self.last_qp = curr_qp

            # --- SPS bookkeeping (optional for streaming) ---
            sps = {
                "sps_id": -1,
                "height": pic_height,
                "width": pic_width,
                "ec_part": 1 if use_two_entropy_coders else 0,
                "use_ada_i": use_ada_i
            }
            sps_id, _ = self.sps_helper.get_sps_id(sps)
            sps["sps_id"] = sps_id

            # --- Get bitstream bytes for this frame ---
            payload = encoded["bit_stream"]
            pts_ms = int((time.perf_counter() - t0) * 1000)
            # print(f"DCVC encode time: {pts_ms} ms")

            yield {
                "frame_id": frame_id,
                "is_key": is_i_frame,
                "pts_ms": pts_ms,
                "payload": payload,
                "qp": curr_qp,
                "height": pic_height,
                "width": pic_width,
            }

    def decompress_stream(self, frame, pic_height=None, pic_width = None, fps=30):
        """
        Frame-by-frame streaming DCVC decoder.

        Args:
            codec: an instance of evdcvc.DCVCVideoCodec (with i/p nets loaded)
            frame_iter: an iterable or generator yielding dicts:
                        {
                        "frame_id": int,
                        "is_key": bool,
                        "payload": bytes
                        }
            fps (int): nominal frame rate (for pacing/logging)
        Yields:
            dict with {
                "frame_id": int,
                "is_key": bool,
                "decoded_frame": np.ndarray (H, W, 3) uint8
            }
        """

        frame_id = frame["frame_id"]
        is_key = frame["is_key"]
        payload = frame["payload"]
        qp = frame["qp"]

        i_frame_net = self.i_frame_net
        p_frame_net = self.p_frame_net

        if (frame_id == 0):
            p_frame_net.set_curr_poc(0)

        if pic_height is None or pic_width is None:
            pic_height, pic_width = 720, 1280

        sps = self._sps
        sps["height"]   = pic_height
        sps["width"]    = pic_width
        sps["use_ada_i"] = 0        
        use_ada_i = 0
        
        try:
            # st = time.time()
            if is_key:
                decoded = i_frame_net.decompress(payload, sps, qp)
                p_frame_net.clear_dpb()
                p_frame_net.add_ref_frame(None, decoded["x_hat"])
            else:
                if self.reset_interval > 0 and frame_id % self.reset_interval == 1:
                    use_ada_i = 1
                if (use_ada_i):
                    sps["use_ada_i"] = use_ada_i
                    p_frame_net.reset_ref_feature()
                decoded = p_frame_net.decompress(payload, sps, qp)
            # print("DCVC decode time: ", (time.time()-st)*1000)

            # --- Convert YCbCr → RGB numpy frame ---
            recon = decoded["x_hat"]  # (1, 3, H', W')
            # scripted postprocess: crop + ycbcr2rgb + clamp + scale + uint8
            x_hat_rgb = dcvc_postprocess_ycbcr(recon, pic_height, pic_width)  # (1,3,H,W), uint8 on GPU

            # Reorder to HWC on GPU
            gpu_frame = x_hat_rgb[0].permute(1, 2, 0).contiguous()  # (H, W, 3), uint8 on GPU

            # Allocate / reuse pinned host buffer once
            if (self.host_frame_t is None) or (self.host_frame_t.shape != gpu_frame.shape):
                # pinned CPU tensor
                self.host_frame_t = torch.empty(
                    gpu_frame.shape,
                    dtype=torch.uint8,
                    device="cpu",
                    pin_memory=True,
                )
                self.host_frame = self.host_frame_t.numpy()

            # Non-blocking copy from GPU → pinned CPU
            self.host_frame_t.copy_(gpu_frame, non_blocking=True)
            frame_rgb = self.host_frame  # (H, W, 3) uint8, numpy array backed by pinned tensor
            
            # print("DCVC time to convert to frame: ", (time.time()-st)*1000)

            yield {
                "frame_id": frame_id,
                "is_key": is_key,
                "decoded_frame": frame_rgb,
            }

        except Exception as e:
            print(f"[WARN] Decode failed for frame {frame_id}: {e}")
    
    
