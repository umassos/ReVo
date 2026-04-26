import time
import torch
import numpy as np
import av
from fractions import Fraction
import collections


class H264VideoCodec:
    """
    Simple H.264 (AVC) streaming wrapper using PyAV.

    Interface is compatible with your sender/receiver:
      - compress_stream(frames, frame_id, fps) -> yields dict with
        {frame_id, is_key, payload, qp, height, width}
      - decompress_stream(frame_dict, ...) -> yields dict with
        {frame_id, is_key, decoded_frame}
    """

    def __init__(self,
                 qp: int = 30,
                 intra_period: int = 30):
        self.qp_i = qp
        self.qp_p = qp
        self.intra_period = intra_period
        self.enc = None
        self.dec = None
        self.width = None
        self.height = None
        self.fps = 30
        # Queue of frame_ids that are "in flight" in x264's pipeline
        self._inflight_ids = collections.deque()

    def _ensure_encoder(self, width: int, height: int, fps: int):
        """
        Lazily create and configure the libx264 encoder.
        We keep it zero-latency and constant-QP.
        """
        if self.enc is not None:
            return

        self.width = width
        self.height = height
        self.fps = fps
        
        # Switched to libx264
        ctx = av.CodecContext.create("libx264", "w")
        ctx.width = width
        ctx.height = height
        ctx.pix_fmt = "yuv420p"
        ctx.time_base = Fraction(1, fps)
        
        qp_stream = self.qp_p

        # Note: x264 uses 'x264-params' similar to x265
        ctx.options = {
            "preset": "medium",
            "tune": "zerolatency",  
            "x264-params": (
                f"qp={qp_stream}:"
                f"keyint={self.intra_period}:"
                f"min-keyint={self.intra_period}:"
                "scenecut=0:bframes=0:rc-lookahead=0:sync-lookahead=0:ndirect_frames=0"
            ),
        }
        ctx.open()
        self.enc = ctx

    def _ensure_decoder(self):
        """
        Lazily create the H.264 decoder.
        """
        if self.dec is not None:
            return

        # Switched to h264 decoder
        ctx = av.CodecContext.create("h264", "r") 
        ctx.options = {"flags": "output_corrupt"}
        
        ctx.open()
        self.dec = ctx

    def compress_stream(self, frames: torch.Tensor, frame_id: int, fps: int = 30):
        """
        frames: (1, 1, C, H, W) torch tensor in [0,1] or [0,255], RGB
        """
        _, _, C, H, W = frames.shape
        assert C == 3, "Expected RGB (C=3)"
        self._ensure_encoder(W, H, fps)

        self._inflight_ids.append(frame_id)

        # torch -> uint8 RGB ndarray (H, W, 3)
        x = frames[0, 0]
        if x.dtype != torch.uint8:
            x = (x.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
        x = x.permute(1, 2, 0).cpu().numpy()

        frame = av.VideoFrame.from_ndarray(x, format="rgb24")

        # Encode frame
        packets = self.enc.encode(frame)
        if not packets:
            return

        payload = b"".join(bytes(p) for p in packets)
        if not payload:
            return

        out_id = self._inflight_ids.popleft()

        is_key = (out_id == 0 or
                  (self.intra_period > 0 and out_id % self.intra_period == 0))
        qp = self.qp_i if is_key else self.qp_p

        yield {
            "frame_id": out_id,
            "is_key": is_key,
            "payload": payload,
            "qp": qp,
            "height": H,
            "width": W,
        }

    def flush(self, fps: int = 30):
        """
        Flush remaining frames from the encoder.
        """
        if self.enc is None:
            return

        while True:
            packets = self.enc.encode(None)
            if not packets:
                break

            payload = b"".join(bytes(p) for p in packets)
            if not payload:
                continue

            if not self._inflight_ids:
                out_id = -1
            else:
                out_id = self._inflight_ids.popleft()

            H, W = self.height, self.width
            is_key = (out_id == 0 or
                      (self.intra_period > 0 and out_id % self.intra_period == 0))
            qp = self.qp_i if is_key else self.qp_p

            yield {
                "frame_id": out_id,
                "is_key": is_key,
                "payload": payload,
                "qp": qp,
                "height": H,
                "width": W,
            }

    def decompress_stream(self, frame, pic_height=None, pic_width=None, fps: int = 30):
        """
        Decodes H.264 payload back to RGB ndarray.
        """
        frame_id = frame["frame_id"]
        is_key = frame["is_key"]
        payload = frame["payload"]

        self._ensure_decoder()

        pkt = av.packet.Packet(payload)
        decoded_frames = self.dec.decode(pkt)
        
        for f in decoded_frames:
            rgb = f.to_ndarray(format="rgb24")
            yield {
                "frame_id": frame_id,
                "is_key": is_key,
                "decoded_frame": rgb,
            }
            break
