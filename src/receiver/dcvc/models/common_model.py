# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torch import nn
import math

from ..layers.cuda_inference import combine_for_reading_2x, \
    restore_y_2x, restore_y_2x_with_cat_after, add_and_multiply, \
    replicate_pad, restore_y_4x, clamp_reciprocal_with_quant
from .entropy_models import BitEstimator, GaussianEncoder, EntropyCoder


class CompressionModel(nn.Module):
    def __init__(self, z_channel, extra_qp=0):
        super().__init__()

        self.z_channel = z_channel
        self.entropy_coder = None
        self.bit_estimator_z = BitEstimator(64 + extra_qp, z_channel)
        self.gaussian_encoder = GaussianEncoder()

        self.masks = {}
        self.cuda_streams = {}

    def get_cuda_stream(self, device, idx=0, priority=0):
        key = f"{device}_{priority}_{idx}"
        if key not in self.cuda_streams:
            self.cuda_streams[key] = torch.cuda.Stream(device, priority=priority)
        return self.cuda_streams[key]

    @staticmethod
    def get_qp_num():
        return 64

    @staticmethod
    def get_padding_size(height, width, p=64):
        new_h = (height + p - 1) // p * p
        new_w = (width + p - 1) // p * p
        padding_right = new_w - width
        padding_bottom = new_h - height
        return padding_right, padding_bottom

    @staticmethod
    def get_downsampled_shape(height, width, p):
        new_h = (height + p - 1) // p * p
        new_w = (width + p - 1) // p * p
        return int(new_h / p + 0.5), int(new_w / p + 0.5)

    def update(self, force_zero_thres=None):
        self.entropy_coder = EntropyCoder()
        self.gaussian_encoder.update(self.entropy_coder, force_zero_thres=force_zero_thres)
        self.bit_estimator_z.update(self.entropy_coder)

    def set_use_two_entropy_coders(self, use_two_entropy_coders):
        self.entropy_coder.set_use_two_entropy_coders(use_two_entropy_coders)

    def pad_for_y(self, y):
        _, _, H, W = y.size()
        padding_r, padding_b = self.get_padding_size(H, W, 4)
        y_pad = replicate_pad(y, padding_b, padding_r)
        return y_pad

    def separate_prior(self, params, is_video=False):
        if is_video:
            quant_step, scales, means = params.chunk(3, 1)
            quant_step = torch.clamp_min(quant_step, 0.5)
            q_enc = 1. / quant_step
            q_dec = quant_step
        else:
            q = params[:, :2, :, :]
            q_enc, q_dec = (torch.sigmoid(q) * 1.5 + 0.5).chunk(2, 1)
            scales, means = params[:, 2:, :, :].chunk(2, 1)
        return q_enc, q_dec, scales, means

    @staticmethod
    def separate_prior_for_video_encoding(params, y):
        q_dec, scales, means = params.chunk(3, 1)
        q_dec, y = clamp_reciprocal_with_quant(q_dec, y, 0.5)
        return y, q_dec, scales, means

    @staticmethod
    def separate_prior_for_video_decoding(params):
        quant_step, scales, means = params.chunk(3, 1)
        quant_step = torch.clamp_min(quant_step, 0.5)
        return quant_step, scales, means

    def process_with_mask(self, y, scales, means, mask):
        return self.gaussian_encoder.process_with_mask(y, scales, means, mask)

    @staticmethod
    def get_one_mask(micro_mask, height, width, dtype, device):
        mask = torch.tensor(micro_mask, dtype=dtype, device=device)
        mask = mask.repeat((height + 1) // 2, (width + 1) // 2)
        mask = mask[:height, :width]
        mask = torch.unsqueeze(mask, 0)
        mask = torch.unsqueeze(mask, 0)
        return mask

    def get_mask_4x(self, batch, channel, height, width, dtype, device):
        curr_mask_str = f"{batch}_{channel}_{width}_{height}_4x"
        with torch.no_grad():
            if curr_mask_str not in self.masks:
                assert channel % 4 == 0
                m = torch.ones((batch, channel // 4, height, width), dtype=dtype, device=device)
                m0 = self.get_one_mask(((1, 0), (0, 0)), height, width, dtype, device)
                m1 = self.get_one_mask(((0, 1), (0, 0)), height, width, dtype, device)
                m2 = self.get_one_mask(((0, 0), (1, 0)), height, width, dtype, device)
                m3 = self.get_one_mask(((0, 0), (0, 1)), height, width, dtype, device)

                mask_0 = torch.cat((m * m0, m * m1, m * m2, m * m3), dim=1)
                mask_1 = torch.cat((m * m3, m * m2, m * m1, m * m0), dim=1)
                mask_2 = torch.cat((m * m2, m * m3, m * m0, m * m1), dim=1)
                mask_3 = torch.cat((m * m1, m * m0, m * m3, m * m2), dim=1)

                self.masks[curr_mask_str] = [mask_0, mask_1, mask_2, mask_3]
        return self.masks[curr_mask_str]

    def get_mask_2x(self, batch, channel, height, width, dtype, device):
        curr_mask_str = f"{batch}_{channel}_{width}_{height}_2x"
        with torch.no_grad():
            if curr_mask_str not in self.masks:
                assert channel % 2 == 0
                m = torch.ones((batch, channel // 2, height, width), dtype=dtype, device=device)
                m0 = self.get_one_mask(((1, 0), (0, 1)), height, width, dtype, device)
                m1 = self.get_one_mask(((0, 1), (1, 0)), height, width, dtype, device)

                mask_0 = torch.cat((m * m0, m * m1), dim=1)
                mask_1 = torch.cat((m * m1, m * m0), dim=1)

                self.masks[curr_mask_str] = [mask_0, mask_1]
        return self.masks[curr_mask_str]

    @staticmethod
    def single_part_for_writing_4x(x):
        x0, x1, x2, x3 = x.chunk(4, 1)
        return (x0 + x1) + (x2 + x3)

    @staticmethod
    def single_part_for_writing_2x(x):
        x0, x1 = x.chunk(2, 1)
        return x0 + x1

    def compress_prior_2x(self, y, common_params, y_spatial_prior):
        y, q_dec, scales, means = self.separate_prior_for_video_encoding(common_params, y)
        dtype = y.dtype
        device = y.device
        B, C, H, W = y.size()
        mask_0, mask_1 = self.get_mask_2x(B, C, H, W, dtype, device)

        _, y_q_0, y_hat_0, s_hat_0 = self.process_with_mask(y, scales, means, mask_0)
        cat_params = torch.cat((y_hat_0, common_params), dim=1)
        scales, means = y_spatial_prior(cat_params).chunk(2, 1)
        _, y_q_1, y_hat_1, s_hat_1 = self.process_with_mask(y, scales, means, mask_1)

        y_hat = add_and_multiply(y_hat_0, y_hat_1, q_dec)

        y_q_w_0 = self.single_part_for_writing_2x(y_q_0)
        y_q_w_1 = self.single_part_for_writing_2x(y_q_1)
        s_w_0 = self.single_part_for_writing_2x(s_hat_0)
        s_w_1 = self.single_part_for_writing_2x(s_hat_1)
        return y_q_w_0, y_q_w_1, s_w_0, s_w_1, y_hat

    def decompress_prior_2x(self, common_params, y_spatial_prior):
        infos = self.decompress_prior_2x_part1(common_params)
        y_hat = self.decompress_prior_2x_part2(common_params, y_spatial_prior, infos)
        return y_hat

    def decompress_prior_2x_part1(self, common_params):
        q_dec, scales, means = self.separate_prior_for_video_decoding(common_params)
        dtype = means.dtype
        device = means.device
        B, C, H, W = means.size()
        mask_0, mask_1 = self.get_mask_2x(B, C, H, W, dtype, device)

        scales_r = combine_for_reading_2x(scales, mask_0, inplace=False)
        indexes, skip_cond = self.gaussian_encoder.build_indexes_decoder(scales_r)
        self.gaussian_encoder.decode_y(indexes)
        infos = {
            "q_dec": q_dec,
            "mask_0": mask_0,
            "mask_1": mask_1,
            "means": means,
            "scales_r": scales_r,
            "skip_cond": skip_cond,
            "indexes": indexes,
        }
        return infos

    def decompress_prior_2x_part2(self, common_params, y_spatial_prior, infos):
        dtype = common_params.dtype
        device = common_params.device
        y_q_r = self.gaussian_encoder.get_y(infos["scales_r"].shape,
                                            infos["scales_r"].numel(),
                                            dtype, device,
                                            infos["skip_cond"], infos["indexes"])
        y_hat_0, cat_params = restore_y_2x_with_cat_after(y_q_r, infos["means"], infos["mask_0"],
                                                          common_params)
        scales, means = y_spatial_prior(cat_params).chunk(2, 1)
        scales_r = combine_for_reading_2x(scales, infos["mask_1"], inplace=True)
        y_q_r = self.gaussian_encoder.decode_and_get_y(scales_r, dtype, device)
        y_hat_1 = restore_y_2x(y_q_r, means, infos["mask_1"])

        y_hat = add_and_multiply(y_hat_0, y_hat_1, infos["q_dec"])
        return y_hat

    def compress_prior_4x(self, y, common_params, y_spatial_prior_reduction,
                          y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                          y_spatial_prior_adaptor_3, y_spatial_prior):
        '''
        y_0 means split in channel, the 0/4 quater
        y_1 means split in channel, the 1/4 quater
        y_2 means split in channel, the 2/4 quater
        y_3 means split in channel, the 3/4 quater
        y_?_0, means multiply with mask_0
        y_?_1, means multiply with mask_1
        y_?_2, means multiply with mask_2
        y_?_3, means multiply with mask_3
        '''
        q_enc, q_dec, scales, means = self.separate_prior(common_params, False)
        common_params = y_spatial_prior_reduction(common_params)
        dtype = y.dtype
        device = y.device
        B, C, H, W = y.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_4x(B, C, H, W, dtype, device)

        y = y * q_enc

        _, y_q_0, y_hat_0, s_hat_0 = self.process_with_mask(y, scales, means, mask_0)

        y_hat_so_far = y_hat_0
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_1(params)).chunk(2, 1)
        _, y_q_1, y_hat_1, s_hat_1 = self.process_with_mask(y, scales, means, mask_1)

        y_hat_so_far = y_hat_so_far + y_hat_1
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_2(params)).chunk(2, 1)
        _, y_q_2, y_hat_2, s_hat_2 = self.process_with_mask(y, scales, means, mask_2)

        y_hat_so_far = y_hat_so_far + y_hat_2
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_3(params)).chunk(2, 1)
        _, y_q_3, y_hat_3, s_hat_3 = self.process_with_mask(y, scales, means, mask_3)

        y_hat = y_hat_so_far + y_hat_3
        y_hat = y_hat * q_dec

        y_q_w_0 = self.single_part_for_writing_4x(y_q_0)
        y_q_w_1 = self.single_part_for_writing_4x(y_q_1)
        y_q_w_2 = self.single_part_for_writing_4x(y_q_2)
        y_q_w_3 = self.single_part_for_writing_4x(y_q_3)
        s_w_0 = self.single_part_for_writing_4x(s_hat_0)
        s_w_1 = self.single_part_for_writing_4x(s_hat_1)
        s_w_2 = self.single_part_for_writing_4x(s_hat_2)
        s_w_3 = self.single_part_for_writing_4x(s_hat_3)
        return y_q_w_0, y_q_w_1, y_q_w_2, y_q_w_3, s_w_0, s_w_1, s_w_2, s_w_3, y_hat

    def decompress_prior_4x(self, common_params, y_spatial_prior_reduction,
                            y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                            y_spatial_prior_adaptor_3, y_spatial_prior):
        _, quant_step, scales, means = self.separate_prior(common_params, False)
        common_params = y_spatial_prior_reduction(common_params)
        dtype = means.dtype
        device = means.device
        B, C, H, W = means.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_4x(B, C, H, W, dtype, device)

        scales_r = self.single_part_for_writing_4x(scales * mask_0)
        y_q_r = self.gaussian_encoder.decode_and_get_y(scales_r, dtype, device)
        y_hat_curr_step = restore_y_4x(y_q_r, means, mask_0)
        y_hat_so_far = y_hat_curr_step

        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_1(params)).chunk(2, 1)
        scales_r = self.single_part_for_writing_4x(scales * mask_1)
        y_q_r = self.gaussian_encoder.decode_and_get_y(scales_r, dtype, device)
        y_hat_curr_step = restore_y_4x(y_q_r, means, mask_1)
        y_hat_so_far = y_hat_so_far + y_hat_curr_step

        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_2(params)).chunk(2, 1)
        scales_r = self.single_part_for_writing_4x(scales * mask_2)
        y_q_r = self.gaussian_encoder.decode_and_get_y(scales_r, dtype, device)
        y_hat_curr_step = restore_y_4x(y_q_r, means, mask_2)
        y_hat_so_far = y_hat_so_far + y_hat_curr_step

        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_3(params)).chunk(2, 1)
        scales_r = self.single_part_for_writing_4x(scales * mask_3)
        y_q_r = self.gaussian_encoder.decode_and_get_y(scales_r, dtype, device)
        y_hat_curr_step = restore_y_4x(y_q_r, means, mask_3)
        y_hat_so_far = y_hat_so_far + y_hat_curr_step

        y_hat = y_hat_so_far * quant_step

        return y_hat

    # Lingdong Wang modification
    @staticmethod
    def probs_to_bits(probs):
        bits = -1.0 * torch.log(probs + 1e-5) / math.log(2.0)
        bits = torch.clamp_min(bits, 0)
        return bits

    def get_y_gaussian_bits(self, y, sigma):
        mu = torch.zeros_like(sigma)
        sigma = sigma.clamp(1e-5, 1e10)
        gaussian = torch.distributions.normal.Normal(mu, sigma)
        probs = gaussian.cdf(y + 0.5) - gaussian.cdf(y - 0.5)
        return self.probs_to_bits(probs)

    def get_y_laplace_bits(self, y, sigma):
        mu = torch.zeros_like(sigma)
        sigma = sigma.clamp(1e-5, 1e10)
        gaussian = torch.distributions.laplace.Laplace(mu, sigma)
        probs = gaussian.cdf(y + 0.5) - gaussian.cdf(y - 0.5)
        return self.probs_to_bits(probs)

    def get_z_bits(self, z, bit_estimator, qp):
        probs = bit_estimator.get_cdf(z + 0.5, qp) - bit_estimator.get_cdf(z - 0.5, qp)
        return self.probs_to_bits(probs)

    @staticmethod
    def quantize(x, method="ste"):
        if method == "uniform":
            noise = torch.nn.init.uniform_(torch.zeros_like(x), -0.5, 0.5)
            x_hat = x + noise
        elif method == "ste":
            # Straight-through estimator
            x_hat = x + (torch.round(x) - x).detach()
        elif method == "round":
            x_hat = torch.round(x)
        else:
            raise ValueError("Unknown quantization method {}".format(method))
        x_hat = torch.clamp(x_hat, -128., 127.)
        return x_hat

    def process_with_mask_diff(self, y, scales, means, mask, force_zero_thres):
        scales_hat = scales * mask
        means_hat = means * mask

        y_res = (y - means_hat) * mask
        y_q = self.quantize(y_res)
        if force_zero_thres is not None:
            cond = scales_hat > force_zero_thres
            y_q = y_q * cond
        y_q = torch.clamp(y_q, -128., 127.)
        y_hat = y_q + means_hat

        return y_res, y_q, y_hat, scales_hat

    def compress_prior_4x_diff(self, y, common_params, y_spatial_prior_reduction,
                               y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                               y_spatial_prior_adaptor_3, y_spatial_prior):
        """
        Differentiable version of compress_prior_4x for training.
        Replaces actual quantization with STE and calculates bit rates.

        Args:
            y: The input tensor to be compressed
            common_params: Common parameters for the model
            y_spatial_prior_reduction: Spatial prior reduction layer
            y_spatial_prior_adaptor_1/2/3: Spatial prior adaptor layers
            y_spatial_prior: Spatial prior layer

        Returns:
            y_hat: The reconstructed tensor
            bits: Estimated bits for rate calculation
        """
        q_enc, q_dec, scales, means = self.separate_prior(common_params, False)
        common_params = y_spatial_prior_reduction(common_params)
        dtype = y.dtype
        device = y.device
        B, C, H, W = y.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_4x(B, C, H, W, dtype, device)

        y = y * q_enc

        # First mask
        y_res_0, y_q_0, y_hat_0, scales_hat_0 = self.process_with_mask_diff(y, scales, means, mask_0, None)
        y_hat_so_far = y_hat_0

        # Calculate bits for the first mask
        bits_0 = self.get_y_gaussian_bits(y_q_0, scales_hat_0)

        # Second mask
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_1(params)).chunk(2, 1)
        y_res_1, y_q_1, y_hat_1, scales_hat_1 = self.process_with_mask_diff(y, scales, means, mask_1, None)
        y_hat_so_far = y_hat_so_far + y_hat_1

        # Calculate bits for the second mask
        bits_1 = self.get_y_gaussian_bits(y_q_1, scales_hat_1)

        # Third mask
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_2(params)).chunk(2, 1)
        y_res_2, y_q_2, y_hat_2, scales_hat_2 = self.process_with_mask_diff(y, scales, means, mask_2, None)
        y_hat_so_far = y_hat_so_far + y_hat_2

        # Calculate bits for the third mask
        bits_2 = self.get_y_gaussian_bits(y_q_2, scales_hat_2)

        # Fourth mask
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_3(params)).chunk(2, 1)
        y_res_3, y_q_3, y_hat_3, scales_hat_3 = self.process_with_mask_diff(y, scales, means, mask_3, None)

        # Calculate bits for the fourth mask
        bits_3 = self.get_y_gaussian_bits(y_q_3, scales_hat_3)

        # Final reconstruction
        y_hat = y_hat_so_far + y_hat_3
        y_hat = y_hat * q_dec

        # Compute total bits
        bits = bits_0 + bits_1 + bits_2 + bits_3

        return y_hat, bits

    def decompress_prior_4x_diff(self, common_params, y_spatial_prior_reduction,
                                 y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                                 y_spatial_prior_adaptor_3, y_spatial_prior):
        """
        Differentiable version of decompress_prior_4x for training.
        This version mimics the reconstruction process but keeps gradients flowing.

        Args:
            common_params: Common parameters for the model
            y_spatial_prior_reduction: Spatial prior reduction layer
            y_spatial_prior_adaptor_1/2/3: Spatial prior adaptor layers
            y_spatial_prior: Spatial prior layer

        Returns:
            y_hat: The reconstructed tensor
        """
        _, quant_step, scales, means = self.separate_prior(common_params, False)
        common_params = y_spatial_prior_reduction(common_params)
        dtype = means.dtype
        device = means.device
        B, C, H, W = means.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_4x(B, C, H, W, dtype, device)

        # Simulate first decompression step with STE
        scales_masked = scales * mask_0
        y_res = torch.zeros_like(scales_masked)
        y_q = self.quantize(y_res, method="ste")
        y_hat_curr_step = y_q + means * mask_0
        y_hat_so_far = y_hat_curr_step

        # Second step
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_1(params)).chunk(2, 1)
        scales_masked = scales * mask_1
        y_res = torch.zeros_like(scales_masked)
        y_q = self.quantize(y_res, method="ste")
        y_hat_curr_step = y_q + means * mask_1
        y_hat_so_far = y_hat_so_far + y_hat_curr_step

        # Third step
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_2(params)).chunk(2, 1)
        scales_masked = scales * mask_2
        y_res = torch.zeros_like(scales_masked)
        y_q = self.quantize(y_res, method="ste")
        y_hat_curr_step = y_q + means * mask_2
        y_hat_so_far = y_hat_so_far + y_hat_curr_step

        # Fourth step
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_3(params)).chunk(2, 1)
        scales_masked = scales * mask_3
        y_res = torch.zeros_like(scales_masked)
        y_q = self.quantize(y_res, method="ste")
        y_hat_curr_step = y_q + means * mask_3
        y_hat_so_far = y_hat_so_far + y_hat_curr_step

        # Apply quantization step
        y_hat = y_hat_so_far * quant_step

        return y_hat

    def compress_prior_2x_diff(self, y, common_params, y_spatial_prior):
        """
        Differentiable version of compress_prior_2x for training.

        Args:
            y: The input tensor to be compressed
            common_params: Common parameters for the model
            y_spatial_prior: Spatial prior module

        Returns:
            y_hat: The reconstructed tensor
            bits: Estimated bits for rate calculation
        """
        # Extract parameters from common_params
        y, q_dec, scales, means = self.separate_prior_for_video_encoding_diff(common_params, y)
        dtype = y.dtype
        device = y.device
        B, C, H, W = y.size()
        mask_0, mask_1 = self.get_mask_2x(B, C, H, W, dtype, device)

        # First mask processing
        scales_0 = scales * mask_0
        means_0 = means * mask_0
        y_res_0 = (y - means_0) * mask_0
        y_q_0 = self.quantize(y_res_0, method="ste")
        y_hat_0 = y_q_0 + means_0

        # Calculate bits for first mask
        bits_0 = self.get_y_gaussian_bits(y_q_0, scales_0)

        # Second mask processing with spatial prior
        cat_params = torch.cat((y_hat_0, common_params), dim=1)
        scales_1, means_1 = y_spatial_prior(cat_params).chunk(2, 1)
        scales_1 = scales_1 * mask_1
        means_1 = means_1 * mask_1
        y_res_1 = (y - means_1) * mask_1
        y_q_1 = self.quantize(y_res_1, method="ste")
        y_hat_1 = y_q_1 + means_1

        # Calculate bits for second mask
        bits_1 = self.get_y_gaussian_bits(y_q_1, scales_1)

        # Combine the results
        y_hat = add_and_multiply(y_hat_0, y_hat_1, q_dec)
        bits = bits_0 + bits_1

        return y_hat, bits

    def separate_prior_for_video_encoding_diff(self, params, y):
        """
        Differentiable version of separate_prior_for_video_encoding.
        Uses STE for quantization operations.

        Args:
            params: Model parameters containing quantization step, scales, and means
            y: Input tensor

        Returns:
            y: Quantized input
            q_dec: Dequantization factor
            scales: Scale parameters
            means: Mean parameters
        """
        # Extract parameters
        q_dec, scales, means = params.chunk(3, 1)

        # Ensure minimum quantization step size while maintaining differentiability
        q_dec = torch.clamp_min(q_dec, 0.5)

        # Apply quantization using reciprocal of q_dec
        q_enc = 1.0 / q_dec
        y = y * q_enc

        return y, q_dec, scales, means