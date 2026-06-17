import os
import sys

import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.rbnn_flow.models.models import (
    compute_feature_size,
    ResidualBlock,
    DownsampleBlock,
    UpsampleBlock,
    CNNEncoder,
    CNNDecoder,
)


# ── compute_feature_size ──────────────────────────────────────────────────────

class TestComputeFeatureSize:

    def test_zero_layers_returns_d_in_unchanged(self):
        assert compute_feature_size(64, 0) == 64

    def test_returns_int(self):
        assert isinstance(compute_feature_size(64, 2), int)

    def test_single_downsample_layer(self):
        # floor((64 + 2*1 - 3) / 2) + 1 = 32
        assert compute_feature_size(64, 1, downsample=True, pad=1, k_size=3, stride=2) == 32

    def test_two_downsample_layers(self):
        # 64 -> 32 -> 16
        assert compute_feature_size(64, 2, downsample=True, pad=1, k_size=3, stride=2) == 16

    def test_single_upsample_layer(self):
        # (8 - 1) * 2 - 2*1 + 3 = 15
        assert compute_feature_size(8, 1, downsample=False, pad=1, k_size=3, stride=2) == 15

    def test_two_upsample_layers(self):
        # 8 -> 15 -> 29
        assert compute_feature_size(8, 2, downsample=False, pad=1, k_size=3, stride=2) == 29

    def test_non_power_of_two_input(self):
        # floor((60 + 2 - 3) / 2) + 1 = 30
        assert compute_feature_size(60, 1, downsample=True, pad=1, k_size=3, stride=2) == 30


# ── ResidualBlock ─────────────────────────────────────────────────────────────

class TestResidualBlock:

    def _block(self, channels=16, dropout=0.0):
        return ResidualBlock(channels=channels, dropout=dropout).eval()

    def test_output_shape_matches_input(self):
        block = self._block()
        x = torch.randn(4, 16, 32, 32)
        assert block(x).shape == x.shape

    def test_output_is_tensor(self):
        block = self._block()
        x = torch.randn(2, 16, 8, 8)
        assert isinstance(block(x), torch.Tensor)

    def test_batch_size_one(self):
        block = self._block()
        x = torch.randn(1, 16, 8, 8)
        assert block(x).shape == (1, 16, 8, 8)

    def test_different_channel_count(self):
        block = self._block(channels=32)
        x = torch.randn(2, 32, 16, 16)
        assert block(x).shape == (2, 32, 16, 16)

    def test_non_square_spatial_dims(self):
        block = self._block(channels=8)
        x = torch.randn(2, 8, 12, 20)
        assert block(x).shape == (2, 8, 12, 20)

    def test_gradient_flows(self):
        block = ResidualBlock(channels=8, dropout=0.0)
        x = torch.randn(2, 8, 4, 4, requires_grad=True)
        block(x).sum().backward()
        assert x.grad is not None


# ── DownsampleBlock ───────────────────────────────────────────────────────────

class TestDownsampleBlock:

    def test_output_channels(self):
        block = DownsampleBlock(c_in=3, c_out=16, kernel_size=3)
        x = torch.randn(2, 3, 32, 32)
        assert block(x).shape[1] == 16

    def test_spatial_dims_with_stride2(self):
        # floor((32 + 2*1 - 3) / 2) + 1 = 16
        block = DownsampleBlock(c_in=3, c_out=8, kernel_size=3, stride=2, padding=1)
        x = torch.randn(2, 3, 32, 32)
        assert block(x).shape == (2, 8, 16, 16)

    def test_output_is_tensor(self):
        block = DownsampleBlock(c_in=1, c_out=4, kernel_size=3)
        x = torch.randn(1, 1, 16, 16)
        assert isinstance(block(x), torch.Tensor)

    def test_batch_dimension_preserved(self):
        block = DownsampleBlock(c_in=3, c_out=8, kernel_size=3)
        x = torch.randn(5, 3, 32, 32)
        assert block(x).shape[0] == 5

    def test_single_channel_in_multichannel_out(self):
        block = DownsampleBlock(c_in=1, c_out=32, kernel_size=3)
        x = torch.randn(2, 1, 16, 16)
        assert block(x).shape[1] == 32

    def test_gradient_flows(self):
        block = DownsampleBlock(c_in=4, c_out=8, kernel_size=3)
        x = torch.randn(2, 4, 16, 16, requires_grad=True)
        block(x).sum().backward()
        assert x.grad is not None


# ── UpsampleBlock ─────────────────────────────────────────────────────────────

class TestUpsampleBlock:

    def test_output_channels(self):
        # ConvTranspose2d k=4, stride=2, pad=1: (16-1)*2 - 2 + 4 = 32
        block = UpsampleBlock(c_in=8, c_out=4, kernel_size=4, stride=2, padding=1)
        x = torch.randn(2, 8, 16, 16)
        assert block(x).shape[1] == 4

    def test_spatial_dims_doubled_with_k4(self):
        # (16 - 1) * 2 - 2*1 + 4 = 32
        block = UpsampleBlock(c_in=8, c_out=4, kernel_size=4, stride=2, padding=1)
        x = torch.randn(2, 8, 16, 16)
        assert block(x).shape == (2, 4, 32, 32)

    def test_output_is_tensor(self):
        block = UpsampleBlock(c_in=4, c_out=2, kernel_size=4, stride=2, padding=1)
        x = torch.randn(1, 4, 8, 8)
        assert isinstance(block(x), torch.Tensor)

    def test_batch_dimension_preserved(self):
        block = UpsampleBlock(c_in=4, c_out=2, kernel_size=4, stride=2, padding=1)
        x = torch.randn(3, 4, 8, 8)
        assert block(x).shape[0] == 3

    def test_spatial_dims_with_k3(self):
        # (16 - 1) * 2 - 2*1 + 3 = 31
        block = UpsampleBlock(c_in=8, c_out=4, kernel_size=3, stride=2, padding=1)
        x = torch.randn(2, 8, 16, 16)
        assert block(x).shape[2:] == (31, 31)

    def test_gradient_flows(self):
        block = UpsampleBlock(c_in=4, c_out=2, kernel_size=4, stride=2, padding=1)
        x = torch.randn(2, 4, 8, 8, requires_grad=True)
        block(x).sum().backward()
        assert x.grad is not None


# ── CNNEncoder ───────────────────────────────────────────────────────────────

class TestCNNEncoder:
    # base_channels=4, num_layers=2, image_size=32:
    #   c_list = [4, 8, 16]
    #   initial_downs: 32 -> 16 (k=7, stride=2, pad=3)
    #   2x DownsampleBlock + ResidualBlock pairs: 16->8->4
    #   fm_size=4, in_dim=16*16=256, _out_channels=16

    def _enc(self, **kwargs):
        defaults = dict(in_channels=3, base_channels=4, d_model=16,
                        kernel_size=3, num_layers=2, dropout=0.0,
                        has_residuals=True, image_size=32)
        defaults.update(kwargs)
        return CNNEncoder(**defaults).eval()

    def test_instantiates(self):
        assert isinstance(self._enc(), CNNEncoder)

    def test_output_shape(self):
        enc = self._enc()
        x = torch.randn(2, 3, 32, 32)
        assert enc(x).shape == (2, 16)

    def test_d_model_attribute(self):
        assert self._enc(d_model=32)._d_model == 32

    def test_out_channels_attribute(self):
        # base_channels * 2^num_layers = 4 * 4 = 16
        assert self._enc()._out_channels == 16

    def test_no_residuals_output_shape(self):
        enc = self._enc(has_residuals=False)
        x = torch.randn(2, 3, 32, 32)
        assert enc(x).shape == (2, 16)

    def test_batch_dimension_preserved(self):
        enc = self._enc()
        B = 5
        x = torch.randn(B, 3, 32, 32)
        assert enc(x).shape[0] == B

    def test_single_channel_input(self):
        enc = self._enc(in_channels=1)
        x = torch.randn(2, 1, 32, 32)
        assert enc(x).shape == (2, 16)

    def test_gradient_flows(self):
        enc = CNNEncoder(in_channels=3, base_channels=4, d_model=16,
                         kernel_size=3, num_layers=2, dropout=0.0,
                         has_residuals=True, image_size=32)
        x = torch.randn(2, 3, 32, 32, requires_grad=True)
        enc(x).sum().backward()
        assert x.grad is not None


# ── CNNDecoder ────────────────────────────────────────────────────────────────

class TestCNNDecoder:
    # base_channels=8, d_model=32, kernel_size=4, num_layers=2:
    #   c_list = [8,16,32] → reversed: [32,16,8]
    #   input_proj: Conv2d(32, 32, k=1)
    #   UpsampleBlock(32,16,k=4) + ResidualBlock(16)
    #   UpsampleBlock(16, 8,k=4) + ResidualBlock(8)
    #   head: Conv2d(8, 3, k=1)
    #   Each upsample: H_out = (H-1)*2 - 2 + 4 = 2H  →  8×8 in → 32×32 out

    def _dec(self, **kwargs):
        defaults = dict(base_channels=8, d_model=32, kernel_size=4,
                        num_layers=2, dropout=0.0, has_residuals=True)
        defaults.update(kwargs)
        return CNNDecoder(**defaults).eval()

    def test_instantiates(self):
        assert isinstance(self._dec(), CNNDecoder)

    def test_output_is_tensor(self):
        dec = self._dec()
        x = torch.randn(2, 32, 8, 8)
        assert isinstance(dec(x), torch.Tensor)

    def test_output_shape(self):
        # (B, d_model, 8, 8) → (B, 3, 32, 32) with num_layers=2
        dec = self._dec()
        x = torch.randn(2, 32, 8, 8)
        assert dec(x).shape == (2, 3, 32, 32)

    def test_output_channels_always_3(self):
        dec = self._dec(base_channels=16, d_model=64)
        x = torch.randn(2, 64, 8, 8)
        assert dec(x).shape[1] == 3

    def test_spatial_dims_doubled_per_layer(self):
        # num_layers=2 with k=4,s=2,p=1: each block exactly doubles H and W
        dec = self._dec()
        x = torch.randn(2, 32, 8, 8)
        out = dec(x)
        assert out.shape[2] == 8 * (2 ** 2) and out.shape[3] == 8 * (2 ** 2)

    def test_batch_dimension_preserved(self):
        dec = self._dec()
        B = 5
        x = torch.randn(B, 32, 4, 4)
        assert dec(x).shape[0] == B

    def test_no_residuals_output_shape(self):
        dec = self._dec(has_residuals=False)
        x = torch.randn(2, 32, 8, 8)
        assert dec(x).shape == (2, 3, 32, 32)

    def test_num_layers_3_output_shape(self):
        # c_list = [8,16,32,64] → reversed [64,32,16,8]; input_proj Conv2d(32,64,k=1)
        # 3 upsample blocks: 4×4 → 8×8 → 16×16 → 32×32
        dec = self._dec(num_layers=3)
        x = torch.randn(2, 32, 4, 4)
        assert dec(x).shape == (2, 3, 32, 32)

    def test_different_base_channels_output_shape(self):
        # base_channels=16, c_list=[16,32,64] reversed → [64,32,16]; input_proj Conv2d(32,64,k=1)
        dec = self._dec(base_channels=16)
        x = torch.randn(2, 32, 8, 8)
        assert dec(x).shape == (2, 3, 32, 32)

    def test_gradient_flows(self):
        dec = CNNDecoder(base_channels=8, d_model=32, kernel_size=4,
                         num_layers=2, dropout=0.0, has_residuals=True)
        x = torch.randn(2, 32, 8, 8, requires_grad=True)
        dec(x).sum().backward()
        assert x.grad is not None


# ── CNNAutoencoder ────────────────────────────────────────────────────────────

class TestCNNAutoencoder:
    # CNNEncoder: (B, C, H, W) → (B, d_model)
    # Reshape:    (B, d_model) → (B, d_model, 1, 1)   [unsqueeze spatial dims]
    # CNNDecoder: (B, d_model, 1, 1) → (B, 3, 2^num_layers, 2^num_layers)
    #
    # Shared config: in_channels=3, base_channels=8, d_model=32,
    #                enc kernel_size=3, dec kernel_size=4, num_layers=2,
    #                image_size=32

    def _ae(self, **kwargs):
        cfg = dict(in_channels=3, base_channels=8, d_model=32,
                   enc_kernel=3, dec_kernel=4, num_layers=2,
                   dropout=0.0, has_residuals=True, image_size=32)
        cfg.update(kwargs)
        enc = CNNEncoder(
            in_channels=cfg['in_channels'],
            base_channels=cfg['base_channels'],
            d_model=cfg['d_model'],
            kernel_size=cfg['enc_kernel'],
            num_layers=cfg['num_layers'],
            dropout=cfg['dropout'],
            has_residuals=cfg['has_residuals'],
            image_size=cfg['image_size'],
        ).eval()
        dec = CNNDecoder(
            base_channels=cfg['base_channels'],
            d_model=cfg['d_model'],
            kernel_size=cfg['dec_kernel'],
            num_layers=cfg['num_layers'],
            dropout=cfg['dropout'],
            has_residuals=cfg['has_residuals'],
        ).eval()
        return enc, dec

    def _forward(self, enc, dec, x):
        z = enc(x)                              # (B, d_model)
        z = z.unsqueeze(-1).unsqueeze(-1)       # (B, d_model, 1, 1)
        return dec(z)

    def test_encoder_output_shape(self):
        enc, _ = self._ae()
        x = torch.randn(2, 3, 32, 32)
        assert enc(x).shape == (2, 32)

    def test_decoder_from_encoder_output_shape(self):
        # Decoder starting from 1×1: after num_layers=2 upsample blocks → 4×4
        enc, dec = self._ae()
        x = torch.randn(2, 3, 32, 32)
        out = self._forward(enc, dec, x)
        assert out.shape == (2, 3, 4, 4)

    def test_output_channels_always_3(self):
        enc, dec = self._ae()
        x = torch.randn(2, 3, 32, 32)
        assert self._forward(enc, dec, x).shape[1] == 3

    def test_batch_dimension_preserved(self):
        enc, dec = self._ae()
        B = 4
        x = torch.randn(B, 3, 32, 32)
        assert self._forward(enc, dec, x).shape[0] == B

    def test_no_residuals_output_shape(self):
        enc, dec = self._ae(has_residuals=False)
        x = torch.randn(2, 3, 32, 32)
        out = self._forward(enc, dec, x)
        assert out.shape == (2, 3, 4, 4)

    def test_gradient_flows_through_both(self):
        enc = CNNEncoder(in_channels=3, base_channels=8, d_model=32,
                         kernel_size=3, num_layers=2, dropout=0.0,
                         has_residuals=True, image_size=32)
        dec = CNNDecoder(base_channels=8, d_model=32, kernel_size=4,
                         num_layers=2, dropout=0.0, has_residuals=True)
        x = torch.randn(2, 3, 32, 32, requires_grad=True)
        z = enc(x).unsqueeze(-1).unsqueeze(-1)
        dec(z).sum().backward()
        assert x.grad is not None

    def test_deterministic_in_eval_mode(self):
        enc, dec = self._ae()
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            out1 = self._forward(enc, dec, x)
            out2 = self._forward(enc, dec, x)
        assert torch.allclose(out1, out2)

    def test_single_channel_input(self):
        enc, dec = self._ae(in_channels=1)
        x = torch.randn(2, 1, 32, 32)
        out = self._forward(enc, dec, x)
        assert out.shape == (2, 3, 4, 4)

    def test_num_layers_3_output_shape(self):
        # Decoder 1×1 with 3 upsample blocks → 8×8
        enc, dec = self._ae(num_layers=3)
        x = torch.randn(2, 3, 32, 32)
        out = self._forward(enc, dec, x)
        assert out.shape == (2, 3, 8, 8)
