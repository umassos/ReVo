# Adapted from VideoMAE: https://github.com/MCG-NJU/VideoMAE
# Wang et al., "VideoMAE: Masked Autoencoders are Data-Efficient Learners for
# Self-Supervised Video Pre-Training", NeurIPS 2022.
#
# Key architectural difference from the original VideoMAE:
#   The encoder here processes ALL patch tokens (both clean and corrupted) rather
#   than only visible (unmasked) tokens. This lets the model exploit full temporal
#   context — including degraded frames — to recover missing content from packet loss.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from functools import partial

from modeling_finetune import Block, _cfg, PatchEmbed, get_sinusoid_encoding_table
from timm.models.layers import trunc_normal_ as __call_trunc_normal_


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class PretrainVisionTransformerEncoder(nn.Module):
    """ViT encoder that embeds a full video clip into a sequence of patch tokens.

    Unlike vanilla VideoMAE, this encoder sees every patch in the clip — no token
    filtering is applied. The `mask` argument is accepted for API compatibility with
    the training pipeline but is not used to select tokens.
    """

    def __init__(self, img_size=224, num_frames=16, patch_size=16, in_chans=3,
                 num_classes=0, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 tubelet_size=2, use_checkpoint=False, use_learnable_pos_emb=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            img_size=img_size, num_frames=num_frames, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim, tubelet_size=tubelet_size)
        num_patches = self.patch_embed.num_patches
        self.use_checkpoint = use_checkpoint

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer, init_values=init_values)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, mask):
        # x: [B, C, T, H, W]  →  patch tokens: [B, N, embed_dim]
        x = self.patch_embed(x)
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()

        B, _, C = x.shape
        # All N tokens are forwarded through the encoder (no masking / token filtering)
        x = x.reshape(B, -1, C)

        if self.use_checkpoint:
            for blk in self.blocks:
                x = checkpoint.checkpoint(blk, x)
        else:
            for blk in self.blocks:
                x = blk(x)

        return self.norm(x)  # [B, N, embed_dim]

    def forward(self, x, mask):
        x = self.forward_features(x, mask)
        x = self.head(x)  # identity when num_classes == 0
        return x


class PretrainVisionTransformerDecoder(nn.Module):
    """Lightweight ViT decoder that predicts raw pixel values for each patch.

    The output head maps each token to a flat pixel vector of size
    (3 × tubelet_size × patch_size²), matching the patch unpatchify convention.
    """

    def __init__(self, patch_size=16, num_classes=768, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, init_values=None, num_patches=196,
                 tubelet_size=2, use_checkpoint=False):
        super().__init__()
        self.num_classes = num_classes
        # Sanity check: output dim must match the flattened tubelet pixel count
        assert num_classes == 3 * tubelet_size * patch_size ** 2
        self.num_features = self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.use_checkpoint = use_checkpoint

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer, init_values=init_values)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, return_token_num):
        if self.use_checkpoint:
            for blk in self.blocks:
                x = checkpoint.checkpoint(blk, x)
        else:
            for blk in self.blocks:
                x = blk(x)

        if return_token_num > 0:
            # Return predictions only for the last return_token_num tokens
            x = self.head(self.norm(x[:, -return_token_num:]))
        else:
            x = self.head(self.norm(x))

        return x  # [B, return_token_num, 3 * tubelet_size * patch_size²]


class PretrainVisionTransformer(nn.Module):
    """Full encoder-decoder ViT for video packet loss recovery.

    Given a clip of T frames (the last frame being corrupted by packet loss),
    the model predicts the pixel content of every spatial patch in the clip.
    At inference time only the patches corresponding to the last frame are
    used to reconstruct the corrupted image.

    Architecture overview:
        encoder          : deep ViT — embeds all T×H×W patch tokens
        encoder_to_decoder : linear projection encoder_embed_dim → decoder_embed_dim
        decoder          : shallow ViT — refines tokens and regresses pixel values
    """

    def __init__(self,
                 img_size=224,
                 num_frames=16,
                 patch_size=16,
                 encoder_in_chans=3,
                 encoder_num_classes=0,
                 encoder_embed_dim=768,
                 encoder_depth=12,
                 encoder_num_heads=12,
                 decoder_num_classes=1536,
                 decoder_embed_dim=512,
                 decoder_depth=8,
                 decoder_num_heads=8,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 use_checkpoint=False,
                 tubelet_size=2,
                 num_classes=0,   # unused; kept to avoid errors from timm's create_fn
                 in_chans=0,      # unused; kept to avoid errors from timm's create_fn
                 **kwargs):
        super().__init__()
        # Output dimension: 3 channels × tubelet_size frames × patch_size² pixels
        decoder_num_classes = 3 * tubelet_size * patch_size ** 2

        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size,
            num_frames=num_frames,
            patch_size=patch_size,
            in_chans=encoder_in_chans,
            num_classes=encoder_num_classes,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_checkpoint=use_checkpoint,
            use_learnable_pos_emb=use_learnable_pos_emb)

        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size,
            num_patches=self.encoder.patch_embed.num_patches,
            num_classes=decoder_num_classes,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_checkpoint=use_checkpoint)

        # Projects encoder tokens into the (typically smaller) decoder embedding space
        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # Fixed positional embeddings for the decoder covering all N patch positions
        self.pos_embed = get_sinusoid_encoding_table(
            self.encoder.patch_embed.num_patches, decoder_embed_dim)

        trunc_normal_(self.mask_token, std=.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.encoder.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, x, mask):
        # x:    [B, 3, T, H, W]  — clip (clean context frames + corrupted last frame)
        # mask: [B, N]            — 0 = clean patch, >0 = corrupted (API-compat, not used for filtering)

        # Encode all patches and project into the decoder's embedding space
        x_vis = self.encoder(x, mask)           # [B, N, encoder_embed_dim]
        x_vis = self.encoder_to_decoder(x_vis)  # [B, N, decoder_embed_dim]

        B, N, C = x_vis.shape
        expand_pos_embed = (
            self.pos_embed
            .expand(B, -1, -1)
            .type_as(x)
            .to(x.device)
            .clone()
            .detach()
        )  # [B, N, decoder_embed_dim]

        # Add positional encoding and decode all tokens
        x_full = x_vis + expand_pos_embed              # [B, N, decoder_embed_dim]
        x = self.decoder(x_full, expand_pos_embed.shape[1])

        # Output: [B, N, 3 * tubelet_size * patch_size²]
        return x
