from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from .hf_enc_dec import Encoder, Decoder
from .vq_modules import VectorQuantizer2 as VectorQuantizer
import torch.nn as nn

from typing import Tuple, Optional


class VQSub(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
            up_block_types: Tuple[str] = ("SDMUpDecoderBlock2D",),
            block_out_channels: Tuple[int] = (64,),
            layers_per_block: int = 1,
            act_fn: str = "silu",
            latent_channels: int = 3,
            sample_size: int = 32,
            num_vq_embeddings: int = 256,
            norm_num_groups: int = 32,
            vq_embed_dim: Optional[int] = None,
            scaling_factor: float = 0.18215,
            segmap_channels: int = 35,
            use_SPADE: bool = True
    ):
        super().__init__()

        # pass init params to Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=False,
        )

        vq_embed_dim = vq_embed_dim if vq_embed_dim is not None else latent_channels

        self.quant_conv = nn.Conv2d(latent_channels, vq_embed_dim, 1)
        self.quantize = VectorQuantizer(num_vq_embeddings, vq_embed_dim, beta=0.25, remap=None, sane_index_shape=False)
        self.post_quant_conv = nn.Conv2d(vq_embed_dim, latent_channels, 1)

        # pass init params to Decoder
        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,  # extend decoder spatial capability..
            segmap_channels=segmap_channels,
            use_SPADE=use_SPADE
        )

    # ------------------------------------------------------------------------------
    ## Main part..
    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant, segmap):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant, segmap)
        return dec

    def decode_code(self, code_b, segmap):
        quant_b = self.quantize.get_codebook_entry(code_b, code_b.shape)
        dec = self.decode(quant_b, segmap)
        return dec

    def forward(self, input, segmap):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant, segmap)
        return dec, diff

    def get_last_layer(self):
        return self.decoder.conv_out.weight

