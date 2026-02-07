import torch
import torch.nn as nn
import numpy as np
from timm.models.vision_transformer import Attention, Mlp

from .commons import *

class PatchEmbedder(nn.Module):
    def __init__(self, input_size, patch_size, hidden_size):
        super().__init__()
        self.patch_size = patch_size
        self.mlp = nn.Sequential(
            nn.Linear(patch_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, x):
        B, L = x.shape
        assert L % self.patch_size == 0, f"Latent size {L} not divisible by patch size {self.patch_size}"
        x = x.reshape(B, int(L / self.patch_size), self.patch_size) # B, T, P
        x = self.mlp(x) # B, T, D
        return x
    
class GiTBlock(nn.Module):
    """
    GiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class GiT(nn.Module):
    """
    Gene Diffusion Transformer (GiT).
    """
    def __init__(
        self,
        latent_size,
        patch_size,
        hidden_size,
        depth,
        num_heads,
        mlp_ratio,
        class_dropout_prob,
        num_classes,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.patch_size = patch_size
        self.num_heads = num_heads
        
        assert latent_size % patch_size == 0
        
        self.x_embedder = PatchEmbedder(latent_size, patch_size, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        self.num_patches = int(latent_size / patch_size)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            GiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, 1)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.num_patches)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize embedders
        nn.init.normal_(self.x_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.x_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        return x.reshape(shape=(x.shape[0], x.shape[1] * x.shape[2]))

    def forward(self, x, t, y):
        x = self.x_embedder(x) + self.pos_embed
        t = self.t_embedder(t)
        y = self.y_embedder(y, self.training)
        c = t + y 
        
        for block in self.blocks:
            x = block(x, c)
        
        x = self.final_layer(x, c) 
        x = self.unpatchify(x)
        
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass with Classifier-Free Guidance.
        Inputs x, t, y are expected to be doubled (2N) for conditional/unconditional split.
        """
        model_out = self.forward(x, t, y)
        
        # Split output (2N -> N, N)
        eps_cond, eps_uncond = torch.split(model_out, len(model_out) // 2, dim=0)
        
        # Apply guidance to all dimensions
        guided_eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
        
        # Reconstruct batch size
        return torch.cat([guided_eps, guided_eps], dim=0)