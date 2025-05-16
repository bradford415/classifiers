from typing import Optional

from torch import nn


class PatchEmbed(nn.Module):
    """Splits an image into patches then embeds the patch through a Conv2d
    
    NOTE: this is different than ViT where ViT flattens the patch then embeds it through a linear layer
    """

    def __init__(
        self,
        img_size: int | tuple = 224,
        patch_size: int | tuple = 4,
        in_chs: int = 3,
        embed_dim: int = 96,
        norm_layer: Optional[nn.Module] = nn.LayerNorm,
    ):
        """Initialize the patch embedding module

        Args:
            img_size: input image size
            patch_size: the spatial size for each patch
            in_chs: number of input channels; for RGB use 3
            embed_dim: the embedding dimension to project the patch to; i.e., num output chs in the conv2d
            norm_layer: the type of normalization layer to apply after the patch is embedded; if None 
                        do not apply normalization
        """
        super().__init__()

        # convert to tuples if an int is provided
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size

        # number of patches across the height & width, respectively
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        ]

        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chs = in_chs
        self.embed_dim = embed_dim

        # Patch projection layer
        self.proj = nn.Conv2d(
            in_chs, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self):
        # ############# START HERE ########
        """TODO"""


class SwinTransformer(nn.Module):
    """Swin Transformer from the paper:
    `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 1000,
        patch_emb_dim: int = 96,
        depths: list[int] = [2, 2, 6, 2],
        num_heads: list[int] = [3, 6, 12, 24],
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale=None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: nn.Module = nn.LayerNorm,
        ape: bool = False,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        fused_window_process: bool = False,
        **kwargs  # TODO: remove
    ):
        """Initializes the SwinTransformer

        Args:
            img_size: the image tensor size input to the model; typically 224 for image
                      classification
            patch_size: the size of each image patch to split the input image into; e.g., 4 -> size 4x4xc
            patch_emb_dim: the dimension to project the image patches to
            patch_norm: if True, normalize patches after the embedding
            depths: the number of swin blocks per level
            mlp_ratio: TODO
        """
        super().__init__()

        self.num_layers = len(depths)
        self.patch_emb_dim = patch_emb_dim
        self.ape = ape
        self.patch_norm = patch_norm

        # TODO: comment
        self.num_features = int(patch_emb_dim * 2 ** (self.num_layers - 1))

        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
