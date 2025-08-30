from typing import Optional

import torch
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
            img_size: input image size (h, w)
            patch_size: the spatial size for each patch
            in_chs: number of input channels; for RGB use 3
            embed_dim: the embedding dimension to project the patch to; i.e., num output chs in the conv2d
            norm_layer: the type of normalization layer to apply after the patch is embedded; if None
                        do not apply normalization
        """
        super().__init__()

        # convert to tuples if an int is provided
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (
            (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        )

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

    def forward(self, x):
        """Convert image to patches embeddings

        Args:
            x: an input image (b, c, h, w)

        Returns:
            the patch embedded image (b, num_patches, patch_embed_dim)
            where num_patches = h//patch_size * w//patch_size
        """
        b, c, h, w = x.shape

        assert self.image_size == (h, w)

        # (b, patch_emb_dim, h_p, w_p) -> (b, patch_emb_dim, h_p*w_p)
        # -> (b, h_p*w_p, patch_emb_dim) where h_p = height // patch_size
        x = self.proj(x).flatten(2).transpose(1, 2)

        if self.norm is not None:
            x = self.norm(x)

        return x


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
        dropout: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: nn.Module = nn.LayerNorm,
        ape: bool = False,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        fused_window_process: bool = False,
    ):
        """Initializes the SwinTransformer

        Args:
            img_size: the image tensor size input to the model; typically 224 for image
                      classification
            patch_size: the size of each image patch to split the input image into; e.g., 4 -> size 4x4xc
            patch_emb_dim: the dimension to project the image patches to; patches are created with a Conv2D layer
                           so patch_emb_dim is the number of filters
            patch_norm: if True, normalize patches after the embedding
            depths: the number of swin blocks per level
            dropout: dropout rate of the relative postions and TODO something w/ BasicLayer
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

        # split image into non-overlapping patches (b, num_patches, patch_emb_dim)
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=patch_emb_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches

        # tuple of number of patches across the height & width, respectively
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.pos_drop == nn.Dropout(dropout)

        # Stochastic depth; TODO comment more what this does
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # Build swin layers
        self.layers = nn.ModuleList()
        for layer_i in range(self.num_layers):  # loop through each level

            # layer dim doubles every layer i.e., [128, 256, 512, 1024]
            dim = int(patch_emb_dim * 2**layer_i)

            # layer spatial resolution halves every layer
            # for input (224, 224) num_patches = 56 i.e., [(56, 56), (28, 28), (14, 14), (7, 7)]
            input_res = (
                (
                    patches_resolution[0] // (2**layer_i),
                    patches_resolution[1] // (2**layer_i),
                ),
            )

            breakpoint()
            ###### START HERE
            # TODO: understand and comment
            drop_path = dpr[sum(depths[:layer_i]) : sum(depths[: layer_i + 1])]

            layer = BasicLayer(
                dim=dim,
                # TODO understand this input resoution
                input_resolution=input_res,
                depth=depths[layer_i],
                num_heads=num_heads[layer_i],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=dropout,
                attn_drop=attn_drop_rate,
                drop_path=drop_path,
                norm_layer=norm_layer,
                downsample=PatchMerging if (layer_i < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                fused_window_process=fused_window_process,
            )
            self.layers.append(layer)


def build_swin(num_classes: int, swin_params: dict[str, any]):
    """Initalize the Swin Transformer model
    
    Args:
        num_classes: number of unique classes in the dataset ontology;
        swin_params: dictionary of parameters used to intialize the swin transformer with
    """

    layernorm = nn.LayerNorm

    swin_model = SwinTransformer(
        img_size=swin_params["img_size"],
        patch_size=swin_params["patch_size"],
        in_chans = 3, # for RGB images
        num_classes = num_classes,
        patch_emb_dim=swin_params["patch_emb_dim"],
        depths=swin_params["depths"], 
        num_heads=swin_params["num_heads"],
        window_size=swin_params["window_size"],
        mlp_ratio=swin_params["mlp_ratio"]
        qkv_bias = True,
        qk_scale=None,
        dropout=swin_params["dropout"],
        attn_drop_rate=swin_params["attn_drop"],
        drop_path_rate=swin_params["attn_drop_rate"],
        norm_layer=layernorm,
        ape=swin_params["ape"],
        patch_norm=swin_params["patch_norm"],
        use_checkpoint=swin_params["use_activation_checkpointing"],
        #### start here, run og swin transformer code and see if this is used
        fused_window_process: bool = False,
        )