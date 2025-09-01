from typing import Optional, Union

import torch
from torch import nn


class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    """

    def __init__(
        self,
        dim: int,
        window_size: tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        """Initialize the W-MSA module

        Args:
            dim: Number of input channels.
            window_size: The height and width of the window.
            num_heads: Number of attention heads.
            qkv_bias:  If True, add a learnable bias to query, key, value. Default: True
            qk_scale: Override default qk scale of head_dim ** -0.5 if set
            attn_drop: Dropout ratio of attention weight. Default: 0.0
            proj_drop: Dropout ratio of output. Default: 0.0
        """
        ########### START HERE, go through windowAttention then back to what calls it

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        breakpoint()
        # define a parameter table of relative position bias (2*win_h-1 * 2*win_w-1, num_heads)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w])) # (2, win_h, win_w) where 2 [h, w]
        coords_flatten = torch.flatten(coords, 1)  # (2, win_h * win_w)

        # create pairwise indices for every patches relative position;
        # computes all pairwise offsets for each pair of tokens (i, j), so for token pair
        # (i, j) we now have offsets at relative_coords[:, i, j] = (delta_y, delta_x)
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # (2, win_h * win_w, win_h * win_w) where 2 = (h, w)

        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # (win_h * win_w, win_h * win_w, 2)


        # shift to start from 0
        # shift the offsets so they're non-negative: go from [-(Wh-1) to +(Wh-1)].
        # to [0 … 2*Wh-2] after the shift
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1

        # encodes (Δy, Δx) into a single index by giving the y-offset a stride;
        # think of it like flattening a 2D index into a 1D array
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
    
        # Final result: a (win_h * win_w, win_h * win_w) matrix, where entry (i, j) 
        # is a unique integer ID for the relative position between tokens i and j.
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        # Intuition of the `relative_position_index` above
        #   1. Each relative position (Δy, Δx) maps to a unique integer index.
        #   2. Later, a learnable bias table of size (2*Wh-1 * 2*Ww-1, num_heads) is created.
        #   3. During attention, relative_position_index[i, j] selects the correct bias entry 
        #      to add to the attention logits for pair (i, j).

        ######## start heree ########
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}"

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class PatchMerging(nn.Module):
    # TODO: go through and understand/comment
    r"""Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage."""

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: Optional[bool] = True,
        qk_scale: Optional[float] = None,
        dropout: Optional[float] = 0.0,
        attn_drop: Optional[float] = 0.0,
        drop_path: Union[tuple[float], float] = 0.0,
        norm_layer: Optional[nn.Module] = nn.LayerNorm,
        downsample: Optional[nn.Module] = None,
        use_checkpoint: bool = False,
        fused_window_process: bool = False,
    ):
        """Intialize the basic swin transformer layer

        Args:
            dim: Number of input channels.
            input_resolution: Input resolution.
            depth: Number of blocks.
            num_heads: Number of attention heads.
            window_size: Local window size.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value. Default: True
            qk_scale: overrides the default qk scale of head_dim ** -0.5 if set; the division by sqrt(head_dim)
            drop: dropout rate
            attn_drop: Attention dropout rate
            drop_path: Stochastic depth rate; probability of ignoring the residual for the specific
                       blocks in the current layer; then length of the tuple should be the length
                       of the `depth`
            norm_layer: Normalization layer. Default: nn.LayerNorm
            downsample: Downsample layer (patch merging) at the end of the layer. Default: None
            use_checkpoint: Whether to use activation checkpointing to save memory. Default: False.
            fused_window_process: If True, use one kernel to fused window shift & window partition
                                  for acceleration, similar for the reversed part. Default: False

        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    dropout=dropout,
                    attn_drop=attn_drop,
                    drop_path=(
                        drop_path[i] if isinstance(drop_path, list) else drop_path
                    ),
                    norm_layer=norm_layer,
                    fused_window_process=fused_window_process,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, norm_layer=norm_layer
            )
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block"""

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        dropout: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        fused_window_process: bool = False,
    ):
        """
        Args:
            dim: Number of input channels.
            input_resolution: Input resolution.
            num_heads: Number of attention heads.
            window_size: Local window size.
            shift_size: TODO: flesh out more; Shift size for SW-MSA
                        (shifted window multiheaded self-attention).
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value. Default: True
            qk_scale: overrides the default qk scale of head_dim ** -0.5 if set; the division by sqrt(head_dim)
            dropout: dropout rate
            attn_drop: Attention dropout rate
            drop_path: Stochastic depth rate; probability of ignoring the residual for the specific
                       current block; only takes a single float since this is the residual we would
                       be skipping
            act_layer: Activation layer. Default: nn.GELU
            norm_layer: Normalization layer. Default: nn.LayerNorm
            fused_window_process: If True, use one kernel to fused window shift & window partition
                                  for acceleration, similar for the reversed part. Default: False
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert (
            0 <= self.shift_size < self.window_size
        ), "shift_size must in 0-window_size"

        # convert window size to tuple if int is provided
        win_size = (
            (self.window_size, self.window_size)
            if isinstance(self.window_size, int)
            else self.window_size
        )

        self.norm1 = norm_layer(dim)

        self.attn = WindowAttention(
            dim,
            window_size=win_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=dropout,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(
                img_mask, self.window_size
            )  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)
            ).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.fused_window_process = fused_window_process

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = torch.roll(
                    x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
                )
                # partition windows
                x_windows = window_partition(
                    shifted_x, self.window_size
                )  # nW*B, window_size, window_size, C
            else:
                x_windows = WindowProcess.apply(
                    x, B, H, W, C, -self.shift_size, self.window_size
                )
        else:
            shifted_x = x
            # partition windows
            x_windows = window_partition(
                shifted_x, self.window_size
            )  # nW*B, window_size, window_size, C

        x_windows = x_windows.view(
            -1, self.window_size * self.window_size, C
        )  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(
            x_windows, mask=self.attn_mask
        )  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = window_reverse(
                    attn_windows, self.window_size, H, W
                )  # B H' W' C
                x = torch.roll(
                    shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
                )
            else:
                x = WindowProcessReverse.apply(
                    attn_windows, B, H, W, C, self.shift_size, self.window_size
                )
        else:
            shifted_x = window_reverse(
                attn_windows, self.window_size, H, W
            )  # B H' W' C
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, "
            f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
        )

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchEmbed(nn.Module):
    """Splits an image into patches then embeds the patch through a Conv2d

    NOTE: this is different than ViT where ViT flattens the patch then embeds it through a linear layer
    """

    def __init__(
        self,
        img_size: Union[int, tuple] = 224,
        patch_size: Union[int, tuple] = 4,
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
        img_size: Union[int, tuple],
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
            depths: the number of swin transformer blocks per level (stage)
            dropout: dropout rate for the:
                        1. projection after attention
                        2. of the relative postions and TODO something w/ BasicLayer
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
            in_chs=in_chans,
            embed_dim=patch_emb_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches

        # tuple of number of patches across the height & width, respectively
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.pos_drop = nn.Dropout(dropout)

        # Stochastic depth; TODO comment more what this does
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # Build swin layers
        self.layers = nn.ModuleList()
        for layer_i in range(self.num_layers):  # loop through each level

            # layer dim doubles every layer i.e., [128, 256, 512, 1024]
            dim = int(patch_emb_dim * 2**layer_i)

            # layer spatial resolution halves both dimensions every layer; the initial PatchEmbed reduces by 1/4
            # for input (224, 224) num_patches = 56 i.e., [(56, 56), (28, 28), (14, 14), (7, 7)]
            input_res = (
                patches_resolution[0] // (2**layer_i),
                patches_resolution[1] // (2**layer_i),
            )

            # define the probability for each swin block for stochastic depth; this is the probability
            # of randomly skipping an entire residual branch in a block during training which acts as a
            # form of regularization; this means that in a residual connection [x + F(x)], the probability
            # that F(x) is skipped (multiplied by 0); the earlier layers have a lower probablity and the
            # later layers have a higher probability; each swin block will have a probability of skipping
            # described in Section A2.1 in the paper
            dpr = [
                x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
            ]  # stochastic depth decay rule

            assert len(dpr) == sum(depths)

            # extract the probablities for each block, for the current layer
            layer_dpr = dpr[sum(depths[:layer_i]) : sum(depths[: layer_i + 1])]

            # whether to downsample at the end of the stage; downsample after every stage but the last one
            # (similar to most feature extractors); TODO: verify this
            downsample = PatchMerging if (layer_i < self.num_layers - 1) else None

            layer = BasicLayer(
                dim=dim,
                input_resolution=input_res,
                depth=depths[layer_i],
                num_heads=num_heads[layer_i],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                dropout=dropout,
                attn_drop=attn_drop_rate,
                drop_path=layer_dpr,
                norm_layer=norm_layer,
                downsample=downsample,
                use_checkpoint=use_checkpoint,
                fused_window_process=fused_window_process,
            )
            self.layers.append(layer)

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


def build_swin(
    num_classes: int, img_size: Union[int, tuple], swin_params: dict[str, any]
):
    """Initalize the Swin Transformer model

    Args:
        num_classes: number of unique classes in the dataset ontology;
        swin_params: dictionary of parameters used to intialize the swin transformer with
    """

    layernorm = nn.LayerNorm

    swin_model = SwinTransformer(
        img_size=img_size,
        patch_size=swin_params["patch_size"],
        in_chans=3,  # for RGB images
        num_classes=num_classes,
        patch_emb_dim=swin_params["patch_emb_dim"],
        depths=swin_params["depths"],
        num_heads=swin_params["num_heads"],
        window_size=swin_params["window_size"],
        mlp_ratio=swin_params["mlp_ratio"],
        qkv_bias=True,
        qk_scale=None,
        dropout=swin_params["dropout"],
        attn_drop_rate=swin_params["attn_dropout"],
        drop_path_rate=swin_params["drop_path_rate"],
        norm_layer=layernorm,
        ape=swin_params["ape"],
        patch_norm=swin_params["patch_norm"],
        use_checkpoint=swin_params["use_activation_checkpointing"],
        fused_window_process=False,
    )
