import math
from typing import Optional, Union

import torch
from torch import nn


def window_partition(x: torch.Tensor, window_size: int):
    """Reshapes x to partition into non-overlapping windows

    Args:
        x: the mask for the shifted windows (B, H, W, C)
        window_size: window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape

    # (b, num_windows_h, window_size, num_windows_w, window_size, c)
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)

    # (b, num_windows_h, num_windows_w,, window_size,  window_size, c) ->
    # (b * num_windows_h * num_windows_w, window_size, window_size, c)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows


class MLP(nn.Module):
    """2-layer MLP for after each attention module"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        """ """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


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
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias (2*win_h-1 * 2*win_w-1, num_heads)
        # see classifiers/models/README.md for me info on this matrix
        # described in Section 3.2 Relative position bias - represents the B hat matrix
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(
            torch.meshgrid([coords_h, coords_w])
        )  # (2, win_h, win_w) where 2 [h, w]
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

        # Initialize the layers for qkv projection and the projection after attention
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # fill the relative position bias matrix from a normal distribution truncated between
        # [-2, 2] which helps avoid too extreme values
        trunc_normal_(
            self.relative_position_bias_table, mean=0, std=0.02, a=-2.0, b=2.0
        )
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
    """Patch Merging Layer to downsample at after each stage (except the last)

    Downsamples by 2x
    """

    def __init__(
        self,
        input_resolution: tuple[int, int],
        dim: int,
        norm_layer: Optional[nn.Module] = nn.LayerNorm,
    ):
        """Initalize the PatchMerging module

        Args:
            input_resolution: Resolution of input feature.
            dim: Number of input channels.
            norm_layer: Normalization layer.  Default: nn.LayerNorm
        """
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim

        # projects the 2x2 merged patches so we need input_dim =  4*dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """Merge patches and downsample

        Args:
            x: flattened input tokens (b, h*w, c)

        Returns:
            a downsample feature map of flattened patches with shape (b, h/2*w/2, 2*c)
            NOTE: the merge is 4*c but the projection at the end reduces the channel dim by 2
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        # reshape back to spatial dims
        x = x.view(B, H, W, C)

        # TODO verify this logic
        # extract all even patches from the height and width
        x0 = x[:, 0::2, 0::2, :]  # (b, h/2, w/2, c)

        # extract all odd patches from the height and even patches from the width
        x1 = x[:, 1::2, 0::2, :]  # (b, h/2, w/2, c)

        # extract all even patches from the height and odd patches from the width
        x2 = x[:, 0::2, 1::2, :]  # (b, h/2, w/2, c)

        # extract all odd patches from the height and odd patches from the width
        x3 = x[:, 1::2, 1::2, :]  # (b, h/2, w/2, c)

        # concatenate the the 4 patches of the 2x2 region along the channel dimension (downsamples by 2x)
        x = torch.cat([x0, x1, x2, x3], -1)  # (b, h/2, w/2, 4*c)

        # collapse the spatial dims to flatten the patches (b, h/2*w/2, 4*c)
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)

        # project the merged patches to (b, h/2*w/2, 2*c)
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

        # build list of the specified number of swin blocks `depth` for the current stage/level;
        # every other block will perform sw-msa insead of regular w-msa
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
                    drop_path_rate=(
                        drop_path[i] if isinstance(drop_path, list) else drop_path
                    ),
                    norm_layer=norm_layer,
                    fused_window_process=fused_window_process,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer - downsample at the end of the first 3 stages (not the last one)
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, norm_layer=norm_layer
            )
        else:
            self.downsample = None

    def forward(self, x):
        """Forward pass through the stack of TransformerBlocks for the current stage/level

        Args:
            x: flattened feature map of patches (b, num_patches, c)

        Returns:
            the final feature map of flattened patches (b, h/32 * w/32, c)
        """
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        # merge patches at the end of the stage to downsample; happens in every stage but the last
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
    """Swin Transformer Block which performs either W-MSA or SW-MSA; every block it alternates
    between the two
    """

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
        drop_path_rate: float = 0.0,
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
            shift_size: Shift size for sw-msa (shifted window multiheaded self-attention); this is
                        shift_size=0 for regular w-msa and  shift_size=window_size // 2 for sw-msa
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value. Default: True
            qk_scale: overrides the default qk scale of head_dim ** -0.5 if set; the division by sqrt(head_dim)
            dropout: dropout rate
            attn_drop: Attention dropout rate
            drop_path_prob: Stochastic depth rate; probability of ignoring the residual for the specific
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

        # Create the Windowed Multiheaded Self-Attention module (W-MSA)
        self.attn = WindowAttention(
            dim,
            window_size=win_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=dropout,
        )

        # Initialize the module which randomly drops residual blocks per sample (0s the tensors out like dropout does)
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

        self.drop_path_rate = drop_path_rate

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        # 2-layer MLP after the windowed attention module
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=dropout,
        )

        # Every other block the shift_size > 0 and sw-msa is used
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA; used to assign different integer IDs to
            # different regions of the image
            # this mask is mentioned in Figure 4, for efficient batch computation:
            #   1. essentially, when we cyclic shift the windows, we will have more windows
            #      (some smaller than MxM - see Fgiure 4) and some of the windows
            #      (the corners, top/bottom, and the left/right edges) will have wrap around
            #      and have patches that are not adjacent in the image
            #   2. Swin only wants to compute attention locally and have patches globally communicate
            #      through window shifting; therefore, to keep this efficient, we create a mask which
            #      for each region with uniuqe IDs `cnt` so later we can tell which tokens belong
            #      to the same masked group
            #   3. For efficient batch computation, we keep the same number of windos as before the
            #      shift but employ the masking mechanism so it's like there are additional windows
            #      after the shift
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # (1, h, w, 1)

            h_slices = (
                slice(0, -self.window_size),  # rows [0, h-window_size]
                slice(
                    -self.window_size, -self.shift_size
                ),  # rows[window_size + 1, h - shift_size]
                slice(-self.shift_size, None),  # rows[h - shift_size, h]
            )
            w_slices = (
                slice(0, -self.window_size),  # cols [0, w - window_size]
                slice(
                    -self.window_size, -self.shift_size
                ),  # cols [window_size + 1, w - shift_size]
                slice(-self.shift_size, None),  # cols [w - shift_size, w]
            )

            # assign a unique ID to each of the 9 regions; for a visual of this masking,
            # see classifiers/models/README.md `Efficient batch computation for shifted configuration`
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # partition the mask into windows (b*num_windows, window_size, window_size, 1)
            mask_windows = window_partition(
                img_mask, self.window_size
            )  # nW, window_size, window_size, 1

            # flatten the window masks (b*num_windows, window_size*window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)

            # subtracts region IDs between every pair of positions inside the same window
            # to get the pairwise difference
            #   1. if patches come from different regions, the
            #      difference will be non-zero and the patches should not attend
            #   2. patches in the same window that come from different regions are not
            #      spatially accurate so it does not make sense for them to attend
            #      since attention in swin is performed locally
            #   3. patches in the same window that come from the same region can attend,
            #      and patches within the same window may come from several different regions
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)

            # block attention between patches not in the same region
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)
            ).masked_fill(attn_mask == 0, float(0.0))
        else:
            # regular W-MSA i.e., no shifted windows
            attn_mask = None

        # save the mask as a model attribute
        self.register_buffer("attn_mask", attn_mask)

        self.fused_window_process = fused_window_process

    def forward(self, x):
        """Forward pass through the SwinTransformerBlock

        Args:
            x: flattened patches of the feature map (b, num_patches, c)

        Returns:
            the output of the swin transformer block (b, num_patches, c); same shape as input
        """
        H, W = self.input_resolution
        B, L, C = x.shape  # L = sequence length/number of patches
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)

        # reshape patches to spatial dims
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                # shift the patches along the spatial dims up and to the left
                shifted_x = torch.roll(
                    x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
                )

                # partition the shifted patches into windows (b * num_windows, win_size, win_size, c)
                x_windows = window_partition(
                    shifted_x, self.window_size
                )  # nW*B, window_size, window_size, C
            else:
                # only for fused window process
                x_windows = WindowProcess.apply(
                    x, B, H, W, C, -self.shift_size, self.window_size
                )
        else:
            shifted_x = x

            # partition patches into windows to shape (num_windows*b, window_size, window_size, c)
            x_windows = window_partition(
                shifted_x, self.window_size
            )  # nW*B, window_size, window_size, C

        # flatten windows (b * num_windows, win_size * win_size, c)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # perform W-MSA or SW-MSA on the flattened windows (b * num_windows, window_size * window_size, c)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # merge windows by reshaping back to spatial dims (b * num_windows, window_size, window_size, c)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # reverse cyclic shift; convert windows back to patches (b, h, w, c) and shift the feature map
        # back to its original spot (down and to the right)
        if self.shift_size > 0:
            if not self.fused_window_process:
                # convert windows back to patches (b, h, w, c)
                shifted_x = window_reverse(
                    attn_windows, self.window_size, H, W
                )  # B H' W' C

                # shift the patches along the spatial dims down and to the right
                x = torch.roll(
                    shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
                )
            else:
                x = WindowProcessReverse.apply(
                    attn_windows, B, H, W, C, self.shift_size, self.window_size
                )
        else:
            # convert windows back to patches (b, h, w, c)
            shifted_x = window_reverse(
                attn_windows, self.window_size, H, W
            )  # B H' W' C
            x = shifted_x

        # flatten patches (b, h * w, c)
        x = x.view(B, H * W, C)

        # add the identity (input x) with the attended outputs with samples randomly dropped;
        # i.e., samples in batches are randomly zeroed out based on the drop_path_rate
        x = shortcut + self.drop_path(x)

        # project x (identity + dropped attended outputs) with an MLP and add x before projection
        # NOTE: swin uses 2 skip connectsions, input -> msa and msa -> MLP output
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

        assert self.img_size == (h, w)

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

        # the channel dimension at the end of the last swin layer (patch_dim * (2^num_layers-1))
        # e.g., dim = 128 num_layers (stages) = 4, then num_features = 128 * 2 ^ 3 = 1024
        # the patch merging stacks 4 patches making the feature (channel) dim = 4 * dim but there's a
        # projection layer at the end of the patch merging which reduces the feature dim to 2 * dim;
        # and we only apply patch merging at [:num_layers-1] so the feature dim only gets amplified 3 times
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

        self.norm = norm_layer(self.num_features)

        # performs average pooling on the output feature map of the last stage
        # along the token dim for classification; using 1 is the same as GlobalAveragePooling;
        # this is pretty much the same as averaging the spatial dimensions like in a feature map,
        # so the resulting vector is (num_chs,)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # Linear head for final class logits
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initializes the following layers weights and biases
        - linear layers to a truncated normal distribution (std=0.02) and sets bias to 0
        - LayerNorms bias to 0 and weight to 1
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward_features(self, x: torch.Tensor):
        """Forward pass through the swin stages and global average pooling

        Args:
            x: the input image to be processed; dimensions are its original input size after
            transformations (i.e., resize to 224x224) (b, orig_c, orig_h, orig_w)
        """
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        # forward pass through each stage of swin; final output is the flattened feature map of patches
        # (b, h/16 * w/16, 4 * c)
        for layer in self.layers:
            x = layer(x)

        ##### start here
        # layer norm across the feature dim (channels); nn.LayerNorm always operates on the last dimension;
        # this means that for shape (b, num_tokens, c), it will normalize across c, for every token in each batch
        x = self.norm(x)  # B L C

        # average over the token dimension to get a global representation of each channel;
        # this is pretty much the same as averaging the spatial dimensions like in a feature map,
        # so the resulting vector is (c,)
        x = self.avgpool(x.transpose(1, 2))  # (b, c, 1)
        x = torch.flatten(x, 1)  # (b, c)
        return x

    def forward(self, x):
        """Forward pass through the swin transformer

        Returns:
            the classification logits for each sample in the batch (b, num_classes)
        """
        # forward pass through the feature extractor and global average pooling (b, c)
        x = self.forward_features(x)

        # single linear layer for class logits (b, num_classes)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += (
            self.num_features
            * self.patches_resolution[0]
            * self.patches_resolution[1]
            // (2**self.num_layers)
        )
        flops += self.num_features * self.num_classes
        return flops


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int):
    """Convert windows back to patches

    Args:
        windows: (num_windows * b, window_size, window_size, c)
        window_size: Window size
        H: Height of image (in patches)
        W: Width of image (in patches)

    Returns:
        x: (B, H, W, C) where H/W equal number of patches along that dimension
    """
    # compute the batch size by dividing the number of windows into the first dim
    B = int(windows.shape[0] / (H * W / window_size / window_size))

    # reshape to (b, num_windows_h, num_windows_w, window_size, window_size, c)
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )

    # reshape windows back to patches
    # (b, num_windows_h, win_size, num_wins_w,  win_size, c) -> (b, h, w, c)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    # return if not training or the probability of drop path is 0
    if drop_prob == 0.0 or not training:
        return x

    # compute probability of keeping the path
    keep_prob = 1 - drop_prob

    # (batch_size, 1, 1, ..., 1) so each sample in the batch gets its own random keep/drop decision.
    # Example: if x has shape [64, 128, 32, 32], then mask has shape [64, 1, 1, 1].
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets

    # choose 0 or 1 for each sample in the batch where 1 = keep the residual and 0 = drop the residual
    # by zeroing every thing out (x * random_tensor); e.g., if drop_prob = 0.2 then there's an 80% chance
    # the residual is kept, for each sample there will be a 20% chance that it is zero'd out effectively
    # skipping the residual for that sample
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)

    # Scaling by 1/keep_prob ensures the expected value of activations (any output of a layer)
    # is preserved (like Dropout).
    # If keep_prob = 0.8, then ~80% of samples keep the path.
    # Otherwise, the network’s activation magnitude would shrink.
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)

    # broadcast the sampled bernoulli values (1 or 0) across each sample; keeps or zeros out the
    # tensor value
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        """

        Args:
            scale_by_keep: Whether to rescale surviving paths to preserve the expected magnitude (like Dropout does)
        """
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


def trunc_normal_(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
):
    """
    Copied from: https://github.com/huggingface/pytorch-image-models/blob/cedba69c198455e35d7fad09155bffaae0b390cd/timm/layers/weight_init.py#L43

    Purpose:
        * Fill a tensor with values from a normal distribution with given mean and standard
          deviation (std), but truncated so that all values lie within [a, b].
        * This is commonly used in neural network weight initialization to avoid extreme values.

    Internally uses CDF → uniform → inverse CDF transform to efficiently generate truncated normal samples.

    Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    NOTE: this impl is similar to the PyTorch trunc_normal_, the bounds [a, b] are
    applied while sampling the normal with mean/std applied, therefore a, b args
    should be adjusted to match the range of mean, std args.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """

    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    with torch.no_grad():

        def norm_cdf(x):
            # Computes standard normal cumulative distribution function
            return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

        if (mean < a - 2 * std) or (mean > b + 2 * std):
            warnings.warn(
                "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                "The distribution of values may be incorrect.",
                stacklevel=2,
            )

        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)

        return tensor


class SwinTransformerSimMIM(SwinTransformer):
    """Swin Transformer for SimMIM

    Used to pretrain the SwinTransformer feature extractor using self-supervised learning;
    once trained, the SwinTransformer can be used to fine-tune on downstream tasks like object detection

    Paper: https://arxiv.org/pdf/2111.09886
    """

    def __init__(self, **swin_kwargs):
        """Initializes the SwinTransformer for SimMIM pretraining

        Args:
            swin_kwargs: keyword arguments for the SwinTransformer classification model;
                         see SwinTransformer().__init__() for the specifc arguments
        """
        # initalize the SwinTransformer
        super().__init__(**swin_kwargs)

        assert self.num_classes == 0

        # create a single masked_token with the same embedding dimension
        # as the SwinTransformer patchification (1, 1, patch_emb_dim); during the forward
        # pass, the token will be expanded to the batch size and number of patches
        # (b, num_patches, patch_emb_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.patch_emb_dim))
        trunc_normal_(self.mask_token, mean=0.0, std=0.02)

    def forward(self, x, mask):
        """Train the SwinTransformer using self-supervised learning (SimMIM)

        Args:
            x: the input image to be processed; dimensions are its original input size after
            transformations (e.g., resize to 224x224) (b, c, h, w)
            mask: a binary mask (1 = masked, 0 = visible) of shape (b, num_patches, num_patches) 
                   which is the shape of the patchified image
        """
        # embed image into flattened patches using a cnn (b, num_patches, patch_emb_dim)
        x = self.patch_embed(x)

        assert mask is not None

        # extract the patches shape (L = sequence length/number of patches)
        B, L, _ = x.shape

        # expand the single mask token to the batch size and number of patches; expand
        # only creates a view so the memory is not copied, this means every masked patch
        # points to the same underlying parameters as the singular mask token initially
        # created in __init__(); operations on any masked patch will be accounted for in
        # backprop and the single mask tokens parameters will be updated accordingly
        # (I think autograd sums the gradients from each patch)
        # shape (b, num_patches, patch_embed_dim)
        mask_tokens = self.mask_token.expand(B, L, -1)

        # flatten the binary mask to (b, num_patches*num_patches, 1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)
        
        assert w.shape[1] == mask_tokens.shape[1]
        
        
        ### start here and figure out what this does, then continue with forward method
        x = x * (1.0 - w) + mask_tokens * w

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        x = x.transpose(1, 2)
        B, C, L = x.shape
        H = W = int(L**0.5)
        x = x.reshape(B, C, H, W)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return super().no_weight_decay() | {"mask_token"}


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

    return swin_model
