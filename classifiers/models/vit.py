from typing import Optional, Union

import torch
from torch import nn
from torch.nn import functional as F


class MultiheadAttention(nn.Module):
    """Multiheaded attention module used in ViT.

    This is pretty much the same as the original MHA except the layernorm
    is at the start of the module rather than after the residual connection
    """

    def __init__(self, input_dim, embed_dim, num_heads):
        """TODO

        NOTE: some MHA implementations allow you to specify the input dimensions to qkv and have
        another parameter for full dimension of attention i.e., the qkv projection dim

        Args:
            input_dim: input dim size for the queries, keys, and values; this is also
                       the final output dim size after the linear projection
            embed_dim: Total dimension of the model; embed_dim will be split across
                       num_heads (embed_dim // num_heads)  after it's projected
            num_heads: Number of attention heads; each head will have dimension of attention_dim // num_heads

        Returns:
            Linear projected attention values (batch, seq_len, embed_dim)
        """
        super().__init__()
        assert (
            embed_dim % num_heads == 0
        ), "The number of heads should be divisble by the attenion_dim"

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # NOTE: the original transformer uses layernorm after each residual
        #       but ViT uses layernorm right before MHA and the MLP
        self.norm = nn.LayerNorm(embed_dim)

        self.q_proj = nn.Linear(input_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(input_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(input_dim, embed_dim, bias=False)

        self.attention = Attention()

        self.linear_proj = nn.Linear(embed_dim, input_dim, bias=False)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor):
        """Forward pass through Multiheaded Attention;
        for self-attention the queries, keys, and values should be the same

        Args:
            queries: Input tensor to compute the attention of
            keys: Input tensor to compute the attention of
            values: Input tensor to compute the context of; for self-attention this should be the same
                    as q & v
        """
        # Linearly project q, k, & v (batch, seq_len, embed_dim)
        queries = self.q_proj(queries)
        keys = self.k_proj(keys)
        values = self.v_proj(values)

        # Split into heads (batch, num_heads, seq_len, head_dim)
        query_heads = queries.view(
            queries.shape[0], queries.shape[1], self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_heads = keys.view(
            keys.shape[0], keys.shape[1], self.num_heads, self.head_dim
        ).transpose(1, 2)
        value_heads = values.view(
            values.shape[0], values.shape[1], self.num_heads, self.head_dim
        ).transpose(1, 2)

        # Compute attention on all heads
        attention = self.attention(query_heads, key_heads, value_heads)

        # Combine all the heads together (concatenation step);
        # (b, heads, seq, head_dim) -> (b, seq, heads, head_dim) -> (b, seq, heads*head_dim)
        attention = (
            attention.transpose(1, 2)
            .contiguous()
            .view(attention.shape[0], -1, self.num_heads * self.head_dim)
        )

        # Final linear projection of MHA
        attention_proj = self.linear_proj(attention)

        return attention_proj


class Attention(nn.Module):
    """Scaled dot product attention with an optional mask

    Typically this is used in MHA where q, k, v have already been linearly
    projected and split into multiple heads; this can be used for sequences
    and feature maps (h, w) but the feature maps features (pixels) should be
    flattened
    """

    def __init__(self):
        """Initialize attention module"""
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute attention on q, k, & v

        q, k, v should be a minimum of 3 dimensions such as (batch, seq_len, embed_dim)
        or (batch, num_heads, seq_len, head_dim) for mha; attention will be computed on the last 2 dims

        Args:
            queries: Input tensor to compute the attention of
            keys: Input tensor to compute the attention of
            values: Input tensor to compute the context of; for self-attention this should be the same
                    as q & v
            mask: Optional tensor containing indices to be masked; typically used in decoders for NLP

        Returns:
           The context vectors (batch_size, seq_len, d_model)
        """
        # Used to scale the qk dot product
        sqrt_dim = torch.sqrt(torch.tensor(k.shape[-1]))

        # (batch_size, num_heads, q_len, k_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / sqrt_dim

        # Mask attention indices if mask is provided; softmax will set -inf to 0
        if mask is not None:
            scores.masked_fill_(mask.view(scores.size()), -float("Inf"))

        attention = F.softmax(scores, dim=-1)

        # Considered the context vectors because it's a weighted sum of the attention scores;
        # this gives a `context` value about an input's location
        context = torch.matmul(attention, v)  # (batch_size, num_heads, v_len, head_dim)
        return context


class FeedForward(nn.Module):
    """TODO: Comment this module"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    """Transformer encoder from the ViT paper"""

    def __init__(self, input_dim, depth, num_heads, embed_dim, mlp_dim):
        """Initialize the transformer module

        Args:
            input_dim: input embedding size; this is also the output size of MHA
            depth: number of MHA modules
            num_heads: number of heads in each MHA module
            embed_dim: Total dimension of the MHA; embed_dim will be split across
                       num_heads (embed_dim // num_heads)  after it's projected
            mlp_dim: dimension on the hidden layer in the MLP after each MHA

        """
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.layers = nn.ModuleList([])

        # Create a list of Transformer Encoders used in ViT
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        MultiheadAttention(
                            input_dim=input_dim,
                            embed_dim=embed_dim,
                            num_heads=num_heads,
                        ),
                        FeedForward(dim=input_dim, hidden_dim=mlp_dim),
                    ]
                )
            )

    def forward(self, x):
        """TODO"""
        # Sequentially loop through all encoders and add the residual after mha and ff
        for mha, ff in self.layers:
            x = mha(x, x, x) + x
            x = ff(x) + x

        return self.norm(x)


class ViT(nn.Module):
    """Standard vision transformer (ViT) from the paper:
    'An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale'
    """

    def __init__(
        self,
        image_size: Union[int, tuple[int, int]],
        patch_size: int,
        num_classes: int,
        patch_emb_dim: int,
        num_encoders: int,
        num_heads: int,
        emb_dim: int = 64,
        mlp_dim: int = 3072,
        pool: str = "cls",
        channels: int = 3,
        attention_dropout: float = 0.0,  # TODO: I should add this back in
        emb_dropout: float = 0.0,
    ):
        """Initialize the vision transformer

        Args:
            TODO
            image_size:
            patch_size:
            patch_emb_dim: input embedding size to the stack of transformer encoders; image patches
                       will be projected to this size before being passed to the encoders
            depth: number of transformer encoders to stack
            num_heads: number of heads in each MHA module
            emb_dim: total dimension of the MHA; embed_dim will be split across
                     num_heads (embed_dim // num_heads)  after it's projected
            mlp_dim: dimension on the hidden layer in the MLP after each MHA
            emb_dropout: dropout of the embedded patches right before being passed to the transformer encoder
        """
        super().__init__()
        # Extract input image and patch height and width
        image_height, image_width = (
            (image_size, image_size) if isinstance(image_size, int) else image_size
        )
        self.patch_height, self.patch_width = (
            (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        )

        assert (
            image_height % self.patch_height == 0
            and image_width % self.patch_width == 0
        ), "Image dimensions must be divisible by the patch size."
        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        # Compute the number of patches and their dimension size
        self.num_patches = (image_height // self.patch_height) * (
            image_width // self.patch_width
        )
        self.patch_dim = channels * self.patch_height * self.patch_width

        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, patch_emb_dim),
            nn.LayerNorm(patch_emb_dim),
        )

        # Learned positional embeddings for all patches and the cls token (1, n_p + 1, input_dim)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, patch_emb_dim)
        )

        # Extra classification token to be prepended onto the image patches;
        # this token will be the solely responsibile for the classification prediction
        self.cls_token = nn.Parameter(torch.randn(1, 1, patch_emb_dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer_encoder = Transformer(
            patch_emb_dim, num_encoders, num_heads, emb_dim, mlp_dim
        )

        # TODO: figure out if I need an nn.identity; for now leaving it out
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(patch_emb_dim, num_classes)
        self.pool = pool

    def forward(self, img):
        """TODO

        Args:
            TODO
        """
        b, c, h, w = img.shape

        # Convert batch of images to patchs
        # (b, c, h, w) -> (b, c, num_p, p_h, num_p, p_w)
        # -> (b, num_p, num_p, p_h, p_w, c) -> (b, num_p, p_h * p_w * c)
        x = (
            img.reshape(
                b,
                c,
                h // self.patch_height,
                self.patch_height,
                w // self.patch_width,
                self.patch_width,
            )
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(b, -1, self.patch_height * self.patch_width * c)
        )

        assert len(x.shape) == 3 and x.shape[-1] == self.patch_dim

        # Embed patches
        x = self.to_patch_embedding(x)

        # Copy the classification token along the batch (b, 1, input_dim)
        # and prepend it to the patches for each sample in the batch (b, num_p + 1, patch_dim)
        cls_tokens = self.cls_token.repeat(b, 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add learned positional embeddings along the batch
        x += self.pos_embedding[:, : self.num_patches + 1]  # TODO check the dim

        x = self.dropout(x)

        # Stack of ViT transformer encoders
        x = self.transformer_encoder(x)

        # Extract the extra cls token or use global average pooling to make the final
        # class prediction; I believe the class token is used most often
        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]
        
        # Classification prediction
        x = self.mlp_head(x)

        return x


# TODO put which version in function name
def vit_base(
    image_size: Union[int, tuple[int, int]],
    patch_size: int,
    num_classes: int,
    patch_emb_dim: int,
    num_encoders: int,
    num_heads: int = 12,
    emb_dim: int = 64,
    mlp_dim: int = 3072,
    pool: str = "cls",
    channels: int = 3,
    attention_dropout: float = 0.0,
    emb_dropout: float = 0.0,
):
    """Create the ViT-base classifier; see ViT() class for parameter descriptions"""
    return ViT(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        patch_emb_dim=patch_emb_dim,
        num_encoders=num_encoders,
        num_heads=num_heads,
        emb_dim=emb_dim,
        mlp_dim=mlp_dim,
        pool=pool,
        channels=channels,
        attention_dropout=attention_dropout,
        emb_dropout=emb_dropout,
    )
