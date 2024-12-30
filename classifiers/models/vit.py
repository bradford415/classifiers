from typing import Optional

import torch
from torch import nn



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

        self.linear_proj = nn.linear(embed_dim, input_dim, bias=False)

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
            values.shape, values.shape[1], self.num_heads, self.head_dim
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
    """TODO"""
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    

class Transformer(nn.Module):
    """TODO
    """

    def __init__(self, input_dim, depth, num_heads, embed_dim, mlp_dim):
        """Initalize the transformer module

        Args:
            input_dim: input embedding size; this is also the output size of MHA
            depth: number of MHA modules
            num_heads: number of heads in each MHA module
            embed_dim: Total dimension of the MHA; embed_dim will be split across
                       num_heads (embed_dim // num_heads)  after it's projected
            mlp_dim: TODO

        """
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.layers = nn.ModuleList([])

        # Create a list of Transformer Encoders used in ViT
        for _ in range(depth):
            self.layers.append(nn.ModuleList[
                MultiheadAttention(input_dim=input_dim, embed_dim=embed_dim, num_heads=num_heads),
                FeedForward(dim=input_dim, mlp_dim=mlp_dim)
                ])
    
    def forward(self, x):
        """TODO"""
        # Sequentially loop through all encoders and add the residual after mha and ff
        for mha, ff in self.layers:
            x = mha(x) + x
            x = ff(x) + x

        return self.norm(x)
        


class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, attention_dropout):
        super().__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.proj_dropout = nn.Dropout(attention_dropout)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor):
        queries = self.query(queries)
        keys = self.key(keys)
        values = self.value(values)
        ###### START HEre and compare with righ 

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights
    

