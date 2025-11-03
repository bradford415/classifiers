import math
from typing import Optional, Union

import torch
from torch import nn
from torch.nn import functional as F

from classifiers.models.swin import SwinTransformer


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
        # as the SwinTransformer patchification (1, 1, patch_emb_dim); this mask_token is
        # will be added to the masked patches that are zeroed out and used to essentially
        # learn what "nothingness" is, similar to how NLP uses a learnable [MASK] token to represent
        # nothing instead of using a vector of 0s; during the forward
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

        Returns:
            the downsampled feature map (b, c, h/32, w/32)
        """
        # embed image into flattened patches using a cnn (b, num_patches, patch_emb_dim)
        x = self.patch_embed(x)

        assert mask is not None

        # extract the patches shape (L = sequence length/number of patches)
        B, L, _ = x.shape

        # expand the single mask token to the batch size and number of patches; expand
        # only creates a view so the memory is not copied, this means every masked patch
        # points to the same underlying parameters as the singular mask token initially
        # created in __init__(), thus each token will have the same value; operations
        # on any masked patch will be accounted for in backprop and the single mask
        # tokens parameters will be updated accordingly
        # (I think autograd sums the gradients from each patch)
        # shape (b, num_patches, patch_embed_dim)
        mask_tokens = self.mask_token.expand(B, L, -1)

        # flatten the binary mask to (b, num_patches*num_patches, 1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)

        assert w.shape[1] == mask_tokens.shape[1]

        # zero out the patches to be maksed by multiplying by 0s and insert
        # the learnable mask_tokens in place of the masked patches; I think the idea
        # of this is that the model can learn what "nothingness" is rather than using
        # just a vector of 0s; this is how NLP tasks use a learnable [MASK] token to
        # hide tokens; one way I'm thinking about this is that images go through several
        # data transformations such as normalization, therefore 0 is not lower bound it's right
        # in the mideel, so the model doesn't really know `0` means nothing but learning this
        # mask token can; Section 3.2. in paper
        x = x * (1.0 - w) + mask_tokens * w

        # skipped by default
        if self.ape:
            x = x + self.absolute_pos_embed

        x = self.pos_drop(x)

        # propagate through each swin stage; for a 192x192 img the final output is (b, 36, 1024)
        # by 192 / 4 (patchification) -> 48 / 2 (stage 1) -> 24 / 2 (stage 2) -> 12 / 2 (stage 3)
        # -> 36 patches
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        # reshape the downsampled image back to spatial dims (b, c, h/32, w/32)
        x = x.transpose(1, 2)
        B, C, L = x.shape
        H = W = int(L**0.5)
        x = x.reshape(B, C, H, W)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return super().no_weight_decay() | {"mask_token"}


class SimMIM(nn.Module):
    """The SimMIM model which takes a modified Swin or ViT as the encoder"""

    def __init__(self, encoder: SwinTransformerSimMIM, encoder_stride: int):
        """Initialize the SimMIM model

        Args:
            encoder: a modified Swin or ViT encoder which masks the patches then
                     encodes them like a normal encoder
            encoder_stride: the downsample factor from the input image size to the
                            output feature map size; for Swin this typically 32
                            e.g., 192 / 4 (patchification) -> 48 / 2 (stage 1) -> 24 / 2 (stage 2)
                            -> 12 / 2 (stage 3) -> 6x6 feature map; therefore 192 / 6 = 32 or
                            4 * 2 * 2 * 2 = 32
        """
        super().__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride

        # Decoder:
        # Goal: predicts masked patches into the image space i.e., 1 patch -> 32x32x3 region of pixels
        # 1. Conv2d: Initialize a 1x1 cnn layer for each output patch of the encoder to decode
        #            each patch back into a region of pixels (e.g., 32x32x3); `num_features` is
        #            the final channel dim, while `out_channels` is essentially working backwards
        #            to compute the number of pixels an enocded patch represents - since the downsample
        #            factor is 32, 1 enoced patch basically represents a 32x32x3 pixel region; so
        #            this conv2d is going from a single embedding (1x1x1024) to a (32x32x3) patch region
        # 2. PixelShuffle: redistributes the channel dimension based on an upscale_factor to return to
        #                  a spatial dimension (b, c * r^2, h, w) -> (b, c, h * r, w * r);
        #                  e.g., for a single patch (b, 32*32*3, 1, 1) -> (b, 3, 32, 32) and since
        #                  the kernel size is 1x1 conv2d operates independently on all patches;
        #                  this is a non learnable layer which is essentially a reshape + permute
        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.encoder.num_features,
                out_channels=self.encoder_stride**2 * 3,
                kernel_size=1,
            ),
            nn.PixelShuffle(self.encoder_stride),
        )

        # set parameters for colorscale (RGB = 3) and the size of image patches
        # during patchification (typically 4)
        self.in_chans = self.encoder.in_chans
        self.patch_size = self.encoder.patch_size

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> float:
        """Forward pass through the full SimMIM model

        Args:
            x: a batch of input images after data augmentations (b, c, h, w)
            mask: a binary mask for which patches to mask (1 = mask, 0 = visible)
                  shape (b, h/4, w/4) where 4 = patch size

        Returns:
            the l1 loss between masked pixels only, averaged by the number of masked pixels
        """
        # encode the image with masked patches (b, num_features, h/32, w/32)
        z = self.encoder(x, mask)

        # reconstruct the pixels from all patches (even the non-masked patches but we'll compute the loss only on the masked patches)
        # (b, 3, orig_h, orig_w)
        x_rec = self.decoder(z)

        # repeat the binary mask along the h & w dims to create a mask
        # of the original image size (b, num_patches, num_patches) -> (b, 1, orig_h, orig_w)
        # NOTE: repeat_interleave repeats individual elements along a dimension while
        #       `repeat` repeats the entire tensors
        mask = (
            mask.repeat_interleave(self.patch_size, 1)
            .repeat_interleave(self.patch_size, 2)
            .unsqueeze(1)
            .contiguous()
        )

        # compute the elementwise l1 loss (|x - x_p|) between the original image (after data transforms)
        # and full reconstructed image (b, c, h, w); do not reduce so we can 0 out the pixels' loss that are
        # not masked so we compute the loss only only on the pixels that ARE masked
        loss_recon = F.l1_loss(x, x_rec, reduction="none")

        # zero out the loss at the pixel locations that are not masked so they do not contribute to
        # the loss, then average across the number of masked pixels in the batch
        # (self.in_chans is to account for each rgb channel)
        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans
        return loss, x_rec

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, "no_weight_decay"):
            return {"encoder." + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, "no_weight_decay_keywords"):
            return {"encoder." + i for i in self.encoder.no_weight_decay_keywords()}
        return {}


def build_swin_simmim(
    img_size: Union[int, tuple], encoder_stride: int, swin_params: dict[str, any]
):
    """Initalize the Swin Transformer model

    Args:
        img_size: the image input size (after data transforms)
        encoder_stride: the downsample factor from the input image size to the final stage
                        typically 32 for swin and 16 for vit
        swin_params: dictionary of parameters used to intialize the swin transformer with
    """

    layernorm = nn.LayerNorm

    swin_simmim_model = SwinTransformerSimMIM(
        img_size=img_size,
        patch_size=swin_params["patch_size"],
        in_chans=3,  # for RGB images
        num_classes=0,  # no classification head
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

    simmim_model = SimMIM(encoder=swin_simmim_model, encoder_stride=encoder_stride)

    return simmim_model


def build_swin(
    num_classes: int,
    img_size: Union[int, tuple],
    swin_params: dict[str, any],
    checkpoint_path: Optional[str],
):
    """Initalize the Swin Transformer model

    Args:
        num_classes: number of unique classes in the dataset ontology;
        swin_params: dictionary of parameters used to intialize the swin transformer with
    """

    layernorm = nn.LayerNorm

    if checkpoint_path is None:
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
