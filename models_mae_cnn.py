# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import timm

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed
import numpy as np
import segmentation_models_pytorch as smp

import torch.nn.functional as F


class MaskedAutoencoderCNN(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, model_arch='Unet', encoder_name='densenet121'):
        super().__init__()
        # --------------------------------------------------------------------------
        # MAE encoder specifics

        self.model = smp.__dict__[model_arch](
            encoder_name=encoder_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=3,                      # model output channels (number of classes in your dataset)
        )
        
        # self.model = timm.create_model(
        # model_name=encoder_name,
        # pretrained=True,
        # num_classes=3,
        # in_chans=3
        # )
        
        # self.model.fc = nn.Linear(self.model.fc.in_features, 10)
        # self.fc = nn.Linear(img_size, img_size)
        self.img_size = img_size
        self.patch_size = patch_size
        # self.model.head = nn.Conv2d(self.model.num_features, 3, kernel_size=1)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, imgs, mask_ratio, heatmaps=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        x = self.patchify(imgs)
        N, L, D = x.shape  # batch, length, dim

        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        x_masked = self.unpatchify(x * (1 - mask.unsqueeze(-1)))
        return x_masked, mask, ids_restore

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = imgs
        # print(target.shape,pred.shape)
        loss = (pred - target) ** 2
        loss = loss.mean()
        # loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75, heatmaps=None):
        imgs_masked, mask, ids_restore = self.random_masking(imgs, mask_ratio)
        # print("note:",imgs.shape,imgs_masked.shape)
        pred = self.model(imgs_masked)
        print(pred.shape, imgs_masked.shape)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


