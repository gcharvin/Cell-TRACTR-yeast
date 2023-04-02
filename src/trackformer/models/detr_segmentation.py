# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
This file provides the definition of the convolutional heads used
to predict masks, as well as the losses.
"""
import io
from collections import defaultdict
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

from ..util import box_ops
from ..util.misc import NestedTensor, interpolate, MLP

from .deformable_detr import DeformableDETR
from .detr import DETR
from .detr_tracking import DETRTrackingBase


class DETRSegmBase(nn.Module):
    def __init__(self, freeze_detr=False,return_intermediate_masks=False, mask_dim = 288):
        if freeze_detr:
            for param in self.parameters():
                param.requires_grad_(False)

        self.mask_dim = mask_dim
        self.mask_embed = MLP(self.hidden_dim, self.hidden_dim, self.mask_dim*2, 3)
        self.decoder_norm  = nn.LayerNorm(self.hidden_dim)
        self.return_intermediate_masks = return_intermediate_masks

    def forward_prediction_heads(self, output, mask_features):
        decoder_output = self.decoder_norm(output)
        mask_embed = self.mask_embed(decoder_output)

        outputs_mask_1 = torch.einsum("bqc,bchw->bqhw", mask_embed[:,:,:self.mask_dim], mask_features)
        outputs_mask_2 = torch.einsum("bqc,bchw->bqhw", mask_embed[:,:,self.mask_dim:], mask_features)

        outputs_mask = torch.stack((outputs_mask_1,outputs_mask_2),axis=2)

        return outputs_mask

    def forward(self, samples: NestedTensor, targets: list = None, track=False, prev_features=None, epoch=None):
        out, targets, features, memory, hs, mask_features, prev_out = super().forward(samples, targets,  track=track, prev_features=prev_features,epoch=epoch)

        pred_masks = self.forward_prediction_heads(hs[-1],mask_features)

        if self.return_intermediate_masks:
            intermediate_masks = []
            for i in range(len(hs) - 1):
                intermediate_masks.append(self.forward_prediction_heads(hs[i],mask_features))

        out["pred_masks"] = pred_masks

        if self.return_intermediate_masks:
            for i,intermediate_mask in intermediate_masks:
                out["aux_outputs"][i]['pred_masks'] = intermediate_mask

        return out, targets, features, memory, hs, mask_features, prev_out



# TODO: with meta classes
class DETRSegm(DETRSegmBase, DETR):
    def __init__(self, mask_kwargs, detr_kwargs):
        DETR.__init__(self, **detr_kwargs)
        DETRSegmBase.__init__(self, **mask_kwargs)


class DeformableDETRSegm(DETRSegmBase, DeformableDETR):
    def __init__(self, mask_kwargs, detr_kwargs):
        DeformableDETR.__init__(self, **detr_kwargs)
        DETRSegmBase.__init__(self, **mask_kwargs)


class DETRSegmTracking(DETRSegmBase, DETRTrackingBase, DETR):
    def __init__(self, mask_kwargs, tracking_kwargs, detr_kwargs):
        DETR.__init__(self, **detr_kwargs)
        DETRTrackingBase.__init__(self, **tracking_kwargs)
        DETRSegmBase.__init__(self, **mask_kwargs)


class DeformableDETRSegmTracking(DETRSegmBase, DETRTrackingBase, DeformableDETR):
    def __init__(self, mask_kwargs, tracking_kwargs, detr_kwargs):
        DeformableDETR.__init__(self, **detr_kwargs)
        DETRTrackingBase.__init__(self, **tracking_kwargs)
        DETRSegmBase.__init__(self, **mask_kwargs)