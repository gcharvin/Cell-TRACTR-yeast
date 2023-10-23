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
from .deformable_transformer import DeformableTransformer
# from .detr import DETR
from .detr_tracking import DETRTrackingBase


class DETRSegmBase(nn.Module):
    def __init__(self, freeze_detr=False,return_intermediate_masks=False, mask_dim = 288):
        if freeze_detr:
            for param in self.parameters():
                param.requires_grad_(False)

        self.return_intermediate_masks = return_intermediate_masks
    
        if self.decoder_use_mask_as_ref:
            self.decoder.forward_prediction_heads = self.forward_prediction_heads
            self.decoder.device = self.device
            self.decoder.iterative_masks = self.iterative_masks
    
    def forward(self, samples: NestedTensor, targets: list = None, track=False, prev_features=None, epoch=None, rand_num=None):
        out, targets, features, memory, hs, prev_out = super().forward(samples, targets,  track=track, prev_features=prev_features,epoch=epoch,rand_num=rand_num)

        if not self.decoder_use_mask_as_ref:
            pred_masks, pred_logits = self.forward_prediction_heads(hs[-1],len(hs)-1)

            out["pred_masks"] = pred_masks
            out["pred_logits"] = pred_logits

            for i in range(len(hs) - 1):
                pred_masks, pred_logits = self.forward_prediction_heads(hs[i],i)
                out["aux_outputs"][i]['pred_logits'] = pred_logits
                if self.return_intermediate_masks:
                    out["aux_outputs"][i]['pred_masks'] = pred_masks

        return out, targets, features, memory, hs, prev_out
    


# # TODO: with meta classes
# class DETRSegm(DETRSegmBase, DETR):
#     def __init__(self, mask_kwargs, detr_kwargs):
#         DETR.__init__(self, **detr_kwargs)
#         DETRSegmBase.__init__(self, **mask_kwargs)


class DeformableDETRSegm(DETRSegmBase, DeformableDETR, DeformableTransformer):
    def __init__(self, mask_kwargs, detr_kwargs, transformer_kwargs):
        DeformableTransformer.__init__(self, **transformer_kwargs)
        DeformableDETR.__init__(self, **detr_kwargs)
        DETRSegmBase.__init__(self, **mask_kwargs)


# class DETRSegmTracking(DETRSegmBase, DETRTrackingBase, DETR):
#     def __init__(self, mask_kwargs, tracking_kwargs, detr_kwargs):
#         DETR.__init__(self, **detr_kwargs)
#         DETRTrackingBase.__init__(self, **tracking_kwargs)
#         DETRSegmBase.__init__(self, **mask_kwargs)


class DeformableDETRSegmTracking(DETRSegmBase, DETRTrackingBase, DeformableDETR, DeformableTransformer):
    def __init__(self, mask_kwargs, tracking_kwargs, detr_kwargs, transformer_kwargs):
        DeformableTransformer.__init__(self, **transformer_kwargs)
        DeformableDETR.__init__(self, **detr_kwargs)
        DETRTrackingBase.__init__(self, **tracking_kwargs)
        DETRSegmBase.__init__(self, **mask_kwargs)