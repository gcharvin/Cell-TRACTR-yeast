# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import copy
import math

import torch
import torch.nn.functional as F
from torch import nn
import fvcore.nn.weight_init as weight_init

from ..util import box_ops
from ..util.misc import NestedTensor, inverse_sigmoid, nested_tensor_from_tensor_list,add_noise_to_boxes, MLP
from .detr import PostProcess, SetCriterion


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableDETR():
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, num_classes, num_queries, num_feature_levels,device,
                 aux_loss=True, with_box_refine=False, two_stage=False, overflow_boxes=False,
                 multi_frame_attention=False, multi_frame_encoding=False, merge_frame_features=False,                 
                 use_dab=True, random_refpoints_xy=False,
                  dn_object_l1 = 0, dn_object_l2 = 0, dn_label=0, refine_object_queries=False,
                  use_div_ref_pts = False, share_bbox_layers=True,decoder_use_mask_as_ref=False,
                  iterative_masks=False,use_img_for_mask=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal
                         number of objects DETR can detect in a single image. For COCO,
                         we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        
        self.device = device
        self.num_queries = num_queries
        self.hidden_dim = self.d_model
        self.decoder_use_mask_as_ref = decoder_use_mask_as_ref
        self.decoder.decoder_use_mask_as_ref = decoder_use_mask_as_ref
        self.iterative_masks = iterative_masks
        # self.transformer = transformer
        self.overflow_boxes = overflow_boxes
        self.class_embed = nn.Linear(self.hidden_dim, num_classes + 2)
        self.bbox_embed = MLP(self.hidden_dim, self.hidden_dim, 8, 3)
        self.mask_embed = MLP(self.hidden_dim, self.hidden_dim, self.mask_dim*2, 3)
        if two_stage:
            self.enc_class_embed = nn.Linear(self.hidden_dim, num_classes + 1)
            self.enc_bbox_embed = MLP(self.hidden_dim, self.hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, self.hidden_dim)

        # match interface with deformable DETR
        self.input_proj = nn.Conv2d(backbone.num_channels[-1], self.hidden_dim, kernel_size=1)

        self.backbone = backbone
        self.aux_loss = aux_loss

        self.use_img_for_mask = use_img_for_mask

        # super().__init__(backbone, transformer, num_classes, num_queries, device, two_stage, aux_loss)
        self.merge_frame_features = merge_frame_features
        self.multi_frame_attention = multi_frame_attention
        self.multi_frame_encoding = multi_frame_encoding
        self.overflow_boxes = overflow_boxes
        self.num_feature_levels = num_feature_levels
        self.num_queries = num_queries
        self.share_bbox_layers = share_bbox_layers

        self.dn_object_l1 = dn_object_l1
        self.dn_object_l2 = dn_object_l2

        self.dn_label = dn_label

        self.use_dab = use_dab
        self.random_refpoints_xy = random_refpoints_xy

        self.refine_object_queries = refine_object_queries

        if self.refine_object_queries:
            self.object_embedding = nn.Embedding(1,self.hidden_dim)

        ### DAB-DETR
        if not two_stage:
            if not use_dab:
                self.query_embed = nn.Embedding(num_queries, self.hidden_dim*2)
            else:
                self.tgt_embed = nn.Embedding(num_queries, self.hidden_dim)
                self.refpoint_embed = nn.Embedding(num_queries, 4)
                if random_refpoints_xy:
                    self.refpoint_embed.weight.data[:, :2].uniform_(0,1)
                    self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
                    self.refpoint_embed.weight.data[:, :2].requires_grad = False
        ### DAB-DETR

        ### DN-DETR --> dn_objects
        # This provides the content queries for the denoised objects. Only used during training
        self.label_enc = nn.Embedding(num_classes + 10, self.hidden_dim) # according to DN-DETR you can just add extra unused classes here
        ### DN-DETR --> dn_objects

        num_channels = backbone.num_channels[-num_feature_levels:]
        
        if num_feature_levels > 1:
            # return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            num_backbone_outs = len(backbone.strides) - 1

            input_proj_list = []
            for i in range(num_feature_levels):
                in_channels = num_channels[i]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(self.hidden_dim // 9, self.hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(self.hidden_dim // 2, self.hidden_dim),
                ))
                in_channels = self.hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(num_channels[0], self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(self.hidden_dim, self.hidden_dim),
                )])
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        # prior_prob = 0.01
        # bias_value = -math.log((1 - prior_prob) / prior_prob)
        # self.class_embed.bias.data = torch.ones_like(self.class_embed.bias) * bias_value

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = self.decoder.num_layers

        if two_stage:
            num_pred += 1

        if use_div_ref_pts:
            self.decoder.use_div_ref_pts = True

        if with_box_refine:
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0.)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0.)
            # self.class_embed = _get_clones(self.class_embed, num_pred)
            if not self.share_bbox_layers:
                self.mask_embed = _get_clones(self.mask_embed, num_pred)
                self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            else:
                self.mask_embed = nn.ModuleList([self.mask_embed for _ in range(num_pred)])
                self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])

            # hack implementation for iterative bounding box refinement
            self.decoder.bbox_embed = self.bbox_embed
            self.decoder.class_embed = self.class_embed
            self.decoder.mask_embed = self.mask_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0.)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0.)
            # nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:4], -2.0)
            # nn.init.constant_(self.bbox_embed.layers[-1].bias.data[6:], -2.0)
            # self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.mask_embed = nn.ModuleList([self.mask_embed for _ in range(num_pred)])
            self.decoder.bbox_embed = None
            self.decoder.class_embed = None
            self.decoder.mask_embed = None

        # if two_stage:
        #     # self.enc_class_embed.bias.data = torch.ones_like(self.enc_class_embed.bias) * bias_value
        #     # nn.init.constant_(self.enc_bbox_embed.layers[-1].weight.data, 0)
        #     # nn.init.constant_(self.enc_bbox_embed.layers[-1].bias.data, 0)

        #     self.enc_out_class_embed = self.class_embed
        #     self.enc_out_bbox_embed = self.bbox_embed[-1]

        #     if self.decoder_use_mask_as_ref:
        #         self.enc_out_mask_embed = self.mask_embed[-1]

        if self.merge_frame_features:
            self.merge_features = nn.Conv2d(self.hidden_dim * 2, self.hidden_dim, kernel_size=1)
            self.merge_features = _get_clones(self.merge_features, num_feature_levels)
    
        self.decoder_norm  = nn.LayerNorm(self.hidden_dim)

    def forward_prediction_heads(self, output, l):
        decoder_output = self.decoder_norm(output.transpose(0,1))
        decoder_output = decoder_output.transpose(0, 1)
        mask_embed = self.mask_embed[l](decoder_output)
        outputs_class = self.class_embed(decoder_output)

        outputs_mask_1 = torch.einsum("bqc,bchw->bqhw", mask_embed[:,:,:self.mask_dim], self.all_mask_features)
        outputs_mask_2 = torch.einsum("bqc,bchw->bqhw", mask_embed[:,:,self.mask_dim:], self.all_mask_features)

        outputs_mask = torch.stack((outputs_mask_1,outputs_mask_2),axis=2)

        return outputs_mask, outputs_class


    def forward(self, samples: NestedTensor, targets: list = None, output_target: str = None, prev_features=None, group_object=False, dn_object=False, dn_enc=False, add_object_queries_to_dn_track=False,return_features_only=False,track=True):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensors: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        self.samples = samples

        if return_features_only:
            return features

        features_all = features

        features = features[-self.num_feature_levels:]

        if prev_features is None:
            prev_features = features
        else:
            prev_features = prev_features[-self.num_feature_levels:]

        src_list = []
        mask_list = []
        pos_list = []

        frame_features = [prev_features, features]
        if not self.multi_frame_attention:
            frame_features = [features]

        for frame, frame_feat in enumerate(frame_features):
            if self.multi_frame_attention and self.multi_frame_encoding:
                pos_list.extend([p[:, frame] for p in pos[-self.num_feature_levels:]])
            else:
                pos_list.extend(pos[-self.num_feature_levels:])

            for l, feat in enumerate(frame_feat):
                src, mask = feat.decompose()

                if self.merge_frame_features:
                    prev_src, _ = prev_features[l].decompose()
                    src_list.append(self.merge_features[l](torch.cat([self.input_proj[l](src), self.input_proj[l](prev_src)], dim=1)))
                else:
                    src_list.append(self.input_proj[l](src))

                mask_list.append(mask)

                assert mask is not None

            if self.num_feature_levels > len(frame_feat):
                _len_srcs = len(frame_feat)
                for l in range(_len_srcs, self.num_feature_levels):
                    if l == _len_srcs:

                        if self.merge_frame_features:
                            src = self.merge_features[l](torch.cat([self.input_proj[l](frame_feat[-1].tensors), self.input_proj[l](prev_features[-1].tensors)], dim=1))
                        else:
                            src = self.input_proj[l](frame_feat[-1].tensors)
                    else:
                        src = self.input_proj[l](src_list[-1])

                    _, m = frame_feat[0].decompose()
                    mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]

                    pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                    src_list.append(src)
                    mask_list.append(mask)
                    if self.multi_frame_attention and self.multi_frame_encoding:
                        pos_list.append(pos_l[:, frame])
                    else:
                        pos_list.append(pos_l)

        bs = src.shape[0]
        training_methods = []
        #### DAB-DETR
        query_attn_mask = None 

        if self.use_dab:
            if self.two_stage:
                query_embeds = None
            else:
                tgt_embed = self.tgt_embed.weight.repeat(bs,1,1)      # nq, 256
                if self.refine_object_queries:
                    tgt_embed += self.object_embedding.weight
                refanchor = self.refpoint_embed.weight.repeat(bs,1,1)      # nq, 4
                query_embeds = torch.cat((tgt_embed, refanchor), dim=-1)

            num_queries = self.num_queries

            # Initialize the attn_mask
            if targets is not None and len(targets[0]) > 0 and 'track_query_hs_embeds' in targets[0][output_target]:
                num_track_queries = targets[0][output_target]['track_query_hs_embeds'].shape[0]
            else: 
                num_track_queries = 0

            num_total_queries = num_queries + num_track_queries 
            query_attn_mask = torch.zeros((num_total_queries,num_total_queries)).bool().to(self.device)

            #### DN-DETR for noised object detection
            if targets is not None and 'boxes' in targets[0][output_target] and dn_object and torch.tensor([target[output_target]['empty'] for target in targets]).sum() == 0: # If there is an empty chamber, skip all denoising
                training_methods.append('dn_object')

                num_boxes = torch.tensor([len(target[output_target]['labels_orig']) - int(target[output_target]['empty']) for target in targets]).to(self.device)
                num_FPs = max(num_boxes) - num_boxes

                if num_FPs.max() < 2:
                    num_FPs += torch.randint(4,(1,)).to(self.device)

                num_dn_object_queries = max(num_boxes + num_FPs)

                if self.two_stage:
                    query_embed_dn_object = torch.zeros((bs,num_dn_object_queries,self.hidden_dim + 4)).to(self.device)
                else:
                    query_embed_dn_object = torch.zeros((bs,num_dn_object_queries,query_embeds.shape[-1])).to(self.device)

                for t,target in enumerate(targets):

                    random_mask = torch.randperm(target[output_target]['boxes_orig'].shape[0]).to(self.device)

                    target['dn_object'] = {}
                    target['dn_object']['boxes'] = target[output_target]['boxes_orig'].clone()[random_mask]
                    target['dn_object']['labels'] = target[output_target]['labels_orig'].clone()[random_mask]
                    target['dn_object']['track_ids'] = target[output_target]['track_ids_orig'].clone()[random_mask]
                    target['dn_object']['flexible_division'] = target[output_target]['flexible_divisions_orig'].clone()[random_mask]
                    target['dn_object']['framenb'] = target[output_target]['framenb']
                    target['dn_object']['empty'] = target[output_target]['empty']
                    target['dn_object']['track_queries_fal_pos_mask'] = torch.zeros((num_dn_object_queries)).bool().to(self.device)
                    target['dn_object']['prev_target'] = target['prev_target'].copy()
                    target['dn_object']['fut_target'] = target['fut_target'].copy()

                    if 'masks' in targets[0]['cur_target']:
                        target['dn_object']['masks'] = target[output_target]['masks_orig'].clone()

                    l_1 = self.dn_object_l1
                    l_2 = self.dn_object_l2

                    noised_boxes = add_noise_to_boxes(target['dn_object']['boxes'][:,:4].clone(),l_1,l_2)
                    noised_labels = target['dn_object']['labels'][:,0].clone()

                    if num_FPs[t] > 0:
                        random_FP_mask = torch.randperm(min(num_FPs[t],target['dn_object']['boxes'].shape[0]))
                        FP_boxes = target['dn_object']['boxes'][random_FP_mask,:4].clone()

                        # If you use a batch size greater than 1, one image may have 5 cells and other has 1 cell. Therefore, you would need to have 4 denoised boxes for the 1 cell
                        while num_FPs[t] > FP_boxes.shape[0]:
                            random_FP_mask = torch.randperm(min(num_FPs[t] - FP_boxes.shape[0],target['dn_object']['boxes'].shape[0]))
                            FP_boxes = torch.cat((FP_boxes,target['dn_object']['boxes'][random_FP_mask,:4].clone()))

                        FP_boxes = add_noise_to_boxes(FP_boxes,l_1*3,l_2*3)
                        noised_boxes = torch.cat((noised_boxes, FP_boxes),axis=0)
                        # No tracking is done here; just a formality so it works in the matcher.py code; but there are FPs as in empty tracking boxes
                        target['dn_object']['track_queries_fal_pos_mask'][-num_FPs[t]:] = True

                        noised_labels = torch.cat((noised_labels,torch.ones((num_FPs[t])).long().to(self.device)))
                    
                    # Also a formality so it works in the mathcer.py code
                    target['dn_object']['track_queries_mask'] = torch.ones((num_dn_object_queries)).bool().to(self.device)
                    target['dn_object']['num_queries'] = num_dn_object_queries
                    target['dn_object']['noised_boxes'] = noised_boxes
                    target['dn_object']['noised_labels'] = noised_labels

                    label_embedding = self.label_enc(noised_labels)
                    query_embed_dn_object[t,:,:self.hidden_dim] = label_embedding
                    query_embed_dn_object[t,:,self.hidden_dim:] = noised_boxes      

                if self.two_stage:
                    query_embeds = query_embed_dn_object
                else:
                    query_embeds = torch.cat((query_embeds,query_embed_dn_object),axis=1)

                num_total_queries += num_dn_object_queries.item()
                new_query_attn_mask = torch.zeros((num_total_queries,num_total_queries)).bool().to(self.device)    
                new_query_attn_mask[:-num_dn_object_queries,:-num_dn_object_queries] = query_attn_mask
                new_query_attn_mask[-num_dn_object_queries:,:-num_dn_object_queries] = True
                new_query_attn_mask[:-num_dn_object_queries,-num_dn_object_queries:] = True
                query_attn_mask = new_query_attn_mask

        else:
            if self.two_stage:
                raise NotImplementedError
            else:
                query_embeds = self.query_embed.weight

        hs, memory, init_reference, inter_references, enc_outputs, training_methods, init_masks_ref, outputs_mask, outputs_class = \
            super().forward(features_all, src_list, mask_list, pos_list, query_embeds, targets, output_target, query_attn_mask, dn_enc, training_methods, add_object_queries_to_dn_track,track=track)

        save_references = torch.cat((init_reference[None],inter_references[...,:init_reference.shape[-1]]),axis=0)

        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if self.decoder_use_mask_as_ref:

                outputs_coord = self.bbox_embed[lvl](hs[lvl]).sigmoid()
                assert torch.sum(torch.isnan(outputs_coord)) == 0, 'Nan in boxes from prediction'
                assert torch.sum(outputs_coord < 0) == 0, 'Negative boxes in prediction'
                
            else:

                if lvl == 0:
                    reference = init_reference
                else:
                    reference = inter_references[lvl - 1]
                reference = inverse_sigmoid(reference)
                # outputs_class = self.class_embed[lvl](hs[lvl])
                tmp = self.bbox_embed[lvl](hs[lvl])
                if reference.shape[-1] == 4:
                    tmp[:,:,:4] += reference
                    tmp[:,:,4:] += reference
                else:
                    assert reference.shape[-1] == 2
                    tmp[..., :2] += reference
                    tmp[..., 4:6] += reference
                outputs_coord = tmp.sigmoid()
                # outputs_classes.append(outputs_class)

                assert torch.sum(torch.isnan(outputs_coord)) == 0, 'Nan in boxes from prediction'
                assert torch.sum(outputs_coord < 0) == 0, 'Negative boxes in prediction'
                
            outputs_coords.append(outputs_coord)

        # outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        training_methods = ['cur_target'] + training_methods

        out = {'pred_boxes': outputs_coord[-1],
               'hs_embed': hs[-1],
               'references': save_references,
               'training_methods': training_methods,}
        
        if outputs_mask is not None:
            out['pred_masks'] = outputs_mask[-1]
            out['pred_logits'] = outputs_class[-1]

        if self.aux_loss:
            if self.decoder_use_mask_as_ref:
                out['aux_outputs'] = self._set_aux_loss(outputs_coord, outputs_class, outputs_mask)
            else:
                out['aux_outputs'] = self._set_aux_loss(outputs_coord)

        if bool(enc_outputs):
            out['enc_outputs'] = enc_outputs

        offset = 0
        memory_slices = []
        batch_size, _, channels = memory.shape
        for src in src_list:
            _, _, height, width = src.shape
            memory_slice = memory[:, offset:offset + height * width].permute(0, 2, 1).view(
                batch_size, channels, height, width)
            memory_slices.append(memory_slice)
            offset += height * width

        memory = memory_slices[::-1] # We only care about encoder output from the first frame; this will be used in segmentation; flip the order to follow DINO

        return out, targets, features_all, memory, hs

    @torch.jit.unused
    def _set_aux_loss(self, outputs_coord, outputs_class=None, outputs_mask=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.

        if self.decoder_use_mask_as_ref:
            assert outputs_class is not None and outputs_mask is not None
            return [{'pred_boxes': a, 'pred_logits': b, 'pred_masks': c} for a, b, c in zip(outputs_coord[:-1], outputs_class[:-1], outputs_mask[:-1])]
        else:
            return [{'pred_boxes': a} for a in outputs_coord[:-1]]

class DeformablePostProcess(PostProcess):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes, results_mask=None):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()

        ###
        # topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        # scores = topk_values

        # topk_boxes = topk_indexes // out_logits.shape[2]
        # labels = topk_indexes % out_logits.shape[2]

        # boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        ###

        scores, labels = prob.max(-1)
        # scores, labels = prob[..., 0:1].max(-1)
        boxes = torch.cat((box_ops.box_cxcywh_to_xyxy(out_bbox[:,:4]),box_ops.box_cxcywh_to_xyxy(out_bbox[:,4:])),axis=1)

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [
            {'scores': s, 'scores_no_object': 1 - s, 'labels': l, 'boxes': b}
            for s, l, b in zip(scores, labels, boxes)]

        if results_mask is not None:
            for i, mask in enumerate(results_mask):
                for k, v in results[i].items():
                    results[i][k] = v[mask]

        return results
