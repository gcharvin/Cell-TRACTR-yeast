# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import copy

import torch
import torch.nn.functional as F
from torch import nn
from scipy.optimize import linear_sum_assignment

from ..util import box_ops
from ..util.misc import (NestedTensor, accuracy, dice_loss, get_world_size,
                         interpolate, is_dist_avail_and_initialized, MLP,combine_div_masks,divide_mask,
                         nested_tensor_from_tensor_list, sigmoid_focal_loss,threshold_indices,
                        combine_div_boxes,calc_iou,divide_box,update_early_or_late_track_divisions,update_object_detection)

# class DETR(nn.Module):
#     """ This is the DETR module that performs object detection. """

#     def __init__(self, backbone, transformer, num_classes, num_queries, device, two_stage=False,
#                  aux_loss=False, overflow_boxes=False):
#         """ Initializes the model.
#         Parameters:
#             backbone: torch module of the backbone to be used. See backbone.py
#             transformer: torch module of the transformer architecture. See transformer.py
#             num_classes: number of object classes
#             num_queries: number of object queries, ie detection slot. This is the maximal
#                          number of objects DETR can detect in a single image. For COCO, we
#                          recommend 100 queries.
#             aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
#         """
#         super().__init__()
#         self.device = device
#         self.num_queries = num_queries
#         self.transformer = transformer
#         self.overflow_boxes = overflow_boxes
#         self.class_embed = nn.Linear(self.hidden_dim, num_classes + 2)
#         self.bbox_embed = MLP(self.hidden_dim, self.hidden_dim, 8, 3)
#         if two_stage:
#             self.enc_class_embed = nn.Linear(self.hidden_dim, num_classes + 1)
#             self.enc_bbox_embed = MLP(self.hidden_dim, self.hidden_dim, 4, 3)
#         self.query_embed = nn.Embedding(num_queries, self.hidden_dim)

#         # match interface with deformable DETR
#         self.input_proj = nn.Conv2d(backbone.num_channels[-1], self.hidden_dim, kernel_size=1)

#         self.backbone = backbone
#         self.aux_loss = aux_loss

    # @property
    # def hidden_dim(self):
    #     """ Returns the hidden feature dimension size. """
    #     return self.transformer.d_model

    # @property
    # def fpn_channels(self):
    #     """ Returns FPN channels. """
    #     return self.backbone.num_channels[:4][::-1]
    #     # return [1024, 512, 256]

    # def forward(self, samples: NestedTensor, targets: list = None):
    #     """ The forward expects a NestedTensor, which consists of:
    #            - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
    #            - samples.mask: a binary mask of shape [batch_size x H x W],
    #                            containing 1 on padded pixels

    #         It returns a dict with the following elements:
    #            - "pred_logits": the classification logits (including no-object) for all queries.
    #                             Shape= [batch_size x num_queries x (num_classes + 1)]
    #            - "pred_boxes": The normalized boxes coordinates for all queries, represented as
    #                            (center_x, center_y, height, width). These values are normalized
    #                            in [0, 1], relative to the size of each individual image
    #                            (disregarding possible padding). See PostProcess for information
    #                            on how to retrieve the unnormalized bounding box.
    #            - "aux_outputs": Optional, only returned when auxilary losses are activated. It
    #                             is a list of dictionnaries containing the two above keys for
    #                             each decoder layer.
    #     """
    #     if not isinstance(samples, NestedTensor):
    #         samples = nested_tensor_from_tensor_list(samples)
    #     features, pos = self.backbone(samples)

    #     src, mask = features[-1].decompose()
    #     # src = self.input_proj[-1](src)
    #     src = self.input_proj(src)
    #     # pos = pos[-1]
    #     pos = pos[-1][:,0] ## DETR does not use prev frame so I got rid of preivous frame pos info

    #     batch_size, _, _, _ = src.shape

    #     query_embed = self.query_embed.weight
    #     query_embed = query_embed.unsqueeze(1).repeat(1, batch_size, 1)
    #     tgt = None
    #     if targets is not None and 'track_query_hs_embeds' in targets[0]:
    #         # [BATCH_SIZE, NUM_PROBS, 4]
    #         track_query_hs_embeds = torch.stack([t['track_query_hs_embeds'] for t in targets])

    #         num_track_queries = track_query_hs_embeds.shape[1]

    #         track_query_embed = torch.zeros(
    #             num_track_queries,
    #             batch_size,
    #             self.hidden_dim).to(query_embed.device)
    #         query_embed = torch.cat([
    #             track_query_embed,
    #             query_embed], dim=0)

    #         tgt = torch.zeros_like(query_embed)
    #         tgt[:num_track_queries] = track_query_hs_embeds.transpose(0, 1)

    #         for i, target in enumerate(targets):
    #             target['track_query_hs_embeds'] = tgt[:, i]

    #     assert mask is not None
    #     hs, hs_without_norm, memory = self.transformer(
    #         src, mask, query_embed, pos, tgt)

    #     outputs_class = self.class_embed(hs)
    #     outputs_coord = self.bbox_embed(hs).sigmoid()
    #     out = {'pred_logits': outputs_class[-1],
    #            'pred_boxes': outputs_coord[-1],
    #            'hs_embed': hs_without_norm[-1]}

    #     if self.aux_loss:
    #         out['aux_outputs'] = self._set_aux_loss(
    #             outputs_class, outputs_coord)

    #     return out, targets, features, memory, hs

    # @torch.jit.unused
    # def _set_aux_loss(self, outputs_class, outputs_coord):
    #     # this is a workaround to make torchscript happy, as torchscript
    #     # doesn't support dictionary with non-homogeneous values, such
    #     # as a dict having both a Tensor and a list.
    #     return [{'pred_logits': a, 'pred_boxes': b}
    #             for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 focal_loss, focal_alpha, focal_gamma, tracking,args):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their
                         relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of
                    available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.focal_loss = focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.tracking = tracking
        self.device = args.device
        self.args = args

    def loss_labels_focal(self, outputs, targets, indices, num_boxes):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full((src_logits.shape[0],src_logits.shape[1],src_logits.shape[2]), self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target_classes_onehot = ((target_classes == 0) * 1).float()

        weights = torch.ones((src_logits.shape)).to(self.device)

        num_preds = src_logits.shape[1]

        if 'CoMOT' in outputs and outputs['CoMOT_loss_ce']:
            weights = torch.zeros((src_logits.shape)).to(self.device)
            weights[idx] = 1.
            num_preds = num_boxes

        weights[target_classes_onehot[:,:,1] == 1.] *= self.args.div_loss_coef
        weights[target_classes_onehot == 1.] *= self.args.pos_wei_loss_coef

        if targets[0]['output_name'] == 'enc_outputs':
            weights[:,:,1] = 0

        loss_ce = sigmoid_focal_loss(
            src_logits, target_classes_onehot, num_boxes, weights,
            alpha=self.focal_alpha, gamma=self.focal_gamma)

        loss_ce *= num_preds

        if targets[0]['output_name'] == 'enc_outputs' and self.args.mult_enc_loss:
            loss_ce *= self.args.dec_layers

        losses = {'loss_ce': loss_ce}

        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss
           and the GIoU loss targets dicts must contain the key "boxes" containing
           a tensor of dim [nb_target_boxes, 4]. The target boxes are expected in
           format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs

        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]

        assert (sum(src_boxes < 0) == 0).all(), 'Pred boxes should have positive values only' 

        target_boxes = [t['boxes'][i] for t, (_, i) in zip(targets, indices) if not t['empty']]

        target_boxes = torch.cat(target_boxes,dim=0)

        # For empty chambers, there is a placeholder bbox of zeros that needs to be removed
        keep_not_empty_boxes = target_boxes.sum(-1) > 0
        target_boxes = target_boxes[keep_not_empty_boxes]
        src_boxes = src_boxes[keep_not_empty_boxes]

        keep = target_boxes[:,-1] > 0

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none') 
        loss_bbox[keep,4:] = loss_bbox[keep,4:] * self.args.div_loss_coef
        loss_bbox[~keep,4:] = 0 # If there is no division, we do not care about the second bounding box
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
       

        loss_giou_track = box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes[:,:4]),
            box_ops.box_cxcywh_to_xyxy(target_boxes[:,:4])) 

        loss_giou_track[:,keep] = loss_giou_track[:,keep] / 2 + box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes[:,4:]),
            box_ops.box_cxcywh_to_xyxy(target_boxes[:,4:]))[:,keep] / 2

        loss_giou = 1 - torch.diag(loss_giou_track)
        loss_giou[keep] = loss_giou[keep] * self.args.div_loss_coef

        losses['loss_giou'] = loss_giou.sum() / num_boxes

        if targets[0]['output_name'] == 'enc_outputs' and self.args.mult_enc_loss:
            losses['loss_bbox'] *= self.args.dec_layers
            losses['loss_giou'] *= self.args.dec_layers

        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of
           dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        target_masks = [t["masks"][i] for t, (_, i) in zip(targets, indices) if not t['empty']]

        target_masks = torch.cat(target_masks,axis=0).to(self.device)
        target_masks,_ = nested_tensor_from_tensor_list(target_masks).decompose()

        src_masks = src_masks[src_idx]

        keep_non_empty_chambers = target_masks.flatten(1).sum(-1) > 0
        target_masks = target_masks[keep_non_empty_chambers]
        src_masks = src_masks[keep_non_empty_chambers]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks, size=target_masks.shape[-2:],
                                    mode="bilinear", align_corners=False)

        division_ind = target_masks[:,1].sum(-1).sum(-1) > 0

        weights_mask = torch.ones((src_masks.shape)).to(self.device)
        weights_mask[division_ind] *= self.args.div_loss_coef # increase weight for divisions prev to current        
        weights_mask[~division_ind,1] = 0 # You don't care about the second prediction if the cell did not divide
        weights_mask[target_masks > 0] *= self.args.mask_weight_target_cell_coef

        weights_mask = weights_mask.flatten(1)

        weights_dice = torch.ones((src_masks.shape)).to(self.device)
        weights_dice[~division_ind,1] = 0 # You don't care about the second prediction
        weights_dice = weights_dice.flatten(1)

        src_masks = src_masks.flatten(1)

        target_masks = target_masks.flatten(1).float()

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes, weights_mask,
             alpha=self.focal_alpha, gamma=self.focal_gamma, mask=True),
            "loss_dice": dice_loss(src_masks.sigmoid()*weights_dice, target_masks, num_boxes)
        }

        if targets[0]['output_name'] == 'enc_outputs' and self.args.mult_enc_loss:
            losses['loss_mask'] *= self.args.dec_layers
            losses['loss_dice'] *= self.args.dec_layers

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            'labels': self.loss_labels_focal if self.focal_loss else self.loss_labels,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'

        return loss_map[loss](outputs, targets, indices, num_boxes)


    def forward(self, outputs, targets, output_target='cur_target', epoch=None):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied,
                      see each loss' doc
        """

        for target in targets:
            target[output_target]['output_name'] = output_target

        if  self.args.flex_div and output_target == 'cur_target' and (epoch is None or epoch > self.args.epoch_to_start_using_flexible_divisions):
            targets = update_early_or_late_track_divisions(outputs,targets, 'prev_target','cur_target','fut_target')

        #### this outputs_without_aux should be replaced with outputs; need to double check
        # outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        outputs_without_aux = outputs

        indices, targets = self.matcher(outputs_without_aux, targets, output_target)

        if self.args.flex_div and output_target == 'cur_target':
            num_queries = self.args.num_queries
            targets, indices = update_object_detection(outputs_without_aux,targets,indices,num_queries,'prev_target','cur_target','fut_target')

        for t,target in enumerate(targets):
            target[output_target]['indices'] = indices[t]

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum((1 - t[output_target]["labels"]).sum() for t in targets if not t[output_target]["empty"])
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=outputs['pred_logits'].device)

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item() 

        assert sum([(target[output_target]['boxes'][:,0] == 0).sum() for target in targets if not target[output_target]['empty']]) == 0
        sizes = [len(target[output_target]['labels']) - int(target[output_target]["empty"]) for target in targets] 

        # Compute all the requested losses
        losses = {}
        targets_loss = [target[output_target] for target in targets]
    
        for loss in self.losses:
            if sum(sizes) != 0 or (sum(sizes) == 0 and loss == 'labels'): # If two empty chambers, only loss will be computed for labels as there is nothing to computer for the boxes / masks
                losses.update(self.get_loss(loss, outputs_without_aux, targets_loss, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the
        # output of each intermediate layer.
        if 'aux_outputs' in outputs:

            if self.args.CoMOT and output_target == 'cur_target' and targets[0]['track'] and outputs['pred_logits'].shape[1] > self.args.num_queries:
                CoMOT = True
            else:
                CoMOT = False

            if CoMOT:
                for target in targets:
                    target['CoMOT'] = {'output_name': 'CoMOT'}
                    target['CoMOT']['labels'] = target[output_target]['labels_orig'].clone()
                    target['CoMOT']['boxes'] = target[output_target]['boxes_orig'].clone()
                    target['CoMOT']['track_ids'] = target[output_target]['track_ids_orig'].clone()
                    target['CoMOT']['empty'] = target[output_target]['empty'].clone()

                    if 'masks' in target[output_target]:
                        target['CoMOT']['masks'] = target[output_target]['masks_orig'].clone()


            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if 'track_queries_TP_mask' in target['cur_target']:
                    assert target['cur_target']['track_queries_TP_mask'].sum() == len(target['cur_target']['track_query_match_ids'])

                if CoMOT:
                    aux_outputs['CoMOT_loss_ce'] = self.args.CoMOT_loss_ce
                    aux_outputs['CoMOT'] = True

                indices,targets = self.matcher(aux_outputs, targets, output_target)
                targets_loss = [target[output_target] for target in targets]
                if 'track_queries_TP_mask' in target['cur_target']:
                    assert target['cur_target']['track_queries_TP_mask'].sum() == len(target['cur_target']['track_query_match_ids'])

                for loss in self.losses:
                    if (loss == 'masks' and 'pred_masks' not in aux_outputs) or (sum(sizes) == 0 and loss != 'labels'):
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue

                    l_dict = self.get_loss(loss, aux_outputs, targets_loss, indices, num_boxes)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

                if CoMOT:

                    aux_outputs_without_track = {}
                    for key in ['pred_logits', 'pred_boxes']:
                        aux_outputs_without_track[key] = aux_outputs[key][:,-self.args.num_queries:].clone()

                    if 'pred_masks' in aux_outputs:
                        aux_outputs_without_track['pred_masks'] = aux_outputs['pred_masks'][:,-self.args.num_queries:].clone()

                    indices,targets = self.matcher(aux_outputs_without_track, targets, 'CoMOT')
                    aux_outputs['CoMOT_indices'] = indices

                    targets_loss = [target['CoMOT'] for target in targets]

                    for loss in self.losses:
                        if (loss == 'masks' and 'pred_masks' not in aux_outputs) or (sum(sizes) == 0 and loss != 'labels') or (loss == 'labels' and not self.args.CoMOT_loss_ce):
                            # Intermediate masks losses are too costly to compute, we ignore them.
                            continue

                        l_dict = self.get_loss(loss, aux_outputs_without_track, targets_loss, indices, num_boxes)
                        l_dict = {k + f'_{i}_CoMOT': v for k, v in l_dict.items()}
                        losses.update(l_dict)

        if 'enc_outputs' in outputs:# and sum(sizes) > 0:
            enc_outputs = outputs['enc_outputs']
            
            for t,target in enumerate(targets):
                target['enc_outputs'] = {'output_name': 'enc_outputs'}
                target['enc_outputs']['labels'] = target[output_target]['labels_orig'].clone()
                target['enc_outputs']['boxes'] = target[output_target]['boxes_orig'].clone()
                target['enc_outputs']['track_ids'] = target[output_target]['track_ids_orig'].clone()
                target['enc_outputs']['empty'] = target[output_target]['empty'].clone()

                if 'masks' in target[output_target]:
                    target['enc_outputs']['masks'] = target[output_target]['masks_orig'].clone()

                if not target['enc_outputs']['empty']:
                    assert (target['enc_outputs']['labels'][0:,0] == 0).all() and (target['enc_outputs']['labels'][0:,1] == 1).all()
                    assert (target['enc_outputs']['boxes'][:,-1] == 0).all()
                    if 'masks' in target[output_target]:
                        assert (target['enc_outputs']['masks'][:,-1] == 0).all()

            num_boxes = max(sum(len(t['enc_outputs']["labels"]) for t in targets) - sum(t['enc_outputs']["empty"] for t in targets),1)

            indices, targets = self.matcher(enc_outputs, targets,'enc_outputs')
            outputs['enc_outputs']['indices'] = indices
            
            targets_loss = [target['enc_outputs'] for target in targets]

            for loss in self.losses:
                if (sum(sizes) == 0 and loss != 'labels') or (loss == 'masks' and 'pred_masks' not in enc_outputs):
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue

                l_dict = self.get_loss(loss, enc_outputs, targets_loss, indices, num_boxes)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    def process_boxes(self, boxes, target_sizes):
        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        return boxes

    @torch.no_grad()
    def forward(self, outputs, target_sizes, results_mask=None):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of
                          each images of the batch For evaluation, this must be the
                          original image size (before any data augmentation) For
                          visualization, this should be the image size after data
                          augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        boxes = self.process_boxes(out_bbox, target_sizes)


        results = [
            {'scores': s, 'labels': l, 'boxes': b, 'scores_no_object': s_n_o}
            for s, l, b, s_n_o in zip(scores, labels, boxes, prob[..., -1])]

        if results_mask is not None:
            for i, mask in enumerate(results_mask):
                for k, v in results[i].items():
                    results[i][k] = v[mask]

        return results



