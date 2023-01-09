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
                         interpolate, is_dist_avail_and_initialized,
                         nested_tensor_from_tensor_list, sigmoid_focal_loss,threshold_indices,
                         calc_bbox_acc, calc_track_acc,combine_div_boxes,calc_iou,divide_box,calc_object_query_FP)

class DETR(nn.Module):
    """ This is the DETR module that performs object detection. """

    def __init__(self, backbone, transformer, num_classes, num_queries, device, two_stage=False,
                 aux_loss=False, overflow_boxes=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal
                         number of objects DETR can detect in a single image. For COCO, we
                         recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.device = device
        self.num_queries = num_queries
        self.transformer = transformer
        self.overflow_boxes = overflow_boxes
        self.class_embed = nn.Linear(self.hidden_dim, num_classes + 2)
        self.bbox_embed = MLP(self.hidden_dim, self.hidden_dim, 8, 3)
        if two_stage:
            self.enc_class_embed = nn.Linear(self.hidden_dim, num_classes + 1)
            self.enc_bbox_embed = MLP(self.hidden_dim, self.hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, self.hidden_dim)

        # match interface with deformable DETR
        self.input_proj = nn.Conv2d(backbone.num_channels[-1], self.hidden_dim, kernel_size=1)

        self.backbone = backbone
        self.aux_loss = aux_loss

    @property
    def hidden_dim(self):
        """ Returns the hidden feature dimension size. """
        return self.transformer.d_model

    @property
    def fpn_channels(self):
        """ Returns FPN channels. """
        return self.backbone.num_channels[:4][::-1]
        # return [1024, 512, 256]

    def forward(self, samples: NestedTensor, targets: list = None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W],
                               containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized
                               in [0, 1], relative to the size of each individual image
                               (disregarding possible padding). See PostProcess for information
                               on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It
                                is a list of dictionnaries containing the two above keys for
                                each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        # src = self.input_proj[-1](src)
        src = self.input_proj(src)
        # pos = pos[-1]
        pos = pos[-1][:,0] ## DETR does not use prev frame so I got rid of preivous frame pos info

        batch_size, _, _, _ = src.shape

        query_embed = self.query_embed.weight
        query_embed = query_embed.unsqueeze(1).repeat(1, batch_size, 1)
        tgt = None
        if targets is not None and 'track_query_hs_embeds' in targets[0]:
            # [BATCH_SIZE, NUM_PROBS, 4]
            track_query_hs_embeds = torch.stack([t['track_query_hs_embeds'] for t in targets])

            num_track_queries = track_query_hs_embeds.shape[1]

            track_query_embed = torch.zeros(
                num_track_queries,
                batch_size,
                self.hidden_dim).to(query_embed.device)
            query_embed = torch.cat([
                track_query_embed,
                query_embed], dim=0)

            tgt = torch.zeros_like(query_embed)
            tgt[:num_track_queries] = track_query_hs_embeds.transpose(0, 1)

            for i, target in enumerate(targets):
                target['track_query_hs_embeds'] = tgt[:, i]

        assert mask is not None
        hs, hs_without_norm, memory = self.transformer(
            src, mask, query_embed, pos, tgt)

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1],
               'pred_boxes': outputs_coord[-1],
               'hs_embed': hs_without_norm[-1]}

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_class, outputs_coord)

        return out, targets, features, memory, hs

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 focal_loss, focal_alpha, focal_gamma, tracking, track_query_false_positive_eos_weight,args):
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
        self.track_query_false_positive_eos_weight = track_query_false_positive_eos_weight
        self.device = args.device
        self.args = args

    def update_target(self,target,index):

        target['boxes'] = torch.cat((target['boxes'][:index],torch.cat((target['boxes'][index,:4],torch.zeros(4,).to(self.device)))[None],torch.cat((target['boxes'][index,4:],torch.zeros(4,).to(self.device)))[None],target['boxes'][index+1:]))
        target['labels'] = torch.cat((target['labels'][:index],torch.cat((target['labels'][index,:1],torch.ones(1,).long().to(self.device)))[None],torch.cat((target['labels'][index,1:],torch.ones(1,).long().to(self.device)))[None],target['labels'][index+1:]))
        
        if 'masks' in target:
            N,_,H,W = target['masks'].shape
            target['masks'] = torch.cat((target['masks'][:index],torch.cat((target['masks'][index,:1],torch.zeros((1,H,W)).to(self.device,dtype=torch.uint8)))[None],torch.cat((target['masks'][index,1:],torch.zeros((1,H,W)).to(self.device,dtype=torch.uint8)))[None],target['masks'][index+1:]))

    def loss_labels(self, outputs, targets, indices, _, track=True, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2),
                                  target_classes,
                                  weight=self.empty_weight,
                                  reduction='none')

        if self.tracking and self.track_query_false_positive_eos_weight:
            for i, target in enumerate(targets):
                if 'track_query_boxes' in target and track:
                    # remove no-object weighting for false track_queries
                    loss_ce[i, target['track_queries_fal_pos_mask']] *= 1 / self.eos_coef
                    # assign false track_queries to some object class for the final weighting
                    target_classes = target_classes.clone()
                    target_classes[i, target['track_queries_fal_pos_mask']] = 0

        # weight = None
        # if self.tracking:
        #     weight = torch.stack([~t['track_queries_placeholder_mask'] for t in targets]).float()
        #     loss_ce *= weight

        loss_ce = loss_ce.sum() / self.empty_weight[target_classes].sum()

        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_labels_focal(self, outputs, targets, indices, num_boxes, track_div, two_stage=False, log=True):
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

        # for idx,swap_ind in enumerate(swap_indices):
        #     if len(swap_ind) > 0:
        #         target_classes_onehot[idx,swap_ind] = torch.flip(target_classes_onehot[idx,swap_ind],dims=[0,1])

        weights = torch.ones((src_logits.shape)).to('cuda')
        weights[target_classes_onehot[:,:,1] == 1.] = self.args.div_loss_coef

        #### need update code. Current method is really confusing
        #### This tells me which cells divided from frame t-2 to t-1 which we are trying to track from frame t-1 to t
        if not two_stage:
            for i in range(len(targets)):
                track_ind = torch.tensor([int(ind) for ind in indices[i][0] if ind < src_logits.shape[1] - self.args.num_queries])
                tgt_ind = indices[i][1][indices[i][0].unsqueeze(1).eq(track_ind).nonzero()[:,0]]

                if len(track_ind) > 0: 
                    weights[i,track_ind[targets[i]['track_div_mask'][tgt_ind]]] = self.args.track_div_loss_coef
        else:
            weights[:,:,1] = 0

        loss_ce = sigmoid_focal_loss(
            src_logits, target_classes_onehot, num_boxes, weights,
            alpha=self.focal_alpha, gamma=self.focal_gamma)

        loss_ce *= src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes, track_div, two_stage=False):
        """ Compute the cardinality error, ie the absolute error in the number of
            predicted non-empty boxes. This is not really a loss, it is intended
            for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes, track_div, two_stage=False):
        """Compute the losses related to the bounding boxes, the L1 regression loss
           and the GIoU loss targets dicts must contain the key "boxes" containing
           a tensor of dim [nb_target_boxes, 4]. The target boxes are expected in
           format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs

        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]

        assert (sum(src_boxes < 0) == 0).all(), 'Pred boxes should have positive values only' 

        target_boxes = [t['boxes'][i] for t, (_, i) in zip(targets, indices)]

        target_boxes = torch.cat(target_boxes,dim=0)

        # For empty chambers, there is a placeholder bbox of zeros that needs to be removed
        keep_not_empty_boxes = target_boxes.sum(-1) > 0
        target_boxes = target_boxes[keep_not_empty_boxes]
        src_boxes = src_boxes[keep_not_empty_boxes]

        keep = target_boxes[:,-1] > 0

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none') 
        loss_bbox[keep,4:] = loss_bbox[keep,4:] * self.args.div_loss_coef
        loss_bbox[track_div,:4] = loss_bbox[track_div,:4] * self.args.track_div_loss_coef
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
        loss_giou[track_div] = loss_giou[track_div] * self.args.track_div_loss_coef

        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes, track_div, two_stage=False):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of
           dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        src_masks = torch.cat((src_masks,torch.cat((src_masks[:,:,1:],src_masks[:,:,:1]),axis=2)),axis=1)

        target_masks = [t["masks"][i] for t, (_, i) in zip(targets, indices)]

        target_masks = torch.cat(target_masks,axis=0).to(src_masks)
        target_masks,_ = nested_tensor_from_tensor_list(target_masks).decompose()

        src_masks = src_masks[src_idx]

        keep_non_empty_chambers = target_masks.flatten(1).sum(-1) > 0
        target_masks = target_masks[keep_non_empty_chambers]
        src_masks = src_masks[keep_non_empty_chambers]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks, size=target_masks.shape[-2:],
                                    mode="bilinear", align_corners=False)
        division_ind = torch.cat([target['boxes'][:,-1] > 0 for target in targets if target['boxes'][0,0] > 0])

        sizes = [len(target['labels']) for target in targets]

        weights_mask = torch.ones((src_masks.shape)).to('cuda')
        weights_mask[division_ind] *= self.args.div_loss_coef # increase weight for divisions prev to current        
        weights_mask[~division_ind,1] = 0 # You don't care about the second prediction if the cell did not divide
        weights_mask[track_div] *= self.args.track_div_loss_coef # increase weight for divided cells that are using the same track query

        for t in range(len(targets)):
            if not targets[t]['empty']:
                mask_all_cells = torch.sum(torch.sum(target_masks[sum(sizes[:t]):sum(sizes[:t+1])],axis=0),axis=0)
                assert mask_all_cells.max() <= 1
                assert mask_all_cells.min() >= 0
                weights_mask[sum(sizes[:t]):sum(sizes[:t+1]),:,mask_all_cells == 1] *= self.args.mask_weight_cells_coef

        weights_mask[target_masks > 0] *= self.args.mask_weight_target_cell_coef

        weights_mask = weights_mask.flatten(1)

        weights_dice = torch.ones((src_masks.shape)).to('cuda')
        weights_dice[~division_ind,1] = 0 # You don't care about the second prediction
        weights_dice = weights_dice.flatten(1)

        src_masks = src_masks.flatten(1)

        target_masks = target_masks.flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes, weights_mask,
             alpha=self.focal_alpha, gamma=self.focal_gamma),
            "loss_dice": dice_loss(src_masks.sigmoid(), target_masks, num_boxes)
        }
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

    def get_loss(self, loss, outputs, targets, indices, num_boxes, track_div, two_stage=False, **kwargs):
        loss_map = {
            'labels': self.loss_labels_focal if self.focal_loss else self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'

        return loss_map[loss](outputs, targets, indices, num_boxes, track_div, two_stage, **kwargs)


    def forward(self, outputs, targets, return_bbox_track_acc=True):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied,
                      see each loss' doc
        """
        B,N,_ = outputs['pred_logits'].shape
        #### this outputs_without_aux should be replaced with outputs; need to double check
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        assert N < (self.args.num_queries + 15), f'Number of predictions ({N}) should not exceed {self.args.num_queries + 15}'

        indices = self.matcher(outputs_without_aux, targets)
        indices, targets = threshold_indices(indices,targets,max_ind=N)

        for i, (target,(ind_out,ind_tgt)) in enumerate(zip(targets,indices)):

            if 'object_detection_div_mask' in target:
                ind_out_clone = ind_out.clone()
                ind_tgt_clone = ind_tgt.clone()
                boxes_clone = target['boxes']
                track_ids_clone = target['track_ids']
                object_detection_div_mask = target['object_detection_div_mask'].clone()
                skip = []
                ind_tgt_blah = ind_tgt.clone()
                for idx in range(len(ind_tgt_clone)):
                    ind_tgt_box_1 = ind_tgt_clone[idx]
                    if object_detection_div_mask[ind_tgt_box_1] in skip:
                        continue
                    elif object_detection_div_mask[ind_tgt_box_1]: # check to see if a single cell was detected instead of two cells; we only care where there was a division from preivous to current frame
                        box_1 = boxes_clone[ind_tgt_box_1][:4]
                        box_label = object_detection_div_mask[ind_tgt_box_1]
                        ind_tgt_box_2 = torch.tensor([num for num in range(len(object_detection_div_mask)) if (object_detection_div_mask[num] == box_label and num != ind_tgt_box_1)])[0]
                        box_2 = boxes_clone[ind_tgt_box_2][:4]
                        sep_box = torch.cat((box_1,box_2),axis=-1)

                        ind_out_box_1 = ind_out_clone[idx]
                        pred_box_1 = outputs['pred_boxes'][i,ind_out_box_1].detach()

                        pred_box_2_ind = (ind_tgt_clone == ind_tgt_box_2).nonzero()[0][0]
                        ind_out_box_2 = ind_out_clone[pred_box_2_ind]
                        pred_box_2 = outputs['pred_boxes'][i,ind_out_box_2].detach()

                        skip.append(box_label)

                        pred_boxes_sep = torch.cat((pred_box_1[:4],pred_box_2[:4]),axis=-1)

                        iou = calc_iou(pred_boxes_sep,sep_box)

                        track_id = track_ids_clone[ind_tgt_box_1]

                        if track_id == -1:
                            track_id = track_ids_clone[ind_tgt_box_2]

                        prev_box_ind = target['prev_target']['track_ids'].eq(track_id).nonzero()[0][0]
                        prev_box = target['prev_target']['boxes'][prev_box_ind]

                        combined_box = combine_div_boxes(sep_box,prev_box)

                        unused_object_query_indices = torch.tensor([ind_out_box_1,ind_out_box_2] + [oq_id for oq_id in torch.arange(N-len(target['track_queries_mask']),N) if (oq_id not in ind_out_clone and outputs['pred_logits'][i,oq_id,0].sigmoid().detach() > 0.5)])
            
                        unused_pred_boxes = outputs['pred_boxes'][i,unused_object_query_indices].detach() 

                        iou_combined = box_ops.generalized_box_iou(box_ops.box_cxcywh_to_xyxy(unused_pred_boxes[:,:4]),box_ops.box_cxcywh_to_xyxy(combined_box[None,:4]),return_iou_only=True)

                        max_ind = torch.argmax(iou_combined)

                        assert iou_combined[max_ind] <= 1 and iou_combined[max_ind] >= 0 and iou <= 1 and iou >= 0

                        if iou_combined[max_ind] - iou > 0:

                            query_id = unused_object_query_indices[max_ind]
                            
                            if max_ind == 1:
                                ind_out_box_1,ind_out_box_2 = ind_out_box_2,ind_out_box_1
                                ind_tgt_box_1,ind_tgt_box_2 = ind_tgt_box_2,ind_tgt_box_1
                            
                            ind_out[ind_out == ind_out_box_1] = query_id

                            assert (ind_out == ind_out_box_2).sum() == 1

                            ind_out = ind_out[ind_out != ind_out_box_2]                                

                            ind_tgt_box_1_new = ind_tgt[(ind_tgt_blah == ind_tgt_box_1).nonzero()[0][0]]
                            ind_tgt_box_2_new = ind_tgt[(ind_tgt_blah == ind_tgt_box_2).nonzero()[0][0]]

                            ind_tgt = ind_tgt[ind_tgt != ind_tgt_box_2_new]
                            ind_tgt[ind_tgt > ind_tgt_box_2_new] = ind_tgt[ind_tgt > ind_tgt_box_2_new] - 1

                            assert len(ind_tgt) + 1 == len(ind_tgt_blah)

                            ind_tgt_blah = ind_tgt_blah[ind_tgt_blah != ind_tgt_box_2]

                            target['boxes'][ind_tgt_box_1_new] = combined_box
                            ind_tgt_boxes = torch.tensor([ind_tgt_box for ind_tgt_box in range(len(target['boxes'])) if ind_tgt_box != ind_tgt_box_2_new])
                            target['boxes'] = target['boxes'][ind_tgt_boxes]
                            target['labels'] = target['labels'][ind_tgt_boxes]
                            target['track_ids'] = target['track_ids'][ind_tgt_boxes]
                            target['object_detection_div_mask'] = target['object_detection_div_mask'][ind_tgt_boxes]
                            
                            assert len(ind_out) == len(ind_tgt)
                            assert len(target['boxes']) == len(target['labels'])

                            if 'masks' in target:
                                raise NotImplementedError

                            indices[i] = (ind_out,ind_tgt)

                            if 'track_query_match_ids' in target:
                                prev_track_ids = target['prev_target']['track_ids'][target['prev_ind'][1]]
                                target_ind_match_matrix = prev_track_ids.unsqueeze(dim=1).eq(target['track_ids'])
                                target['target_ind_matching'] = target_ind_match_matrix.any(dim=1)
                                target['cells_leaving_mask'] = torch.cat((~target_ind_match_matrix.any(dim=1),(torch.tensor([False, ] * target['num_FPs'])).bool().to(self.device)))
                                
                                target['track_query_match_ids'] = target_ind_match_matrix.nonzero()[:, 1]

            test_early_div = (target['fut_target']['boxes'][:,-1] > 0).any()
            
            if test_early_div: # quick check to see if any cells divided in the future frame
                
                pred_boxes = outputs['pred_boxes'].detach()
                ind_tgt_clone = ind_tgt.clone()
                ind_out_clone = ind_out.clone()
                boxes = target['boxes'].clone()
                for idx in range(len(ind_tgt)):
                    if (~target['track_queries_mask'])[ind_out_clone[idx]] and target['object_detection_div_mask'][ind_tgt_clone[idx]] == 0.: # check that we are looking at non-tracked cells

                        box = boxes[ind_tgt_clone[idx]]

                        fut_prev_boxes = target['fut_prev_target']['boxes']
                        box_ind_match_id = box[:4].eq(fut_prev_boxes[:,:4]).all(axis=-1).nonzero()[0][0]
                        fut_prev_track_id = target['fut_prev_target']['track_ids'][box_ind_match_id]

                        #### TODO maybe add feature to allow division possible
                        if fut_prev_track_id not in target['fut_target']['track_ids']:
                            continue  # Cell leaves chamber in future frame

                        fut_box_ind = (target['fut_target']['track_ids'] == fut_prev_track_id).nonzero()[0][0]
                        fut_box = target['fut_target']['boxes'][fut_box_ind]

                        if fut_box[-1] > 0: # If cell divides next frame, we check to see if the model is predicting an early division

                            div_box = divide_box(box,fut_box)

                            unused_object_query_indices = torch.tensor([ind_out[idx]] + [oq_id for oq_id in torch.arange(N-len(target['track_queries_mask']),N) if (oq_id not in ind_out and outputs['pred_logits'][i,oq_id,0].sigmoid().detach() > 0.5)])
                
                            if len(unused_object_query_indices) > 1:

                                unused_pred_boxes = pred_boxes[i,unused_object_query_indices]

                                iou_div_all = box_ops.generalized_box_iou(box_ops.box_cxcywh_to_xyxy(unused_pred_boxes[:,:4]),box_ops.box_cxcywh_to_xyxy(torch.cat((div_box[None,:4],div_box[None,4:]),axis=0)),return_iou_only=True)

                                match_ind = torch.argmax(iou_div_all,axis=0).to('cpu')

                                if len(torch.unique(match_ind)) == 2:
                                    selected_pred_boxes = unused_pred_boxes[match_ind,:4]
                                    iou_div = calc_iou(div_box, torch.cat((selected_pred_boxes[0],selected_pred_boxes[1])))

                                    iou = calc_iou(box,torch.cat((pred_boxes[i,ind_out[idx],:4],torch.zeros_like(pred_boxes[i,ind_out[idx],:4]))))

                                    assert iou_div <= 1 and iou_div >= 0 and iou <= 1 and iou >= 0

                                    if iou_div - iou > 0:
                                        target['boxes'][ind_tgt_clone[idx]] = torch.cat((div_box[:4],torch.zeros_like(div_box[:4])))
                                        target['boxes'] = torch.cat((target['boxes'],torch.cat((div_box[4:],torch.zeros_like(div_box[:4])))[None]))

                                        target['labels'][ind_tgt_clone[idx]] = torch.tensor([0,1]).to(self.device)
                                        target['labels'] = torch.cat((target['labels'],torch.tensor([0,1])[None,].to(self.device)))

                                        if 'masks' in target:
                                            raise NotImplementedError

                                        ind_out[ind_out == ind_out_clone[idx]] = torch.tensor([unused_object_query_indices[match_ind[0]]])
                                        ind_out = torch.cat((ind_out,torch.tensor([unused_object_query_indices[match_ind[1]]])))
                                        ind_tgt = torch.cat((ind_tgt,torch.tensor([target['boxes'].shape[0]-1])))                    

                                        assert len(ind_out) == len(ind_tgt)
                                        assert len(target['boxes']) == len(target['labels'])

                                        indices[i] = (ind_out,ind_tgt)


        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets) - sum(t["empty"] for t in targets) # Empty chambers have an empty box and label as placeholder so we need to subtract it as a box
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=outputs['pred_logits'].device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item() 

        assert sum([(target['boxes'][:,0] == 0).sum() for target in targets if not target['empty']]) == 0

        sizes = [len(target['labels']) - int(target["empty"]) for target in targets] 
        track_div = torch.zeros((int(num_boxes))).bool()

        for t,target in enumerate(targets):
            if sizes[t] != 0 and 'track_div_mask' in target:
                track_div[sum(sizes[:t]):sum(sizes[:t+1])] = target['track_div_mask'][indices[t][0]]

        # Compute the segmentation and tracking metrics
        cls_threshold = 0.5
        iou_threshold = 0.75
        if return_bbox_track_acc:
            metrics = {}
            bbox_det_only_acc, bbox_FN_acc = calc_bbox_acc(outputs,targets,indices,cls_thresh=cls_threshold,iou_thresh=iou_threshold)
            metrics['overall_object_det_acc'] = bbox_det_only_acc + bbox_FN_acc
            metrics['no_tracking_object_det_acc'] = bbox_det_only_acc
            metrics['untracked_object_det_acc'] = bbox_FN_acc
            track_acc, div_acc, track_post_div_acc, cells_leaving_acc, rand_FP_acc = calc_track_acc(outputs,targets,indices,cls_thresh=cls_threshold,iou_thresh=iou_threshold)
            metrics['track_queries_only_track_acc'] = track_acc
            metrics['divisions_track_acc'] = div_acc
            metrics['post_division_track_acc'] = track_post_div_acc
            metrics['cells_leaving_track_acc'] = cells_leaving_acc
            metrics['rand_FP_track_acc'] = rand_FP_acc
            object_query_FP_track_acc = calc_object_query_FP(outputs,targets,indices,cls_thresh=cls_threshold,iou_thresh=iou_threshold)
            metrics['object_query_FP_track_acc'] = object_query_FP_track_acc
            metrics['overall_track_acc'] = track_acc + object_query_FP_track_acc

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            if sum(sizes) != 0 or (sum(sizes) == 0 and loss == 'labels'): # If two empty chambers, only loss will be computed for labels as there is nothing to computer for the boxes / masks
                assert N < (self.args.num_queries + 15)
                losses.update(self.get_loss(loss, outputs_without_aux, targets, indices, num_boxes,track_div))

        # In case of auxiliary losses, we repeat this process with the
        # output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                B,N,_ = aux_outputs['pred_logits'].shape
                assert N < (self.args.num_queries + 15), f'Number of predictions ({N}) should not exceed {self.args.num_queries + 15}'
                
                indices = self.matcher(aux_outputs, targets)

                indices, targets = threshold_indices(indices,targets,max_ind=N)

                for loss in self.losses:
                    if (loss == 'masks' and 'pred_masks' not in aux_outputs) or (sum(sizes) == 0 and loss != 'labels'):
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, track_div, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            B,N,_ = enc_outputs['pred_logits'].shape
            # bin_targets = copy.deepcopy(targets)
            bin_targets = [{} for _ in range(len(targets))]

            for b,bin_target in enumerate(bin_targets):
                bin_target['labels'] = targets[b]['labels'].clone()
                bin_target['boxes'] = targets[b]['boxes'].clone()
                bin_target['empty'] = targets[b]['empty'].clone()
                # if 'track_query_match_ids' in bin_target:
                #     bin_target.pop('track_query_match_ids')
                count = 0
                for b in range(len(bin_target['boxes'])):
                    if bin_target['boxes'][b + count,-1] > 0:
                        self.update_target(bin_target,b+count)
                        count += 1

                if not bin_target['empty']:
                    assert (bin_target['labels'][0:,0] == 0).all() and (bin_target['labels'][0:,1] == 1).all()
                    assert (bin_target['boxes'][:,-1] == 0).all()

            enc_outputs['pred_boxes'] = torch.cat((enc_outputs['pred_boxes'],torch.zeros_like(enc_outputs['pred_boxes']).to(self.device)),axis=-1)

            num_boxes = sum(len(t["labels"]) for t in bin_targets) - sum(t["empty"] for t in bin_targets)
            track_div = torch.zeros((num_boxes)).bool()

            indices = self.matcher(enc_outputs, bin_targets)

            # should remove this line in future 
            indices, targets = threshold_indices(indices,targets,max_ind=N)
            
            for loss in self.losses:
                if (loss == 'masks' and 'pred_masks' not in aux_outputs) or (sum(sizes) == 0 and loss != 'labels'):
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, track_div, two_stage=True, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        if return_bbox_track_acc:
            return losses, metrics
        else:
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


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k)
            for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
