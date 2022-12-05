import math
import random
from contextlib import nullcontext
import numpy as np

import torch
import torch.nn as nn

from ..util import box_ops
from ..util.misc import NestedTensor, get_rank, threshold_indices
from .deformable_detr import DeformableDETR
from .detr import DETR
from .matcher import HungarianMatcher


class DETRTrackingBase(nn.Module):

    def __init__(self,
                 track_query_false_positive_prob: float = 0.0,
                 track_query_false_negative_prob: float = 0.0,
                 matcher: HungarianMatcher = None,
                 backprop_prev_frame=False):
        self._matcher = matcher
        self._track_query_false_positive_prob = track_query_false_positive_prob
        self._track_query_false_negative_prob = track_query_false_negative_prob
        self._backprop_prev_frame = backprop_prev_frame

        self._tracking = False

    def train(self, mode: bool = True):
        """Sets the module in train mode."""
        self._tracking = False
        return super().train(mode)

    def tracking(self):
        """Sets the module in tracking mode."""
        self.eval()
        self._tracking = True

    def add_track_queries_to_targets(self, targets, prev_indices, swap_prev_indices, prev_out, prev_prev_track=False, keep_all_track=False, group_object=False):
    
        device = prev_out['pred_boxes'].device

        min_prev_target_ind = min([len(prev_ind[1]) for prev_ind in prev_indices])

        rand_num = random.random()

        # We need to access the number of divisions before we can add the track queries
        for i, (target, prev_ind) in enumerate(zip(targets, prev_indices)):
            prev_out_ind, prev_target_ind = prev_ind

            if prev_prev_track or keep_all_track:
                random_subset_mask = torch.randperm(len(prev_target_ind))
            elif rand_num > 0.8:  #max number to be tracked since 
                random_subset_mask = torch.randperm(len(prev_target_ind))[:min_prev_target_ind]
            elif rand_num < 0.5 and not group_object:
                random_subset_mask = torch.randperm(len(prev_target_ind))[:0]
            else:
                random_subset_mask = torch.randperm(len(prev_target_ind))[:min_prev_target_ind - 1]

            prev_out_ind = prev_out_ind[random_subset_mask]
            prev_target_ind = prev_target_ind[random_subset_mask]

            prev_indices[i] = (prev_out_ind,prev_target_ind)

        #### TODO this two should be combined into the same block
        #### Now that FPs are added to offset number of track queries, this doesn't matter

        if prev_prev_track:
            '''
            If you are tracing cells that have just divided, they will need to be properly updated
            This will only occur when you feed the frames t-2,t-1 & t. 
            A division can be predicted from frame t-2 to t-1, then you can track those divided cells from t-1 to t
            If frame t-2 is not present, then you don't need to worry about tracking already divided cells 
            prev_prev_track signals if frame t-2 is present
            '''
            tgt_div_masks = []
            for t,target in enumerate(targets):
                '''
                    Only two consecutive frames get tracked at once.
                    This is because during cell divisions, the two daughter cells need to be dinstiguishable
                    This is a hack, but for now, I convert the labels from a current frame to a previous frame at the same time point.
                    Current frames may contain divided cells labelled with the same number but if we want to track 
                    another frame, we need to convert one of the daughter cells into a new number.

                    In the example below, frame t-1 cur and prev are the same exact image except the cells are labelled differently.
                    From frame t-2 to t-1, there is a division, however we also want to track from frame t-1 to t.
                    We need to relabel the frame t-1 cur to fit the frame t-1 prev so it can properly track to frame t.


                    frame #     t-2  t-1        t-1  t
                                prev cur       prev cur
                                 _______        _______
                                | 1 | 1 |      | 4 | 4 |    
                                |   |   |      |   | 4 |
                                | 2 | 2 |      | 5 |   |
                                |   | 2 | ---> | 6 | 5 |
                                | 3 |   |      |   | 6 |
                                |   | 3 |      | 7 |   |
                                |   |   |      |   | 7 |
                                |   |   |      |   |   |
                                |___|___|      |___|___|            
                            
                
                
                '''
                prev_cur_target = target['prev_cur_target']
                prev_output_boxes = prev_cur_target['boxes'][prev_indices[t][1]]

                prev_target = target['prev_target']
                prev_input_boxes = prev_target['boxes']
                assert torch.sum(prev_input_boxes[:,4:] != 0) == 0, 'Inputs should contain no divisions. Only outputs may contain divisions'

                prev_out_ind, prev_target_ind = prev_indices[t]
                new_prev_target_ind = torch.zeros((prev_input_boxes.shape[0]))
                tgt_div_mask = torch.zeros((prev_input_boxes.shape[0])).bool()

                count = 0
                div = 0
                for i, prev_output_box in enumerate(prev_output_boxes):

                    # Inputs do not contain divisions; Outputs can contain divisions
                    # This is why we only look at the first box for prev_input_boxes
                    box_matching = prev_output_box[:4].eq(prev_input_boxes[:,:4]).all(axis=1)
                    box_input_ind = torch.where(box_matching == True)[0]

                    box_div_matching = prev_output_box[4:].eq(prev_input_boxes[:,:4]).all(axis=1)
                    box_input_div_ind = torch.where(box_div_matching == True)[0]

                    assert len(box_input_ind) == 1, 'Cell in the output for frame t-1 needs to match at least one (two if division) in the input for frame t-1'
                    assert len(box_input_div_ind) < 2, 'Cell in the output for frame t-1 can only match to 1 cell or none in the input for frame t-1'

                    if len(box_input_div_ind) == 0: # no division

                        new_prev_target_ind[count] = box_input_ind
                        count += 1
                    elif len(box_input_div_ind) == 1: # division

                        new_prev_target_ind[count] = box_input_ind
                        new_prev_target_ind[count+1] = box_input_div_ind
                        tgt_div_mask[count] = True
                        tgt_div_mask[count+1] = True
                        count += 2
                        prev_out_ind = torch.cat((prev_out_ind[:i+div],prev_out_ind[i+div:i+div+1],prev_out_ind[i+div:]))
                        div += 1
                    else:
                        raise ValueError(f'Inconsistent tracking between prev cur and prev targets')

                assert len(prev_out_ind) == len(new_prev_target_ind)
                prev_indices[t] = (prev_out_ind,new_prev_target_ind.long())
                tgt_div_masks.append(tgt_div_mask)

        #### Should refactor the if then statement below to handle any situation

        # Due to transformers needing the same number of track queries per batch (object queries is always the same), we need to add FPs to offset divisions or number of track queries
        # Only matters if we are using the prev prev frame when a division is tracked from prev prev frame to prev frame. This is because a single decoder output embedding can predict a cell division
        if prev_prev_track or keep_all_track:
            num_cells = torch.tensor([len(prev_ind[0]) for prev_ind in prev_indices])

            max_track = max(num_cells)
            add_FPs = max_track - num_cells

            num_prev_target_ind_for_fps_list = [fp for fp in add_FPs]

        else: # currently set to add 0,1 or 2 FPs with 50% chance
            num_fps_rand = torch.randint(0,3,(1,)).item()

            if len(random_subset_mask) == 0: ### If the task is object detection only, there is no need to add FN as this is unrealistic
                num_fps_rand = 0

            num_prev_target_ind_for_fps_list = [num_fps_rand for _ in range(len(targets))]

            
        for i, (target, store_prev_ind) in enumerate(zip(targets, prev_indices)):

            prev_out_ind, prev_target_ind = store_prev_ind

            # detected prev frame tracks
            prev_track_ids = target['prev_target']['track_ids'][prev_target_ind]

            # match track ids between frames
            target_ind_match_matrix = prev_track_ids.unsqueeze(dim=1).eq(target['track_ids'])
            target_ind_matching = target_ind_match_matrix.any(dim=1)
            target_ind_matched_idx = target_ind_match_matrix.nonzero()[:, 1]
            target_ind_not_matched_idx = (1 - target_ind_match_matrix.sum(dim=0)).nonzero()[:,0]

            if group_object:
                target['group_object_gts'] = {}
                group_object_gts = target['group_object_gts']
                group_object_gts['labels'] = target['labels'].clone()
                group_object_gts['boxes'] = target['boxes'].clone()
                group_object_gts['empty'] = target['empty'].clone()

                if 'masks' in target:
                    group_object_gts['masks'] = target['masks'].clone()
                count = 0
                for b in range(len(target['boxes'])):
                    if target['boxes'][b,-1] > 0:
                        group_object_gts['boxes'] = torch.cat((target['boxes'][:b],torch.cat((target['boxes'][b,:4],torch.zeros(4,).to(device)))[None],torch.cat((target['boxes'][b,4:],torch.zeros(4,).to(device)))[None],target['boxes'][b+1:]))
                        group_object_gts['labels'] = torch.cat((target['labels'][:b],torch.cat((target['labels'][b,:1],torch.ones(1,).long().to(device)))[None],torch.cat((target['labels'][b,1:],torch.ones(1,).long().to(device)))[None],target['labels'][b+1:]))
                        
                        if 'masks' in target:
                            N,_,H,W = target['masks'].shape
                            group_object_gts['masks'] = torch.cat((target['masks'][:b],torch.cat((target['masks'][b,:1],torch.zeros((1,H,W)).to(device=device,dtype=torch.uint8)))[None],torch.cat((target['masks'][b,1:],torch.zeros((1,H,W)).to(device=device,dtype=torch.uint8)))[None],target['masks'][b+1:]))

                        count += 1

            # If there is a FN for a cell in the previous frame that divides, then the labels/boxes/masks need to be adjusted for object detection to detect the divided cells separately
            if len(target_ind_not_matched_idx) > 0:
                count = 0
                for nidx in range(len(target_ind_not_matched_idx)):
                    target_ind_not_matched_i = target_ind_not_matched_idx[nidx] + count
                    if target['boxes'][target_ind_not_matched_i][-1] > 0:
                        target['boxes'] = torch.cat((target['boxes'][:target_ind_not_matched_i],torch.cat((target['boxes'][target_ind_not_matched_i,:4],torch.zeros(4,).to(device)))[None],torch.cat((target['boxes'][target_ind_not_matched_i,4:],torch.zeros(4,).to(device)))[None],target['boxes'][target_ind_not_matched_i+1:]))
                        target['track_ids'] = torch.cat((target['track_ids'][:target_ind_not_matched_i],torch.tensor([-1]).to(device),target['track_ids'][target_ind_not_matched_i:]))
                        target['labels'] = torch.cat((target['labels'][:target_ind_not_matched_i],torch.cat((target['labels'][target_ind_not_matched_i,:1],torch.ones(1,).long().to(device)))[None],torch.cat((target['labels'][target_ind_not_matched_i,1:],torch.ones(1,).long().to(device)))[None],target['labels'][target_ind_not_matched_i+1:]))
                        
                        if 'masks' in target:
                            N,_,H,W = target['masks'].shape
                            target['masks'] = torch.cat((target['masks'][:target_ind_not_matched_i],torch.cat((target['masks'][target_ind_not_matched_i,:1],torch.zeros((1,H,W)).to(device=device,dtype=torch.uint8)))[None],torch.cat((target['masks'][target_ind_not_matched_i,1:],torch.zeros((1,H,W)).to(device=device,dtype=torch.uint8)))[None],target['masks'][target_ind_not_matched_i+1:]))

                        count += 1


                target_ind_match_matrix = prev_track_ids.unsqueeze(dim=1).eq(target['track_ids'])
                target_ind_matching = target_ind_match_matrix.any(dim=1)
                target_ind_matched_idx = target_ind_match_matrix.nonzero()[:, 1]

            # index of prev frame detection in current frame box list
            target['track_query_match_ids'] = target_ind_matched_idx

            if prev_prev_track and len(prev_out_ind) > 0:
                target['track_div_mask'] = torch.cat([
                    tgt_div_masks[i].to('cuda'),
                    torch.tensor([False, ] * (self.num_queries + num_prev_target_ind_for_fps_list[i])).to(device)
                    ]).bool()
            else:
                target['track_div_mask'] = torch.cat([
                    torch.tensor([False, ] * (len(target_ind_matching) + self.num_queries + num_prev_target_ind_for_fps_list[i])).to(device)
                    ]).bool()

            target['track_queries_TP_mask'] = torch.cat([
                target_ind_matching,
                torch.tensor([False, ] * (self.num_queries + num_prev_target_ind_for_fps_list[i])).to(device)
                ]).bool()

            # random false positives
            if num_prev_target_ind_for_fps_list[i] > 0:

                # Due to divisions, we do not want a repeat false positive which would occur with the current code below
                prev_out_ind_uni = torch.unique(prev_out_ind)
                
                prev_boxes_matched = prev_out['pred_boxes'][i,prev_out_ind_uni]

                not_prev_out_ind = torch.arange(prev_out['pred_boxes'].shape[1])
                not_prev_out_ind = [
                    ind.item()
                    for ind in not_prev_out_ind
                    if ind not in prev_out_ind_uni]

                random_false_out_ind = []

                blah = max(prev_boxes_matched.shape[0],num_prev_target_ind_for_fps_list[i])

                prev_target_ind_for_fps = torch.randperm(blah)[:num_prev_target_ind_for_fps_list[i]]

                for j in prev_target_ind_for_fps:
                    prev_boxes_unmatched = prev_out['pred_boxes'][i,not_prev_out_ind]

                    if len(prev_boxes_matched) > j:
                        prev_box_matched = prev_boxes_matched[j]
                        box_weights = \
                            prev_box_matched.unsqueeze(dim=0)[:, :2] - \
                            prev_boxes_unmatched[:, :2]
                        box_weights = box_weights[:, 0] ** 2 + box_weights[:, 0] ** 2
                        box_weights = torch.sqrt(box_weights)

                        random_false_out_idx = not_prev_out_ind.pop(
                            torch.multinomial(box_weights.cpu(), 1).item())
                    else:
                        random_false_out_idx = not_prev_out_ind.pop(torch.randperm(len(not_prev_out_ind))[0])

                    random_false_out_ind.append(random_false_out_idx)

                prev_out_ind = torch.tensor(prev_out_ind.tolist() + random_false_out_ind).long()

                target_ind_matching = torch.cat([
                    target_ind_matching,
                    torch.tensor([False, ] * len(random_false_out_ind)).bool().to(device)
                ])

            # track query masks
            track_queries_mask = torch.ones_like(target_ind_matching).bool()
            track_queries_fal_pos_mask = torch.zeros_like(target_ind_matching).bool()
            track_queries_fal_pos_mask[~target_ind_matching] = True

            # set prev frame info

            target['track_query_hs_embeds'] = prev_out['hs_embed'][i, prev_out_ind]

            boxes = prev_out['pred_boxes'].detach()[i, prev_out_ind]

            if len(swap_prev_indices[i]) > 0:
                for b in range(len(boxes)):
                    if prev_out_ind[b] in swap_prev_indices[i]:
                        boxes[b] = torch.cat((boxes[b,4:],boxes[b,:4]),dim=-1)

            assert torch.sum(boxes < 0) == 0, 'Bboxes need to have positive values'

            if prev_prev_track:
                div = 0
                for k in range(1,len(tgt_div_masks[i])):
                    if tgt_div_masks[i][k-1] and tgt_div_masks[i][k]:
                        boxes[k,:4] = boxes[k-1,4:] 
                    
            target['track_query_boxes'] = boxes[:,:4]

            assert torch.sum(target['track_query_boxes'] < 0) == 0, 'Bboxes need to have positive values'

            assert target['track_query_hs_embeds'].shape[0] + self.num_queries == len(target['track_queries_TP_mask'])
            assert target['track_query_boxes'].shape[0] + self.num_queries == len(target['track_queries_TP_mask'])

            target['track_queries_mask'] = torch.cat([
                track_queries_mask,
                torch.tensor([False, ] * self.num_queries).to(device)
            ]).bool()

            target['track_queries_fal_pos_mask'] = torch.cat([
                track_queries_fal_pos_mask,
                torch.tensor([False, ] * self.num_queries).to(device)
            ]).bool()

    def forward(self, samples: NestedTensor, targets: list = None, prev_features=None, prev_out=None):
        prev_prev_track = False
        if targets is not None and not self._tracking:
            prev_targets = [target['prev_target'] for target in targets]

            backprop_context = torch.no_grad
            if self._backprop_prev_frame:
                backprop_context = nullcontext

            with backprop_context():
                if all(['prev_prev_image' in t for t in targets]):
                    for target, prev_target in zip(targets, prev_targets):
                        prev_target['prev_target'] = target['prev_prev_target']

                    prev_prev_targets = [target['prev_prev_target'] for target in targets]
                    prev_cur_targets = [target['prev_cur_target'] for target in targets]

                    for target, prev_cur_target in zip(targets, prev_cur_targets):
                        prev_cur_target['prev_target'] = target['prev_prev_target']

                    # PREV PREV
                    prev_prev_out, _, prev_prev_features, _, _ = super().forward([t['prev_prev_image'] for t in targets])

                    prev_prev_outputs_without_aux = {
                        k: v for k, v in prev_prev_out.items() if 'aux_outputs' not in k}
                    prev_prev_indices = self._matcher(prev_prev_outputs_without_aux, prev_prev_targets)

                    prev_prev_indices, swap_prev_prev_indices = threshold_indices(prev_prev_indices,max_ind=prev_prev_out['pred_logits'].shape[1])

                    self.add_track_queries_to_targets(
                        prev_cur_targets, prev_prev_indices, swap_prev_prev_indices, prev_prev_out, keep_all_track=True)

                    prev_prev_track = True

                    # PREV
                    prev_out, _, prev_features, _, _ = super().forward(
                        [t['prev_image'] for t in targets],
                        prev_cur_targets,
                        prev_prev_features)

                    prev_outputs_without_aux = {
                        k: v for k, v in prev_out.items() if 'aux_outputs' not in k}

                    prev_indices = self._matcher(prev_outputs_without_aux, prev_cur_targets)

                    prev_indices, swap_prev_indices = threshold_indices(prev_indices,max_ind=prev_out['pred_logits'].shape[1])

                else:
                    prev_out, _, prev_features, _, _ = super().forward([t['prev_image'] for t in targets])
                    prev_prev_track = False

                    prev_outputs_without_aux = {
                        k: v for k, v in prev_out.items() if 'aux_outputs' not in k}

                    prev_indices = self._matcher(prev_outputs_without_aux, prev_targets)

                    prev_indices, swap_prev_indices = threshold_indices(prev_indices,max_ind=prev_out['pred_logits'].shape[1])

                self.add_track_queries_to_targets(targets, prev_indices, swap_prev_indices, prev_outputs_without_aux,
                                                  prev_prev_track=prev_prev_track, group_object=self.group_object)


        out, targets, features, memory, hs  = super().forward(samples, targets, prev_features, group_object=self.group_object)

        return out, targets, features, memory, hs, prev_out


# TODO: with meta classes
class DETRTracking(DETRTrackingBase, DETR):
    def __init__(self, tracking_kwargs, detr_kwargs):
        DETR.__init__(self, **detr_kwargs)
        DETRTrackingBase.__init__(self, **tracking_kwargs)


class DeformableDETRTracking(DETRTrackingBase, DeformableDETR):
    def __init__(self, tracking_kwargs, detr_kwargs):
        DeformableDETR.__init__(self, **detr_kwargs)
        DETRTrackingBase.__init__(self, **tracking_kwargs)
