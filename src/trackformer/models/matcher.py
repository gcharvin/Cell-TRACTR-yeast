# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from ..util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best
    predictions, while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1,
                 focal_loss: bool = False, focal_alpha: float = 0.25, focal_gamma: float = 2.0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates
                       in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the
                       matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.focal_loss = focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets,track=True):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the
                                classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted
                               box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target
                     is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number
                           of ground-truth objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        batch_size, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        #
        # [batch_size * num_queries, num_classes]
        if self.focal_loss:
            out_prob = outputs["pred_logits"]
            div_out_prob = torch.stack((outputs['pred_logits'][:,:,1],outputs['pred_logits'][:,:,0]),axis=-1)
            out_prob = torch.cat((out_prob,div_out_prob),axis=1)
            
            out_prob = out_prob.flatten(0,1).sigmoid()
            
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)

        # [batch_size * num_queries, 4]
        out_bbox = outputs["pred_boxes"]
        assert torch.sum(torch.isnan(out_bbox)) == 0, 'Nan in boxes before duplication'
        div_out_bbox = torch.cat((outputs['pred_boxes'][:,:,4:],outputs['pred_boxes'][:,:,:4]),axis=-1)
        out_bbox = torch.cat((out_bbox,div_out_bbox),axis=1)
        
        out_bbox = out_bbox.flatten(0,1)

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        keep = tgt_ids[:,1] == 0

        # Compute the classification cost.
        if self.focal_loss:
            neg_cost_class = (1 - self.focal_alpha) * (out_prob ** self.focal_gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = self.focal_alpha * ((1 - out_prob) ** self.focal_gamma) * (-(out_prob + 1e-8).log())

            cost_class = pos_cost_class[:,:1][:, tgt_ids[:,0]] - neg_cost_class[:,:1][:, tgt_ids[:,0]] #+ (pos_cost_class[:,1:][:, tgt_ids[:,1]] - neg_cost_class[:,1:][:, tgt_ids[:,1]])
        else:
            # # Contrary to the loss, we don't use the NLL, but approximate it in 1 - proba[target class].
            # # The 1 is a constant that doesn't change the matching, it can be ommitted.
            # cost_class = -out_prob[:, tgt_ids]

            tgt_0 = tgt_ids[:,0].repeat(out_prob.shape[0],1)
            tgt_1 = tgt_ids[:,1].repeat(out_prob.shape[0],1)

            out_prob_0 = out_prob[:,:1].repeat(1,tgt_ids.shape[0])
            out_prob_1 = out_prob[:,1:].repeat(1,tgt_ids.shape[0])

            cost_class = self.focal_alpha * ((1 - out_prob_0) ** self.focal_gamma) * -(1-tgt_0) * (out_prob_0 + 1e-8).log() - (1 - self.focal_alpha) * (out_prob_0 ** self.focal_gamma) * tgt_0 * (1 - out_prob_0 + 1e-8).log() + \
                         self.focal_alpha * ((1 - out_prob_1) ** self.focal_gamma) * -(1-tgt_1) * (out_prob_1 + 1e-8).log() - (1 - self.focal_alpha) * (out_prob_1 ** self.focal_gamma) * tgt_1 * (1 - out_prob_1 + 1e-8).log()

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox[:,:4], tgt_bbox[:,:4], p=1)
        cost_bbox[:,keep] += torch.cdist(out_bbox[:,4:], tgt_bbox[:,4:], p=1)[:,keep]

        # Checking that there are not nan in the boxes before or after convestion
        assert torch.sum(torch.isnan(out_bbox)) == 0, 'Nan in boxes before conversion'
        assert torch.sum(torch.isnan(box_cxcywh_to_xyxy(out_bbox[:,:4]))) + torch.sum(torch.isnan(box_cxcywh_to_xyxy(out_bbox[:,4:]))) == 0, 'Nan in boxes after conversion'

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox[:,:4]),box_cxcywh_to_xyxy(tgt_bbox[:,:4]))
        cost_giou[:,keep] += -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox[:,4:]),box_cxcywh_to_xyxy(tgt_bbox[:,4:]))[:,keep]

        
        # Final cost matrix
        cost_matrix = self.cost_bbox * cost_bbox \
            + self.cost_class * cost_class \
            + self.cost_giou * cost_giou
        cost_matrix = cost_matrix.view(batch_size, num_queries*2, -1).cpu()

        del cost_bbox
        del cost_class
        del cost_giou

        sizes = [len(v["boxes"]) for v in targets]

        for i, target in enumerate(targets):
            if 'track_query_match_ids' not in target:
                cost_matrix[i,cost_matrix.shape[1] // 2:] = np.inf
                continue

            prop_i = 0
            for j in range(cost_matrix.shape[1] // 2):
                # if target['track_queries_fal_pos_mask'][j] or target['track_queries_placeholder_mask'][j]:
                if target['track_queries_fal_pos_mask'][j]:
                    # false positive and palceholder track queries should not
                    # be matched to any target
                    cost_matrix[i, j] = np.inf
                    cost_matrix[i, j + cost_matrix.shape[1] // 2] = np.inf
                elif target['track_queries_mask'][j]:
                    track_query_id = target['track_query_match_ids'][prop_i]
                    prop_i += 1

                    # If the cell tracks to a cell division, we use first prediction and flipped prediction
                    if target['labels'][track_query_id,1] == 0: 
                        save_1 = torch.clone(cost_matrix[i,j,track_query_id + sum(sizes[:i])])
                        save_2 = torch.clone(cost_matrix[i,j + cost_matrix.shape[1] // 2,track_query_id + sum(sizes[:i])])

                        cost_matrix[i, :, track_query_id + sum(sizes[:i])] = np.inf

                        cost_matrix[i, j] = np.inf
                        cost_matrix[i, j, track_query_id + sum(sizes[:i])] = save_1

                        cost_matrix[i, j + cost_matrix.shape[1] // 2] = np.inf
                        cost_matrix[i, j + cost_matrix.shape[1] // 2, track_query_id + sum(sizes[:i])] = save_2
                    # If cells tracks to only one cell, we use only the first prediction
                    else:
                        cost_matrix[i, :, track_query_id + sum(sizes[:i])] = np.inf
                        cost_matrix[i, j] = np.inf
                        cost_matrix[i, j + cost_matrix.shape[1] // 2] = np.inf
                        cost_matrix[i, j, track_query_id + sum(sizes[:i])] = -1
                # New objects cannot predict cell divisions so we ignore the flipped predictions here
                else:
                    cost_matrix[i,j + cost_matrix.shape[1] // 2] = np.inf

        indices = [linear_sum_assignment(c[i])
                   for i, c in enumerate(cost_matrix.split(sizes, -1))]

        # Meant for debugging purposes only; will check when a flipped division has been matched
        if np.sum(indices[0][0] > cost_matrix.shape[1] // 2) + np.sum(indices[1][0] > cost_matrix.shape[1] // 2) > 0:
            a = 0

        for target in targets:
            if 'track_query_match_ids' not in target:
                assert np.sum(indices[0][0] >= cost_matrix.shape[1] // 2) == 0 and np.sum(indices[1][0] >= cost_matrix.shape[1] // 2) == 0

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(
        cost_class=args.set_cost_class,
        cost_bbox=args.set_cost_bbox,
        cost_giou=args.set_cost_giou,
        focal_loss=args.focal_loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,)
