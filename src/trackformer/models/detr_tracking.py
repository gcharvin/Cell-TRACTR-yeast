import random
from contextlib import nullcontext

import torch
import torch.nn as nn

from ..util import box_ops
from ..util import misc as utils
from .deformable_detr import DeformableDETR
from .detr import DETR
from .matcher import HungarianMatcher


class DETRTrackingBase(nn.Module):

    def __init__(self,
                 track_query_false_positive_prob: float = 0.0,
                 track_query_false_negative_prob: float = 0.0,
                 matcher: HungarianMatcher = None,
                 backprop_prev_frame=False,
                 dn_track = False,
                 dn_track_l1 = 0,
                 dn_track_l2 = 0,
                 dn_object = False,
                 refine_div_track_queries = False,
                 evaluate_dataset_with_no_data_aug=False,):

        self._matcher = matcher
        self._track_query_false_positive_prob = track_query_false_positive_prob
        self._track_query_false_negative_prob = track_query_false_negative_prob
        self._backprop_prev_frame = backprop_prev_frame
        self.dn_track = dn_track
        self.dn_track_l1 = dn_track_l1
        self.dn_track_l2 = dn_track_l2
        self.dn_object = dn_object
        self.refine_div_track_queries = refine_div_track_queries
        self.evaluate_dataset_with_no_data_aug = evaluate_dataset_with_no_data_aug

        if self.dn_track:
            self.dn_track_embedding = nn.Embedding(1,self.hidden_dim)

        self._tracking = False

    def train(self, mode: bool = True):
        """Sets the module in train mode."""
        self._tracking = False
        return super().train(mode)

    def tracking(self):
        """Sets the module in tracking mode."""
        self.eval()
        self._tracking = True

    def convert_prev_target(self,targets):

        for target in targets:
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
            prev_ind = target['prev_ind']

            prev_cur_target = target['prev_cur_target']
            prev_output_boxes = prev_cur_target['boxes'][prev_ind[1]]

            prev_target = target['prev_target']
            prev_input_boxes = prev_target['boxes']
            assert torch.sum(prev_input_boxes[:,4:] != 0) == 0, 'Inputs should contain no divisions. Only outputs may contain divisions'

            prev_out_ind, prev_target_ind = prev_ind
            new_prev_target_ind = torch.zeros((prev_input_boxes.shape[0]))
            tgt_div_mask = torch.zeros((prev_input_boxes.shape[0])).bool()

            tgt_div_ind = torch.zeros((prev_input_boxes.shape[0])).long()

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
                    tgt_div_ind[count] = div+1
                    tgt_div_ind[count+1] = div+1
                    count += 2
                    prev_out_ind = torch.cat((prev_out_ind[:i+div],prev_out_ind[i+div:i+div+1],prev_out_ind[i+div:]))
                    div += 1
                else:
                    raise ValueError(f'Inconsistent tracking between prev cur and prev targets')

            assert len(prev_out_ind) == len(new_prev_target_ind)
            target['prev_ind'] = [prev_out_ind,new_prev_target_ind.long()]
            target['tgt_div_masks'] = tgt_div_mask
            target['tgt_div_ind'] = tgt_div_ind

    def calc_num_FPs(self,targets,return_FPs=False,add_extra_FPs=True):

        num_cells = torch.tensor([len(target['prev_ind'][0]) for target in targets])

        max_track = max(num_cells)
        num_FPs = max_track - num_cells

        # num_FPs = torch.tensor([FP for FP in add_FPs])

        if sum(num_cells) != 0 and add_extra_FPs: # Only add FPs if it is tracking; it is not realistic to add extra FPs when no cells are being tracked
            max_FPs = max(num_FPs)
            num_fps_rand = torch.randint(0,max(3-max_FPs,1),(1,)).item()
            num_FPs += num_fps_rand 

        if return_FPs:
            return num_FPs
        else:
            for t,target in enumerate(targets):
                target['num_FPs'] = num_FPs[t]

    def add_FPs(self,target,i,prev_out):
        # Due to divisions, we do not want a repeat false positive which would occur with the current code below
        prev_out_ind_uni = torch.unique(target['prev_ind'][0])
        
        prev_boxes_matched = prev_out['pred_boxes'][i,prev_out_ind_uni]

        not_prev_out_ind = torch.arange(prev_out['pred_boxes'].shape[1])
        not_prev_out_ind = [
            ind.item()
            for ind in not_prev_out_ind
            if ind not in prev_out_ind_uni]

        random_false_out_ind = []
        # The FPs are supposed to match a real box that is being tracked; Sometimes you need more FPs than real boxes that exists
        blah = max(prev_boxes_matched.shape[0],target['num_FPs']) 

        prev_target_ind_for_fps = torch.randperm(blah)[:target['num_FPs']]

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

        target['prev_ind'][0] = torch.tensor(target['prev_ind'][0].tolist() + random_false_out_ind).long()

        target['rand_FP_mask'] = torch.cat([
            torch.zeros_like(target['target_ind_matching']).bool().to(self.device),
            torch.tensor([True, ] * len(random_false_out_ind)).bool().to(self.device)
        ])

        target['target_ind_matching'] = torch.cat([
            target['target_ind_matching'],
            torch.tensor([False, ] * len(random_false_out_ind)).bool().to(self.device)
        ])


    def update_target(self,target,index,update_track_ids=True):

        if update_track_ids:
            target['track_ids'] = torch.cat((target['track_ids'][:index],torch.tensor([-1]).to(self.device),target['track_ids'][index:]))

        target['boxes'] = torch.cat((target['boxes'][:index],torch.cat((target['boxes'][index,:4],torch.zeros(4,).to(self.device)))[None],torch.cat((target['boxes'][index,4:],torch.zeros(4,).to(self.device)))[None],target['boxes'][index+1:]))
        target['labels'] = torch.cat((target['labels'][:index],torch.cat((target['labels'][index,:1],torch.ones(1,).long().to(self.device)))[None],torch.cat((target['labels'][index,1:],torch.ones(1,).long().to(self.device)))[None],target['labels'][index+1:]))
        
        if 'masks' in target:
            N,_,H,W = target['masks'].shape
            target['masks'] = torch.cat((target['masks'][:index],torch.cat((target['masks'][index,:1],torch.zeros((1,H,W)).to(self.device,dtype=torch.uint8)))[None],torch.cat((target['masks'][index,1:],torch.zeros((1,H,W)).to(self.device,dtype=torch.uint8)))[None],target['masks'][index+1:]))

    def add_track_queries_to_targets(self, targets, prev_indices, prev_out, prev_prev_track=False, keep_all_track=False, group_object=False, dn_track=False):
    
        # min_prev_target_ind = min([len(prev_ind[1]) for prev_ind in prev_indices])

        rand_num = random.random()

        if torch.tensor([target['empty'] for target in targets]).sum() > 0:
            dn_track = False

        # We need to access the number of divisions before we can add the track queries
        for i, (target, prev_ind) in enumerate(zip(targets, prev_indices)):
            prev_out_ind, prev_target_ind = prev_ind

            if prev_prev_track or keep_all_track or self.evaluate_dataset_with_no_data_aug:
                random_subset_mask = torch.randperm(len(prev_target_ind))
            elif rand_num > 0.3:  #max number to be tracked since 
                random_subset_mask = torch.randperm(len(prev_target_ind))[:len(prev_target_ind)]
            elif rand_num < 0.2:
                random_subset_mask = torch.randperm(len(prev_target_ind))[:len(prev_target_ind)-1]
            else:
                random_subset_mask = torch.randperm(len(prev_target_ind))[:len(prev_target_ind)-2]

            target['prev_ind'] = [prev_out_ind[random_subset_mask],prev_target_ind[random_subset_mask]]

            if dn_track:
                target['dn_track'] = {}
                dn_track = target['dn_track']
                dn_track['labels'] = target['labels'].clone()
                dn_track['boxes'] = target['boxes'].clone()
                dn_track['empty'] = target['empty'].clone()
                dn_track['prev_target'] = target['prev_target']
                
                random_subset_mask = torch.randperm(len(prev_target_ind))
                dn_track['prev_ind'] = [prev_out_ind[random_subset_mask],prev_target_ind[random_subset_mask]]

                if prev_prev_track:
                    dn_track['prev_cur_target'] = target['prev_cur_target']

        #### TODO this two should be combined into the same block
        #### Now that FPs are added to offset number of track queries, this doesn't matter

        if prev_prev_track:
            '''
            If you are tracking cells that have just divided, they will need to be properly updated
            This will only occur when you feed the frames t-2,t-1 & t. 
            A division can be predicted from frame t-2 to t-1, then you can track those divided cells from t-1 to t
            If frame t-2 is not present, then you don't need to worry about tracking already divided cells 
            prev_prev_track signals if frame t-2 is present
            '''

            self.convert_prev_target(targets)

            if dn_track:
                self.convert_prev_target([target['dn_track'] for target in targets])

        # Due to transformers needing the same number of track queries per batch (object queries is always the same), we need to add FPs to offset divisions or number of track queries
        # Only matters if we are using the prev prev frame when a division is tracked from prev prev frame to prev frame. This is because a single decoder output embedding can predict a cell division
        
        self.calc_num_FPs(targets,add_extra_FPs=not self.evaluate_dataset_with_no_data_aug)

        if dn_track:
            num_FPs = self.calc_num_FPs([target['dn_track'] for target in targets],return_FPs=True)

            for i,target in enumerate(targets):
                target['dn_track']['num_FPs'] = num_FPs[i]

            
        for i, target in enumerate(targets):

            # detected prev frame tracks
            track_ids = target['prev_target']['track_ids']

            if 'dn_track' in target:
                dn_track = target['dn_track']
                prev_track_ids = track_ids[dn_track['prev_ind'][1]]

                # match track ids between frames
                target_ind_match_matrix = prev_track_ids.unsqueeze(dim=1).eq(target['track_ids'])
                dn_track['target_ind_matching'] = target_ind_match_matrix.any(dim=1)
                #dn_track['cells_leaving_mask'] = torch.cat((~target_ind_match_matrix.any(dim=1),(torch.tensor([False, ] * target['dn_track']['num_FPs'])).bool().to(self.device)))
                dn_track['track_query_match_ids'] = target_ind_match_matrix.nonzero()[:, 1]

            prev_track_ids = track_ids[target['prev_ind'][1]]

            # match track ids between frames
            target_ind_match_matrix = prev_track_ids.unsqueeze(dim=1).eq(target['track_ids'])
            target['target_ind_matching'] = target_ind_match_matrix.any(dim=1)
            target['cells_leaving_mask'] = torch.cat((~target_ind_match_matrix.any(dim=1),(torch.tensor([False, ] * target['num_FPs'])).bool().to(self.device)))
            target['track_query_match_ids'] = target_ind_match_matrix.nonzero()[:, 1]
            target_ind_not_matched_idx = (1 - target_ind_match_matrix.sum(dim=0)).nonzero()[:,0]

            if group_object:
                target['group_object'] = {}
                group_object = target['group_object']
                group_object['labels'] = target['labels'].clone()
                group_object['boxes'] = target['boxes'].clone()
                group_object['empty'] = target['empty'].clone()
                group_object['num_queries'] = self.num_queries

                if 'masks' in target:
                    group_object['masks'] = target['masks'].clone()
                count = 0
                for b in range(len(target['boxes'])):
                    if target['boxes'][b,-1] > 0:
                        self.update_target(target['group_object'],b+count,update_track_ids=False)
                        count += 1

            # If there is a FN for a cell in the previous frame that divides, then the labels/boxes/masks need to be adjusted for object detection to detect the divided cells separately
            if len(target_ind_not_matched_idx) > 0:
                count = 0
                store_div_ind = []
                for nidx in range(len(target_ind_not_matched_idx)):
                    target_ind_not_matched_i = target_ind_not_matched_idx[nidx] + count
                    if target['boxes'][target_ind_not_matched_i][-1] > 0:
                        
                        self.update_target(target,target_ind_not_matched_i)
                        store_div_ind.append(int(target_ind_not_matched_i))
                        store_div_ind.append(int(target_ind_not_matched_i+1))
                        count += 1

                target['object_detection_div_mask'] = torch.zeros(target['boxes'].shape[0]).to(self.device)

                if len(store_div_ind) > 0:
                    for s in range(len(store_div_ind) // 2):
                        target['object_detection_div_mask'][torch.tensor(store_div_ind[s*2:(s+1)*2])] = s + 1

                target_ind_match_matrix = prev_track_ids.unsqueeze(dim=1).eq(target['track_ids'])
                target['target_ind_matching'] = target_ind_match_matrix.any(dim=1)
                target['track_query_match_ids'] = target_ind_match_matrix.nonzero()[:, 1]

            # Add random false positives
            if target['num_FPs'] > 0:
                assert self.evaluate_dataset_with_no_data_aug == False
                self.add_FPs(target,i,prev_out)

            if 'dn_track' in target and dn_track['num_FPs'] > 0:
                self.add_FPs(dn_track,i,prev_out)

                if prev_prev_track and len(dn_track['prev_ind'][0]) > 0:
                    dn_track['track_div_mask'] = torch.cat([
                        dn_track['tgt_div_masks'].to('cuda'),
                        torch.tensor([False, ] * dn_track['num_FPs']).to(self.device)
                        ]).bool()

            if prev_prev_track and len(target['prev_ind'][0]) > 0:
                target['track_div_mask'] = torch.cat([
                    target['tgt_div_masks'].to('cuda'),
                    torch.tensor([False, ] * (self.num_queries + target['num_FPs'])).to(self.device)
                    ]).bool()
            else:
                target['track_div_mask'] = torch.cat([
                    torch.tensor([False, ] * (len(target['target_ind_matching']) + self.num_queries)).to(self.device)
                    ]).bool()

            target['track_queries_TP_mask'] = torch.cat([
                target['target_ind_matching'],
                torch.tensor([False, ] * (self.num_queries)).to(self.device)
                ]).bool()

            # track query masks
            track_queries_mask = torch.ones_like(target['target_ind_matching']).bool()
            track_queries_fal_pos_mask = torch.zeros_like(target['target_ind_matching']).bool()
            track_queries_fal_pos_mask[~target['target_ind_matching']] = True

            if 'dn_track' in target:
                dn_track['track_queries_mask'] = torch.ones_like(dn_track['target_ind_matching']).to(self.device).bool()
                dn_track['track_queries_fal_pos_mask'] = torch.zeros_like(dn_track['target_ind_matching']).to(self.device).bool()
                dn_track['track_queries_fal_pos_mask'][~dn_track['target_ind_matching']] = True      

                # dn_track['track_query_hs_embeds'] = prev_out['hs_embed'][i, dn_track['prev_ind'][0]]
                dn_track['track_query_hs_embeds'] = self.dn_track_embedding.weight.repeat(len(dn_track['prev_ind'][0]),1)

                boxes = prev_out['pred_boxes'].detach()[i, dn_track['prev_ind'][0]]

                assert torch.sum(boxes < 0) == 0, 'Bboxes need to have positive values'

                if prev_prev_track:
                    for k in range(1,len(dn_track['tgt_div_masks'])):
                        if dn_track['tgt_div_masks'][k-1] and dn_track['tgt_div_masks'][k]:
                            boxes[k,:4] = boxes[k-1,4:] 
                        
                boxes = boxes[:,:4]

                dn_track['track_query_boxes_gt'] = boxes.clone()

                l_1 = self.dn_track_l1
                l_2 = self.dn_track_l2

                noised_boxes = torch.rand_like(boxes) * 2 - 1
                boxes[..., :2] += boxes[..., 2:] * noised_boxes[..., :2] * l_1
                boxes[..., 2:] *= 1 + l_2 * noised_boxes[..., 2:]
                boxes = torch.clamp(boxes,0,1)

                dn_track['track_query_boxes'] = boxes
                dn_track['num_queries'] = boxes.shape[0]

                assert torch.sum(dn_track['track_query_boxes'] < 0) == 0, 'Bboxes need to have positive values'

                assert dn_track['track_query_hs_embeds'].shape[0] == len(dn_track['target_ind_matching'])
                assert dn_track['track_query_boxes'].shape[0] == len(dn_track['target_ind_matching'])

            # set prev frame info

            target['track_query_hs_embeds'] = prev_out['hs_embed'][i, target['prev_ind'][0]] 

            boxes = prev_out['pred_boxes'].detach()[i, target['prev_ind'][0]]

            # if len(swap_prev_indices[i]) > 0:
            #     for b in range(len(boxes)):
            #         if target['prev_ind'][0][b] in swap_prev_indices[i]:
            #             boxes[b] = torch.cat((boxes[b,4:],boxes[b,:4]),dim=-1)

            assert torch.sum(boxes < 0) == 0, 'Bboxes need to have positive values'

            if prev_prev_track:
                for k in range(1,len(target['tgt_div_ind'])):
                    if target['tgt_div_ind'][k-1] > 0 and target['tgt_div_ind'][k-1] == target['tgt_div_ind'][k]:
                        boxes[k,:4] = boxes[k-1,4:] 
                    
            target['track_query_boxes'] = boxes[:,:4]

            assert torch.sum(target['track_query_boxes'] < 0) == 0, 'Bboxes need to have positive values'

            assert target['track_query_hs_embeds'].shape[0] + self.num_queries == len(target['track_queries_TP_mask'])
            assert target['track_query_boxes'].shape[0] + self.num_queries == len(target['track_queries_TP_mask'])

            target['track_queries_mask'] = torch.cat([
                track_queries_mask,
                torch.tensor([False, ] * self.num_queries).to(self.device)
            ]).bool()

            target['track_queries_fal_pos_mask'] = torch.cat([
                track_queries_fal_pos_mask,
                torch.tensor([False, ] * self.num_queries).to(self.device)
            ]).bool()

    def forward(self, samples: utils.NestedTensor, targets: list = None, prev_features=None, prev_out=None, tm_threshold=None, evaluate_track=False):     
        if targets is not None and (not self._tracking or self.evaluate_dataset_with_no_data_aug):
            prev_targets = [target['prev_target'] for target in targets]

            prev_prev_track = False

            if tm_threshold is None:
                tm_threshold = 0.25

            random_nb = random.random()

            if random_nb < tm_threshold:
                track_cells = False
            else:
                track_cells = True

            if track_cells:

                backprop_context = torch.no_grad
                if self._backprop_prev_frame:
                    backprop_context = nullcontext

                with backprop_context():
                    if False:
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

                        prev_indices, prev_targets = utils.threshold_indices(prev_indices,prev_targets,max_ind=prev_out['pred_logits'].shape[1])

                        N = prev_out['pred_logits'].shape[1]

                        for i, (target,(prev_ind_out,prev_ind_tgt)) in enumerate(zip(targets,prev_indices)):
                    
                            prev_target = target['prev_target']
                            prev_prev_target = target['prev_prev_target']
                            prev_cur_target = target['prev_cur_target']

                            # if 'object_detection_div_mask' in target:
                            prev_ind_out_clone = prev_ind_out.clone()
                            prev_ind_tgt_clone = prev_ind_tgt.clone()
                            prev_boxes_clone = prev_target['boxes'].clone()
                            skip = []
                            prev_ind_tgt_blah = prev_ind_tgt.clone()
                            prev_pred_boxes = prev_out['pred_boxes'][i].detach()

                            # we are cehcking to see if the model predicts a single cell instead of two cells (pre-req - must have just divided)
                            for idx in range(len(prev_ind_tgt_clone)):

                                prev_ind_tgt_box_1 = prev_ind_tgt_clone[idx]

                                prev_tgt_box_1 = prev_boxes_clone[prev_ind_tgt_box_1][:4] # these are inputs so all pred_boxes will contain one box

                                if prev_ind_tgt_box_1 in skip:
                                    continue

                                prev_cur_boxes = prev_cur_target['boxes']
                                # cell could have just divided so need to check first and second box
                                box_eq = torch.add(prev_tgt_box_1[:4].eq(prev_cur_boxes[:,:4]).all(axis=-1),prev_tgt_box_1[:4].eq(prev_cur_boxes[:,4:]).all(axis=-1))

                                if box_eq.sum() == 0:
                                    continue

                                prev_box_ind_match_id = box_eq.nonzero()[0][0]

                                if prev_cur_boxes[prev_box_ind_match_id][-1] > 0:
                                    prev_prev_track_id = prev_cur_target['track_ids'][prev_box_ind_match_id]
                                    prev_prev_box_ind = (prev_prev_target['track_ids'] == prev_prev_track_id).nonzero()[0][0]
                                    prev_prev_box = prev_prev_target['boxes'][prev_prev_box_ind]

                                    div_boxes = prev_cur_boxes[prev_box_ind_match_id]

                                    combined_box = utils.combine_div_boxes(div_boxes,prev_prev_box)

                                    if prev_tgt_box_1[:4].eq(prev_cur_boxes[:,:4]).all(axis=-1).sum() == 1:
                                        prev_tgt_box_2 = div_boxes[4:]
                                    else:
                                        prev_tgt_box_2 = div_boxes[:4]

                                    prev_ind_tgt_box_2 = prev_tgt_box_2[:4].eq(prev_boxes_clone[:,:4]).all(axis=-1).nonzero()[0][0].to('cpu')
                                    prev_ind_box_2 = (prev_ind_tgt_clone == prev_ind_tgt_box_2).nonzero()[0][0]

                                    skip.append(prev_ind_tgt_box_2)

                                    prev_ind_out_box_1 = prev_ind_out_clone[idx]
                                    prev_ind_out_box_2 = prev_ind_out_clone[prev_ind_box_2]

                                    prev_pred_box_1 = prev_pred_boxes[prev_ind_out_box_1,:4]
                                    prev_pred_box_2 = prev_pred_boxes[prev_ind_out_box_2,:4]
                                    
                                    iou = utils.calc_iou(torch.cat((prev_pred_box_1,prev_pred_box_2)),div_boxes)

                                    unused_object_query_indices = torch.tensor([prev_ind_out_box_1,prev_ind_out_box_2] + [oq_id for oq_id in torch.arange(N-self.num_queries,N) if (oq_id not in prev_ind_out and prev_out['pred_logits'][i,oq_id,0].sigmoid().detach() > 0.5)])

                                    unused_pred_boxes = prev_pred_boxes[unused_object_query_indices].detach() 

                                    iou_combined = box_ops.generalized_box_iou(box_ops.box_cxcywh_to_xyxy(unused_pred_boxes[:,:4]),box_ops.box_cxcywh_to_xyxy(combined_box[None,:4]),return_iou_only=True)

                                    max_ind = torch.argmax(iou_combined)

                                    assert iou_combined[max_ind] <= 1 and iou_combined[max_ind] >= 0 and iou <= 1 and iou >= 0

                                    if iou_combined[max_ind] - iou > 0:

                                        query_id = unused_object_query_indices[max_ind]
                                        
                                        if max_ind == 1:
                                            prev_ind_out_box_1,prev_ind_out_box_2 = prev_ind_out_box_2,prev_ind_out_box_1
                                            prev_ind_tgt_box_1,prev_ind_tgt_box_2 = prev_ind_tgt_box_2,prev_ind_tgt_box_1
                                        
                                        prev_ind_out[prev_ind_out == prev_ind_out_box_1] = query_id

                                        assert (prev_ind_out == prev_ind_out_box_2).sum() == 1

                                        prev_ind_out = prev_ind_out[prev_ind_out != prev_ind_out_box_2]                                

                                        prev_ind_tgt_box_1_new = prev_ind_tgt[(prev_ind_tgt_blah == prev_ind_tgt_box_1).nonzero()[0][0]]
                                        prev_ind_tgt_box_2_new = prev_ind_tgt[(prev_ind_tgt_blah == prev_ind_tgt_box_2).nonzero()[0][0]]

                                        prev_ind_tgt = prev_ind_tgt[prev_ind_tgt != prev_ind_tgt_box_2_new]
                                        prev_ind_tgt[prev_ind_tgt > prev_ind_tgt_box_2_new] = prev_ind_tgt[prev_ind_tgt > prev_ind_tgt_box_2_new] - 1

                                        assert len(prev_ind_tgt) + 1 == len(prev_ind_tgt_blah)

                                        prev_ind_tgt_blah = prev_ind_tgt_blah[prev_ind_tgt_blah != prev_ind_tgt_box_2]


                                        track_ids_div = prev_target['track_ids'][torch.tensor([prev_ind_tgt_box_1_new,prev_ind_tgt_box_2_new])]

                                        if track_ids_div[0] in target['track_ids'] and track_ids_div[1] in target['track_ids']: # if cell 1 and 2 are still present, we need to treat this as a division

                                            cur_ind_tgt_box_1 = (target['track_ids'] == track_ids_div[0]).nonzero()[0][0]
                                            cur_ind_tgt_box_2 = (target['track_ids'] == track_ids_div[1]).nonzero()[0][0]

                                            assert target['boxes'][cur_ind_tgt_box_1][-1] == 0 and target['boxes'][cur_ind_tgt_box_2][-1] == 0

                                            target['boxes'][cur_ind_tgt_box_1] = torch.cat((target['boxes'][cur_ind_tgt_box_1,:4],target['boxes'][cur_ind_tgt_box_2,:4]))
                                            target['labels'][cur_ind_tgt_box_1] = torch.tensor([0,0]).to(self.device)
                                            target['track_ids'][cur_ind_tgt_box_1] = prev_target['track_ids'][prev_ind_tgt_box_1_new]

                                            cur_ind_keep = torch.tensor([cur_ind_tgt_box for cur_ind_tgt_box in range(len(target['boxes'])) if cur_ind_tgt_box != cur_ind_tgt_box_2])
                                            target['boxes'] = target['boxes'][cur_ind_keep]
                                            target['labels'] = target['labels'][cur_ind_keep]
                                            target['track_ids'] = target['track_ids'][cur_ind_keep]

                                        elif track_ids_div[0] in target['track_ids']: # if cell 2 left the chamber but cell 1 is still present

                                            cur_ind_tgt_box_1 = (target['track_ids'] == track_ids_div[0]).nonzero()[0][0]

                                            assert target['boxes'][cur_ind_tgt_box_1][-1] == 0 
                                            # Don't need to update target boxes or labels here
                                            target['track_ids'][cur_ind_tgt_box_1] = prev_target['track_ids'][prev_ind_tgt_box_1_new]

                                        elif track_ids_div[1] in target['track_ids']: # if cell 1 left the chamber but cell 2 is still present
    
                                            cur_ind_tgt_box_2 = (target['track_ids'] == track_ids_div[1]).nonzero()[0][0]

                                            assert target['boxes'][cur_ind_tgt_box_2][-1] == 0 
                                            # Don't need to update target boxes or labels here
                                            target['track_ids'][cur_ind_tgt_box_2] = prev_target['track_ids'][prev_ind_tgt_box_1_new]

                                        else: # This means both cells left chamber in next frame; nothing to be done since the modified cell(s) in the previous frame tracks to nothing 
                                            pass

                                        prev_target['boxes'][prev_ind_tgt_box_1_new] = combined_box
                                        prev_ind_tgt_boxes = torch.tensor([prev_ind_tgt_box for prev_ind_tgt_box in range(len(prev_target['boxes'])) if prev_ind_tgt_box != prev_ind_tgt_box_2_new])
                                        prev_target['boxes'] = prev_target['boxes'][prev_ind_tgt_boxes]
                                        prev_target['labels'] = prev_target['labels'][prev_ind_tgt_boxes]
                                        prev_target['track_ids'] = prev_target['track_ids'][prev_ind_tgt_boxes]
                                        assert len(prev_ind_out) == len(prev_ind_tgt)

                                        prev_indices[i] = (prev_ind_out,prev_ind_tgt)

                                        if 'masks' in target:
                                            raise NotImplementedError

                            test_early_div = (target['boxes'][:,-1] > 0).any()
                            # we are checking to see if the model predicted two separate cells (object detection instead of one cell) - pre-req: cell must divide in next frame (current frame)
                            if test_early_div: # quick check to see if any cells divided in the current frame (we are accessing object detection in the prev frame)
                                prev_ind_out_clone = prev_ind_out.clone()
                                prev_ind_tgt_clone = prev_ind_tgt.clone()
                                prev_boxes_clone = prev_target['boxes'].clone()
                                prev_track_ids_clone = prev_target['track_ids'].clone()
                                skip = [] 
                                for idx in range(len(prev_ind_tgt_clone)):

                                    prev_ind_out_box = prev_ind_out_clone[idx]
                                    prev_ind_tgt_box = prev_ind_tgt_clone[idx]

                                    if prev_ind_tgt_box in skip:
                                        continue

                                    prev_track_id = prev_track_ids_clone[prev_ind_tgt_box]

                                    # Cell needs to track to next frame (current frame) to check if it divided or not
                                    if prev_track_id not in target['track_ids']:
                                        continue

                                    cur_ind_tgt_box = (target['track_ids'] == prev_track_id).nonzero()[0][0]
                                    cur_tgt_box = target['boxes'][cur_ind_tgt_box].clone()

                                    # Cell in next frame (current frame) needs to divide otherwise we won't consider an early division
                                    if cur_tgt_box[-1] == 0:
                                        continue

                                    prev_box = prev_boxes_clone[prev_ind_tgt_box]

                                    div_box = utils.divide_box(prev_box,cur_tgt_box)

                                    assert N == self.num_queries
                                    unused_object_query_indices = torch.tensor([prev_ind_out_box] + [oq_id for oq_id in torch.arange(self.num_queries) if (oq_id not in prev_ind_out and prev_out['pred_logits'][i,oq_id,0].detach() > 0.5)])
                        
                                    if len(unused_object_query_indices) > 1:

                                        unused_prev_pred_boxes = prev_pred_boxes[unused_object_query_indices]

                                        iou_div_all = box_ops.generalized_box_iou(
                                                                                  box_ops.box_cxcywh_to_xyxy(unused_prev_pred_boxes[:,:4]),
                                                                                  box_ops.box_cxcywh_to_xyxy(torch.cat((div_box[None,:4],div_box[None,4:]),axis=0)),
                                                                                  return_iou_only=True
                                                                                  )

                                        match_ind = torch.argmax(iou_div_all,axis=0).to('cpu')

                                        if len(torch.unique(match_ind)) == 2: # If same cell matches best to the target box; then discard
                                            selected_pred_boxes = unused_prev_pred_boxes[match_ind,:4]
                                            iou_div = utils.calc_iou(div_box, torch.cat((selected_pred_boxes[0],selected_pred_boxes[1])))

                                            prev_pred_box = prev_pred_boxes[prev_ind_out_box]
                                            prev_pred_box[4:] = 0

                                            iou = utils.calc_iou(prev_box,prev_pred_box)

                                            if iou_div - iou > 0:
                                                prev_target['boxes'][prev_ind_tgt_box] = torch.cat((div_box[:4],torch.zeros_like(div_box[:4])))
                                                prev_target['boxes'] = torch.cat((prev_target['boxes'],torch.cat((div_box[4:],torch.zeros_like(div_box[:4])))[None]))

                                                # prev_target['labels'][prev_ind_tgt_box] = torch.tensor([0,1]).to(self.device) # inputs --> so no divisions here
                                                prev_target['labels'] = torch.cat((prev_target['labels'],torch.tensor([0,1])[None,].to(self.device)))

                                                new_track_id = torch.max((torch.cat((prev_target['track_ids'],target['track_ids'])))) + 1
                                                prev_target['track_ids'] = torch.cat((prev_target['track_ids'],torch.tensor([new_track_id]).to(self.device)))

                                                target['track_ids'] = torch.cat((target['track_ids'],torch.tensor([new_track_id]).to(self.device)))
                                                target['labels'][cur_ind_tgt_box] = torch.tensor([0,1]).to(self.device) # division so we need to change this since division is now happening a frame early
                                                target['labels'] = torch.cat((target['labels'],torch.tensor([0,1])[None].to(self.device)))
                                                target['boxes'] = torch.cat((target['boxes'],torch.cat((cur_tgt_box[4:],torch.zeros_like(cur_tgt_box[4:])))[None]))
                                                target['boxes'][cur_ind_tgt_box] = torch.cat((cur_tgt_box[:4],torch.zeros_like(cur_tgt_box[:4])))


                                                if 'masks' in target:
                                                    raise NotImplementedError

                                                prev_ind_out[prev_ind_out == prev_ind_out_box] = unused_object_query_indices[match_ind[0]]
                                                prev_ind_out = torch.cat((prev_ind_out,unused_object_query_indices[match_ind[1]][None]))
                                                prev_ind_tgt = torch.cat((prev_ind_tgt,torch.tensor([prev_target['boxes'].shape[0]-1])))                    

                                                assert len(prev_ind_out) == len(prev_ind_tgt)
                                                assert len(prev_target['boxes']) == len(prev_target['labels'])

                                                prev_indices[i] = (prev_ind_out,prev_ind_tgt)

                    self.add_track_queries_to_targets(targets, prev_indices, prev_outputs_without_aux,
                                                    prev_prev_track=prev_prev_track,group_object=self.group_object,dn_track=self.dn_track)

            else: # if we are perforoming object detection, we need to make sure the ground truths are all single cells as divided cells will be grouped as one
                # print('object detection')
                for target in targets:
                    
                    count = 0
                    store_div_ind = []
                    for b in range(len(target['boxes'])):
                        if target['boxes'][b + count,-1] > 0:
                            self.update_target(target,b+count)
                            store_div_ind.append(b+count)
                            store_div_ind.append(b+count+1)
                            count += 1

                    target['object_detection_div_mask'] = torch.zeros(target['boxes'].shape[0]).to(self.device)

                    if len(store_div_ind) > 0:
                        for s in range(len(store_div_ind) // 2):
                            target['object_detection_div_mask'][torch.tensor(store_div_ind[s*2:(s+1)*2])] = s + 1

                    target['track_queries_mask'] = torch.zeros((self.num_queries)).bool().to(self.device)

                    

                    assert (target['boxes'][:,-1] == 0).all()

        out, targets, features, memory, hs  = super().forward(samples, targets, prev_features, group_object=self.group_object, dn_object=self.dn_object)

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
