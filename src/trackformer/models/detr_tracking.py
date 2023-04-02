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
                 matcher: HungarianMatcher = None,
                 backprop_prev_frame=False,
                 dn_track = False,
                 dn_track_l1 = 0,
                 dn_track_l2 = 0,
                 dn_object = False,
                 dn_enc=False,
                 refine_div_track_queries = False,
                 no_data_aug=False,
                 epoch_to_start_using_flexible_divisions=10,
                 use_prev_prev_frame=False,
                 dn_track_add_object_queries=False,
                 object_detection_only = False,
                 num_queries = 30,):

        self._matcher = matcher
        self._backprop_prev_frame = backprop_prev_frame
        self.num_queries = num_queries
        self.dn_track = dn_track
        self.dn_track_l1 = dn_track_l1
        self.dn_track_l2 = dn_track_l2
        self.dn_object = dn_object
        self.dn_enc = dn_enc
        self.refine_div_track_queries = refine_div_track_queries
        self.no_data_aug = no_data_aug
        self.object_detection_only = object_detection_only
        self.copy_dict_keys = ['labels','boxes','masks','track_ids','flexible_divisions','empty','framenb','labels_orig','boxes_orig','masks_orig','track_ids_orig','flexible_divisions_orig']
        self.eval_prev_prev_frame = False

        self.epoch_to_start_using_flexible_divisions = epoch_to_start_using_flexible_divisions
        self.use_prev_prev_frame = use_prev_prev_frame
        self.dn_track_add_object_queries = dn_track_add_object_queries

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

    def calc_num_FPs(self,targets,dict_names,add_extra_FPs=True):

        for dict_name in dict_names:
            # Number of cells being tracked per batch
            num_cells = torch.tensor([len(target[dict_name]['prev_ind'][0]) for target in targets])

            # Calculate the FPs necessary for batching to work. There needs to be an equal amount of track queries per batch so FPs are added to offset a sample with less cells
            max_track = max(num_cells)
            num_FPs = max_track - num_cells

            # Only add FPs if it is tracking; it is not realistic to add extra FPs when no cells are being tracked
            if sum(num_cells) != 0 and add_extra_FPs: 
                # If 3 or more FPs are being added then we do not add more as we don't want to litter the sample with FPs causing an unrealistic sample and waste computational resources
                num_fps_rand = torch.randint(0,max(3-max(num_FPs),1),(1,)).item()
                num_FPs += num_fps_rand 

            for t,target in enumerate(targets):
                target[dict_name]['num_FPs'] = num_FPs[t]

    def get_FP_boxes(self,target,i,prev_out):
        # Due to divisions, we do not want a repeat false positive which would occur with the current code below
        # With divisions, target['prev_ind][0] can point to the 
        prev_out_ind_uni = torch.unique(target['prev_ind'][0])
        random_subset_mask = torch.randperm(len(prev_out_ind_uni))
        prev_out_ind_uni = prev_out_ind_uni[random_subset_mask]

        not_prev_out_ind = torch.arange(prev_out['pred_boxes'].shape[1])
        not_prev_out_ind = [ind.item() for ind in not_prev_out_ind if ind not in prev_out_ind_uni]

        random_false_out_ind = []
        FP_boxes = []

        for j in range(target['num_FPs']):
            if j < len(prev_out_ind_uni):
                box = prev_out['pred_boxes'][i,prev_out_ind_uni[j]].clone()
                box[:4] = utils.add_noise_to_boxes(box[:4],l_1=0.4,l_2=0.2)
            else:
                if j < len(not_prev_out_ind):
                    box = prev_out['pred_boxes'][i,not_prev_out_ind[j]].clone()
                else:
                    box = torch.zeros((8)).to(self.device)
            
            if j < len(not_prev_out_ind):
                random_false_out_ind.append(not_prev_out_ind[j])
            else:
                random_false_out_ind.append(int(torch.randint(0,len(prev_out['pred_boxes']),(1,))[0]))

            FP_boxes.append(box)

        FP_boxes = torch.stack(FP_boxes)

        target['prev_ind'][0] = torch.tensor(target['prev_ind'][0].tolist() + random_false_out_ind).long()

        return FP_boxes

    def update_target(self,target,index):

        cur_target = target['cur_target']
        man_track = target['man_track']

        track_id = cur_target['track_ids'][index]
        track_id_1, track_id_2 = man_track[man_track[:,-1] == track_id,0]

        track_id_1_orig = cur_target['track_ids_orig'][cur_target['boxes'][index,:4].eq(cur_target['boxes_orig'][:,:4]).all(-1)][0]

        if track_id_1 != track_id_1_orig:
            track_id_1, track_id_2 = track_id_2, track_id_1

        cur_target['track_ids'] = torch.cat((cur_target['track_ids'][:index],torch.tensor([track_id_1]).to(self.device),torch.tensor([track_id_2]).to(self.device),cur_target['track_ids'][index+1:]))
        cur_target['boxes'] = torch.cat((cur_target['boxes'][:index],torch.cat((cur_target['boxes'][index,:4],torch.zeros(4,).to(self.device)))[None],torch.cat((cur_target['boxes'][index,4:],torch.zeros(4,).to(self.device)))[None],cur_target['boxes'][index+1:]))
        cur_target['labels'] = torch.cat((cur_target['labels'][:index],torch.tensor([0,1]).long().to(self.device)[None],torch.tensor([0,1]).long().to(self.device)[None],cur_target['labels'][index+1:]))
        cur_target['flexible_divisions'] = torch.cat((cur_target['flexible_divisions'][:index],torch.tensor([False]).to(self.device),cur_target['flexible_divisions'][index:]))
        
        if 'masks' in cur_target:
            N,_,H,W = cur_target['masks'].shape
            cur_target['masks'] = torch.cat((cur_target['masks'][:index],torch.cat((cur_target['masks'][index,:1],torch.zeros((1,H,W)).to(self.device,dtype=torch.uint8)))[None],torch.cat((cur_target['masks'][index,1:],torch.zeros((1,H,W)).to(self.device,dtype=torch.uint8)))[None],cur_target['masks'][index+1:]))


    def add_dn_track_queries_to_prev_targets(self, targets):

        for target in targets:
            random_subset_mask = torch.randperm(len(target['prev_prev_target']['boxes']))
            target['prev_target']['prev_ind'] = [random_subset_mask,random_subset_mask]

        self.calc_num_FPs(targets,['prev_target'],add_extra_FPs=False)

        for i, target in enumerate(targets):

            prev_prev_target = target['prev_prev_target']
            prev_target = target['prev_target']

            # detected prev frame tracks
            track_ids = prev_prev_target['track_ids']

            prev_track_ids = track_ids[prev_target['prev_ind'][1]]

            # match track ids between frames
            prev_target_ind_match_matrix = prev_track_ids.unsqueeze(dim=1).eq(prev_target['track_ids'])
            prev_target['target_ind_matching'] = prev_target_ind_match_matrix.any(dim=1)
            prev_target['track_query_match_ids'] = prev_target_ind_match_matrix.nonzero()[:, 1]

            boxes = prev_prev_target['boxes'].clone().detach()[prev_target['prev_ind'][0],:4]

            if prev_target['num_FPs'] > 0:
                boxes = torch.cat((boxes,torch.zeros((prev_target['num_FPs'],4)).to(self.device)))
                start = len(boxes) - prev_target['num_FPs']
                end = len(boxes)
                prev_target['prev_ind'][0] = torch.tensor(prev_target['prev_ind'][0].tolist() + [new_ind for new_ind in range(start,end)]).long()
                prev_target['target_ind_matching'] = torch.cat([prev_target['target_ind_matching'],torch.tensor([False, ] * prev_target['num_FPs']).bool().to(self.device)])

            prev_target['track_queries_fal_pos_mask'] = torch.zeros_like(prev_target['target_ind_matching']).to(self.device).bool()
            prev_target['track_queries_fal_pos_mask'][~prev_target['target_ind_matching']] = True      

            prev_target['track_queries_mask'] = torch.cat([
                torch.ones_like(prev_target['target_ind_matching']).to(self.device).bool(),
                torch.tensor([False, ] * self.num_queries).to(self.device)
            ]).bool()

            prev_target['track_queries_TP_mask'] = torch.cat([
                prev_target['target_ind_matching'],
                torch.tensor([False, ] * self.num_queries).to(self.device)
            ]).bool()

            prev_target['track_queries_fal_pos_mask'] = torch.cat([
                prev_target['track_queries_fal_pos_mask'],
                torch.tensor([False, ] * self.num_queries).to(self.device)
            ]).bool()

            prev_target['track_query_hs_embeds'] = self.dn_track_embedding.weight.repeat(len(prev_target['prev_ind'][0]),1)

            assert torch.sum(boxes < 0) == 0, 'Bboxes need to have positive values'

            prev_target['track_query_boxes'] = boxes
            prev_target['num_queries'] = boxes.shape[0]

            assert torch.sum(prev_target['track_query_boxes'] < 0) == 0, 'Bboxes need to have positive values'
            assert prev_target['track_query_hs_embeds'].shape[0] == len(prev_target['target_ind_matching'])
            assert prev_target['track_query_boxes'].shape[0] == len(prev_target['target_ind_matching'])

    def get_random_mask(self,targets,prev_indices):

        rand_num = torch.rand(1)[0]

        # We need to access the number of divisions before we can add the track queries
        for target, prev_ind in zip(targets, prev_indices):
            prev_out_ind, prev_target_ind = prev_ind

            if self.no_data_aug or self.prev_prev_track:
                random_subset_mask = torch.arange(len(prev_target_ind))
            elif rand_num > 0.5:  # max number to be tracked since 
                random_subset_mask = torch.randperm(len(prev_target_ind))[:len(prev_target_ind)]
            elif rand_num < 0.3:
                random_subset_mask = torch.randperm(len(prev_target_ind))[:max(len(prev_target_ind)-1,1)]
            else:
                random_subset_mask = torch.randperm(len(prev_target_ind))[:max(len(prev_target_ind)-2,1)]

            target['cur_target']['prev_ind'] = [prev_out_ind[random_subset_mask],prev_target_ind[random_subset_mask]]

            if self.dn_track and torch.tensor([target['cur_target']['empty'] for target in targets]).sum() == 0:

                target['dn_track'] = {}
                for copy_dict_key in self.copy_dict_keys: # ['labels','boxes','masks','track_ids','empty']
                    if copy_dict_key in target['cur_target']:
                        target['dn_track'][copy_dict_key] = target['cur_target'][copy_dict_key].clone()

                target['dn_track']['prev_target'] = target['prev_target']
                target['dn_track']['fut_target'] = target['fut_target']
                
                random_subset_mask = torch.randperm(len(prev_target_ind))
                target['dn_track']['prev_ind'] = [prev_out_ind[random_subset_mask],prev_target_ind[random_subset_mask]]

    def add_track_queries_to_targets(self, targets, prev_out, add_object_queries_to_dn_track=False):
    
        # Due to transformers needing the same number of track queries per batch (object queries is always the same), we need to add FPs to offset divisions or number of track queries
        # Only matters if we are using the prev prev frame when a division is tracked from prev prev frame to prev frame. This is because a single decoder output embedding can predict a cell division
        
        dict_names = ['cur_target']

        if 'dn_track' in targets[0]:
            dict_names += ['dn_track']

        self.calc_num_FPs(targets,dict_names,add_extra_FPs=not self.no_data_aug and not self.prev_prev_track)

        for i, target in enumerate(targets):

            # detected prev frame tracks
            track_ids = target['prev_target']['track_ids']

            if 'dn_track' in target:
                dn_track = target['dn_track']
                prev_track_ids = track_ids[dn_track['prev_ind'][1]]

                # match track ids between frames
                target_ind_match_matrix = prev_track_ids.unsqueeze(dim=1).eq(dn_track['track_ids'])
                dn_track['target_ind_matching'] = target_ind_match_matrix.any(dim=1)
                dn_track['cells_leaving_mask'] = torch.cat((~target_ind_match_matrix.any(dim=1),(torch.tensor([False, ] * dn_track['num_FPs'])).bool().to(self.device)))
                dn_track['track_query_match_ids'] = target_ind_match_matrix.nonzero()[:, 1]

                if not (self.dn_track_add_object_queries and add_object_queries_to_dn_track):
                    # For cells that enter the frame, we need to drop these cells during denoised training because no object queries are used. so the matcher will fail
                    # Unless object queries are add to dn_track, we need to get rid of them
                    new_cell_indices = sorted((target_ind_match_matrix.sum(0) == 0).nonzero(),reverse=True)

                    for new_cell_ind in new_cell_indices:

                        if new_cell_ind == dn_track['boxes'].shape[0] - 1:
                            dn_track['boxes'] = dn_track['boxes'][:new_cell_ind]
                            dn_track['labels'] = dn_track['labels'][:new_cell_ind]
                            dn_track['track_ids'] = dn_track['track_ids'][:new_cell_ind]

                            if 'masks' in dn_track:
                                dn_track['masks'] = dn_track['masks'][:new_cell_ind]

                        else:
                            dn_track['boxes'] = torch.cat((dn_track['boxes'][:new_cell_ind], dn_track['boxes'][new_cell_ind+1:]),axis=0)
                            dn_track['labels'] = torch.cat((dn_track['labels'][:new_cell_ind], dn_track['labels'][new_cell_ind+1:]),axis=0)
                            dn_track['track_ids'] = torch.cat((dn_track['track_ids'][:new_cell_ind], dn_track['track_ids'][new_cell_ind+1:]),axis=0)

                            if 'masks' in dn_track:
                                dn_track['masks'] = torch.cat((dn_track['masks'][:new_cell_ind], dn_track['masks'][new_cell_ind+1:]),axis=0)

                        target_ind_match_matrix = prev_track_ids.unsqueeze(dim=1).eq(dn_track['track_ids'])
                        dn_track['target_ind_matching'] = target_ind_match_matrix.any(dim=1)
                        dn_track['cells_leaving_mask'] = torch.cat((~target_ind_match_matrix.any(dim=1),(torch.tensor([False, ] * dn_track['num_FPs'])).bool().to(self.device)))
                        dn_track['track_query_match_ids'] = target_ind_match_matrix.nonzero()[:, 1]

            prev_track_ids = track_ids[target['cur_target']['prev_ind'][1]]

            # match track ids between frames
            target_ind_match_matrix = prev_track_ids.unsqueeze(dim=1).eq(target['cur_target']['track_ids'])
            target['cur_target']['target_ind_matching'] = target_ind_match_matrix.any(dim=1)
            target['cur_target']['track_query_match_ids'] = target_ind_match_matrix.nonzero()[:, 1]
            target_ind_not_matched_idx = (1 - target_ind_match_matrix.sum(dim=0)).nonzero()[:,0] # cells in cur_target that don't match to anything in prev_traget (could be FN or cell entering frame)

            # If there is a FN for a cell in the previous frame that divides, then the labels/boxes/masks need to be adjusted for object detection to detect the divided cells separately
            if len(target_ind_not_matched_idx) > 0:
                count = 0
                for nidx in range(len(target_ind_not_matched_idx)):
                    target_ind_not_matched_i = target_ind_not_matched_idx[nidx] + count
                    if target['cur_target']['boxes'][target_ind_not_matched_i][-1] > 0:
                        self.update_target(target,target_ind_not_matched_i)
                        count += 1

                # target['track_ids'] has change since FP divided cells have been separated into two boxes
                target_ind_match_matrix = prev_track_ids.unsqueeze(dim=1).eq(target['cur_target']['track_ids'])
                target['cur_target']['target_ind_matching'] = target_ind_match_matrix.any(dim=1)
                target['cur_target']['track_query_match_ids'] = target_ind_match_matrix.nonzero()[:, 1]

            target['cur_target']['target_ind_matching'] = torch.cat([
                target['cur_target']['target_ind_matching'],
                torch.tensor([False, ] * target['cur_target']['num_FPs']).bool().to(self.device)
                ])

            target['cur_target']['track_queries_TP_mask'] = torch.cat([
                target['cur_target']['target_ind_matching'],
                torch.tensor([False, ] * (self.num_queries)).to(self.device)
                ]).bool()

            # track query masks
            track_queries_mask = torch.ones_like(target['cur_target']['target_ind_matching']).bool()
            track_queries_fal_pos_mask = torch.zeros_like(target['cur_target']['target_ind_matching']).bool()
            track_queries_fal_pos_mask[~target['cur_target']['target_ind_matching']] = True

            if 'dn_track' in target:

                dn_track['target_ind_matching'] = torch.cat([
                    dn_track['target_ind_matching'],
                    torch.tensor([False, ] * dn_track['num_FPs']).bool().to(self.device)
                    ])

                dn_track['track_queries_TP_mask'] = dn_track['target_ind_matching']
                dn_track['track_queries_mask'] = torch.ones_like(dn_track['target_ind_matching']).to(self.device).bool()
                dn_track['track_queries_fal_pos_mask'] = torch.zeros_like(dn_track['target_ind_matching']).to(self.device).bool()
                dn_track['track_queries_fal_pos_mask'][~dn_track['target_ind_matching']] = True      

                if self.dn_track_add_object_queries and add_object_queries_to_dn_track:
                    dn_track['track_queries_TP_mask'] = torch.cat([
                        dn_track['track_queries_TP_mask'],
                        torch.tensor([False, ] * self.num_queries).to(self.device)
                    ]).bool()
                    dn_track['track_queries_mask'] = torch.cat([
                        dn_track['track_queries_mask'],
                        torch.tensor([False, ] * self.num_queries).to(self.device)
                    ]).bool()

                    dn_track['track_queries_fal_pos_mask'] = torch.cat([
                        dn_track['track_queries_fal_pos_mask'],
                        torch.tensor([False, ] * self.num_queries).to(self.device)
                    ]).bool()

                boxes = target['prev_target']['boxes'][dn_track['prev_ind'][1],:4]

                if dn_track['num_FPs'] > 0:
                    FP_boxes = torch.zeros((dn_track['num_FPs'],4)).to(self.device)
                    # FP_boxes = self.get_FP_boxes(dn_track,i,prev_out) # maybe update this in the futre
                    boxes = torch.cat((boxes,FP_boxes),axis=0)

                assert torch.sum(boxes < 0) == 0, 'Bboxes need to have positive values'
                        
                dn_track['track_query_boxes_gt'] = boxes.clone()

                l_1 = self.dn_track_l1
                l_2 = self.dn_track_l2

                noised_boxes = torch.rand_like(boxes) * 2 - 1
                boxes[..., :2] += boxes[..., 2:] * noised_boxes[..., :2] * l_1
                boxes[..., 2:] *= 1 + l_2 * noised_boxes[..., 2:]
                boxes = torch.clamp(boxes,0,1)

                dn_track['track_query_boxes'] = boxes
                dn_track['num_queries'] = len(dn_track['track_queries_mask'])

                dn_track['track_query_hs_embeds'] = self.dn_track_embedding.weight.repeat(len(boxes),1)

                assert torch.sum(dn_track['track_query_boxes'] < 0) == 0, 'Bboxes need to have positive values'

            boxes = prev_out['pred_boxes'].detach()[i, target['cur_target']['prev_ind'][0]]

            if self.prev_prev_track:
                for prev_ind_out in target['cur_target']['prev_ind'][0]:
                    if (target['cur_target']['prev_ind'][0] == prev_ind_out).sum() == 2:
                        ind = torch.where(target['cur_target']['prev_ind'][0] == prev_ind_out)[0][1]
                        boxes[ind,:4] = boxes[ind,4:]
                    elif (target['cur_target']['prev_ind'][0] == prev_ind_out).sum() == 1:
                        pass
                    else:
                        NotImplementedError

            if target['cur_target']['num_FPs'] > 0:
                FP_boxes = self.get_FP_boxes(target['cur_target'],i,prev_out)
                boxes = torch.cat((boxes,FP_boxes))

            assert torch.sum(boxes < 0) == 0, 'Bboxes need to have positive values'
                    
            target['cur_target']['track_query_boxes'] = boxes[:,:4]
            target['cur_target']['track_query_hs_embeds'] = prev_out['hs_embed'][i, target['cur_target']['prev_ind'][0]] 

            assert torch.sum(target['cur_target']['track_query_boxes'] < 0) == 0, 'Bboxes need to have positive values'
            assert target['cur_target']['track_query_hs_embeds'].shape[0] + self.num_queries == len(target['cur_target']['track_queries_TP_mask'])
            assert target['cur_target']['track_query_boxes'].shape[0] + self.num_queries == len(target['cur_target']['track_queries_TP_mask'])

            target['cur_target']['track_queries_mask'] = torch.cat([
                track_queries_mask,
                torch.tensor([False, ] * self.num_queries).to(self.device)
            ]).bool()

            target['cur_target']['track_queries_fal_pos_mask'] = torch.cat([
                track_queries_fal_pos_mask,
                torch.tensor([False, ] * self.num_queries).to(self.device)
            ]).bool()

            target['cur_target']['num_queries'] = len(track_queries_mask) + self.num_queries

    def forward(self, samples: utils.NestedTensor, targets: list = None, track=False, prev_features=None, prev_out=None, epoch=None):   
        
        add_object_queries_to_dn_track=False  
        if targets is not None and (not self._tracking or self.no_data_aug):

            if track:
                backprop_context = torch.no_grad
                if self._backprop_prev_frame:
                    backprop_context = nullcontext

                with backprop_context():

                    if (epoch is not None and epoch < 20) or not self.use_prev_prev_frame:
                        track_prev_prev_frame_num = False
                    else:
                        track_prev_prev_frame_num = torch.rand(1)[0] < 0.3

                    if torch.tensor([target['prev_prev_target']['empty'] or target['prev_target']['empty']for target in targets]).sum() == 0 and (track_prev_prev_frame_num or self.eval_prev_prev_frame):

                        # PREV PREV
                        prev_prev_features = super().forward(
                            [t['prev_prev_image'] for t in targets],
                             return_features_only=True)

                        targets = utils.man_track_ids(targets,'prev_prev_target','prev_target')
                        self.add_dn_track_queries_to_prev_targets(targets)

                        # PREV
                        prev_out, _, prev_features,_, _, _ = super().forward(
                            [t['prev_image'] for t in targets],
                            targets,
                            'prev_target',
                            prev_prev_features)

                        self.prev_prev_track = True

                        prev_outputs_without_aux = {k: v for k, v in prev_out.items() if 'aux_outputs' not in k}

                        prev_indices, targets = self._matcher(prev_outputs_without_aux, targets, 'prev_target')

                        if epoch is None or epoch > self.epoch_to_start_using_flexible_divisions:
                            targets = utils.update_early_or_late_track_divisions(
                                prev_out,
                                targets,
                                'prev_prev_target',
                                'prev_target',
                                'cur_target',
                            )

                        new_prev_indices = []
                        for target, (prev_ind_out, prev_ind_tgt) in zip(targets,prev_indices):
                            boxes = target['prev_target']['boxes'][prev_ind_tgt]
                            boxes_orig = target['prev_target']['boxes_orig']

                            new_prev_ind_tgt = torch.ones((len(boxes_orig)),dtype=torch.int64) * -1
                            new_prev_ind_out = torch.ones((len(boxes_orig)),dtype=torch.int64) * -1

                            count = 0
                            for b,box in enumerate(boxes):
                                if box[-1] > 0:
                                    new_prev_ind_out[b+count] = prev_ind_out[b]
                                    new_prev_ind_out[b+count+1] = prev_ind_out[b]
                                    new_ind_1 = boxes_orig[:,:4].eq(box[:4]).all(-1).nonzero()[0][0]
                                    new_ind_2 = boxes_orig[:,:4].eq(box[4:]).all(-1).nonzero()[0][0]
                                    new_prev_ind_tgt[b+count] = new_ind_1
                                    new_prev_ind_tgt[b+count+1] = new_ind_2
                                    count += 1
                                else:
                                    new_ind = boxes_orig[:,:4].eq(box[:4]).all(-1).nonzero()[0][0]
                                    new_prev_ind_tgt[b+count] = new_ind         
                                    new_prev_ind_out[b+count] = prev_ind_out[b]  

                            assert -1 not in new_prev_ind_out and -1 not in new_prev_ind_tgt

                            new_prev_indices.append((new_prev_ind_out,new_prev_ind_tgt))

                        prev_indices = new_prev_indices

                        fix_incorrectly_tracked_cells = False

                        if fix_incorrectly_tracked_cells:
                            for t,target in enumerate(len(targets)):
                                gt_boxes = target['prev_target']['boxes'][prev_indices[t][1]]
                                pred_boxes = prev_outputs_without_aux['pred_boxes'][t,prev_indices[t][0]]

                                div_keep = gt_boxes[:,-1] > 0
                                divisor = torch.ones((len(gt_boxes))).to('cuda') + (div_keep).float()
                                iou_matrix = torch.diagonal(box_ops.generalized_box_iou(box_ops.box_cxcywh_to_xyxy(pred_boxes[:,:4]),box_ops.box_cxcywh_to_xyxy(gt_boxes[:,:4]),return_iou_only=True)) 
                                iou_matrix[div_keep] += torch.diagonal(box_ops.generalized_box_iou(box_ops.box_cxcywh_to_xyxy(pred_boxes[div_keep,4:]),box_ops.box_cxcywh_to_xyxy(gt_boxes[div_keep,4:]),return_iou_only=True)) 
                                iou_matrix /= divisor

                                where_incorrect = torch.where(iou_matrix < 0.5)[0]

                                if len(where_incorrect) > 0:
                                    for where_inc in where_incorrect:
                                        prev_outputs_without_aux['pred_boxes'][t,prev_indices[t][0][where_inc]] = gt_boxes[where_inc].clone()
                                        prev_outputs_without_aux['hs_embed'][t,prev_indices[t][0][where_inc]] = self.dn_track_embedding.weight

                    else:
                        prev_out, _, prev_features, _, _, _ = super().forward([t['prev_image'] for t in targets])
                        self.prev_prev_track = False

                        prev_outputs_without_aux = {k: v for k, v in prev_out.items() if 'aux_outputs' not in k}

                        prev_indices, targets = self._matcher(prev_outputs_without_aux, targets, 'prev_target')

                        N = prev_out['pred_logits'].shape[1]

                        for target in targets:
                            target['prev_target']['track_queries_mask'] = torch.zeros((self.num_queries)).bool().to(self.device)

                        targets, prev_indices = utils.update_object_detection(prev_outputs_without_aux,targets,prev_indices,self.num_queries,'prev_prev_target','prev_target','cur_target')

                    targets = utils.man_track_ids(targets,'prev_target','cur_target')
                    add_object_queries_to_dn_track = torch.rand(1)[0] < 0.2
                    self.get_random_mask(targets,prev_indices)
                    self.add_track_queries_to_targets(targets, prev_outputs_without_aux,add_object_queries_to_dn_track=add_object_queries_to_dn_track)

            else: # if we are perforoming object detection, we need to make sure the ground truths are all single cells as divided cells will be grouped as one
                
                for target in targets:
                    target['cur_target']['track_queries_mask'] = torch.zeros((self.num_queries)).bool().to(self.device)
                    target['cur_target']['num_queries'] = self.num_queries
                    
        dn_enc = self.dn_enc and torch.rand(1)[0] < 0.3

        out, targets, features, memory, hs, mask_features  = super().forward(samples, targets, 'cur_target', prev_features, dn_object=self.dn_object, dn_enc=dn_enc, add_object_queries_to_dn_track=add_object_queries_to_dn_track)

        return out, targets, features, memory, hs, mask_features, prev_out

# TODO: with meta classes
class DETRTracking(DETRTrackingBase, DETR):
    def __init__(self, tracking_kwargs, detr_kwargs):
        DETR.__init__(self, **detr_kwargs)
        DETRTrackingBase.__init__(self, **tracking_kwargs)


class DeformableDETRTracking(DETRTrackingBase, DeformableDETR):
    def __init__(self, tracking_kwargs, detr_kwargs):
        DeformableDETR.__init__(self, **detr_kwargs)
        DETRTrackingBase.__init__(self, **tracking_kwargs)
