import torch
from .misc import man_track_ids
from .box_ops import combine_div_boxes, calc_iou, combine_div_masks, divide_mask, divide_box, generalized_box_iou, box_cxcywh_to_xyxy

def update_early_or_late_track_divisions(
    outputs,
    targets,
    training_method,
    prev_target_name,
    cur_target_name,
    fut_target_name,
    ):

    device = outputs['pred_logits'].device
    use_masks = 'masks' in targets[0][training_method][cur_target_name]

    # check for early / late cell division and adjust ground truths as necessary
    for t, target in enumerate(targets):

        man_track = target[training_method]['man_track']

        prev_target = target[training_method][prev_target_name]
        cur_target = target[training_method][cur_target_name]
        fut_target = target[training_method][fut_target_name]

        if cur_target['empty']:
            continue

        if 'track_query_match_ids' in cur_target:

            # Get all prdictions for TP track queries
            pred_boxes_track = outputs['pred_boxes'][t][cur_target['track_queries_TP_mask']].detach()
            pred_logits_track = outputs['pred_logits'][t][cur_target['track_queries_TP_mask']].sigmoid().detach()
            # Check to see if there were any divisions in the future frame; if not, we skip to check for early division

            targets = man_track_ids(targets,training_method,prev_target_name,cur_target_name)

            boxes = cur_target['boxes'].clone()
            track_ids = cur_target['track_ids'].clone()

            for p, pred_box in enumerate(pred_boxes_track):
                box = boxes[cur_target['track_query_match_ids'][p]].clone()
                track_id = track_ids[cur_target['track_query_match_ids'][p]].clone()

                if track_id not in prev_target['track_ids']:
                    print(f'track_id: {track_id}')
                    print(f"dataset_nb: {target['dataset_nb']}")
                    print(f'training method: {training_method}')
                    print(f'cur_target_name: {cur_target_name}')
                    print(f'Image_id: {target[training_method][cur_target_name]["image_id"]}')
                    print(f'framenb: {target[training_method][cur_target_name]["framenb"]}')
                    print(f'Num of boxes: {target[training_method][cur_target_name]["boxes"].shape[0]}')
                    print(f'Num of orig boxes: {target[training_method][cur_target_name]["boxes_orig"].shape[0]}')
                    print(f'Num of prev boxes: {target[training_method][prev_target_name]["boxes"].shape[0]}')
                    print(f'Num of prev orig boxes: {target[training_method][prev_target_name]["boxes_orig"].shape[0]}')
                    print(target[training_method]['man_track'][target[training_method]['man_track'][:,0] == track_id])
                    mot_cellnb = target[training_method]['man_track'][target[training_method]['man_track'][:,0] == track_id,-1][0]
                    if mot_cellnb > 0:
                        print(f'mother cellnb: {mot_cellnb}')
                        print(target[training_method]['man_track'][target[training_method]['man_track'][:,-1] == mot_cellnb])       
                    print(target[training_method]['man_track'])

                                        
                assert track_id in prev_target['track_ids']

                if prev_target['flexible_divisions'][prev_target['track_ids'] == track_id]:
                    continue

                # First check if the model predicted a single cell instead of a division
                if box[-1] > 0 and pred_logits_track[p,0] > 0.5 and pred_logits_track[p,-1] < 0.5: # gt - division ; pred - single-cell

                    area_box_1 = box[2] * box[3]
                    area_box_2 = box[6] * box[7]

                    prev_box = prev_target['boxes'][prev_target['track_ids'] == track_id]
                    area_prev_box = prev_box[0,2] * prev_box[0,3]

                    if area_box_1 > area_box_2 * 2 or area_box_2 > area_box_1 * 2: # and 0.8 * area_prev_box < (area_box_1 + area_box_2): # Last part checks if cell is exiting frame then it would discount it as a filamented cell
                        continue # force filament cells to divide training otherwise it favors flexible divisoin due to asymmetrical division
                    
                    combined_box = combine_div_boxes(box)
                    iou_div = calc_iou(box,pred_box)

                    pred_box[4:] = 0
                    iou_combined = calc_iou(combined_box,pred_box)

                    if iou_combined - iou_div > 0 and iou_combined > 0.5: 
                        cur_target['boxes'][cur_target['track_query_match_ids'][p]] = combined_box
                        cur_target['labels'][cur_target['track_query_match_ids'][p]] = torch.tensor([0,1]).to(device)
                        cur_target['flexible_divisions'][cur_target['track_query_match_ids'][p]] = True
                        

                        track_id = cur_target['track_ids'][cur_target['track_query_match_ids'][p]].clone()

                        div_bool_1 = cur_target['boxes_orig'][:,:4].eq(box[None,:4]).all(1)
                        div_bool_2 = cur_target['boxes_orig'][:,:4].eq(box[None,4:]).all(1)

                        div_track_id_1 = cur_target['track_ids_orig'][div_bool_1].clone()
                        div_track_id_2 = cur_target['track_ids_orig'][div_bool_2].clone()

                        cur_target['boxes_orig'][div_bool_1] = combined_box.clone()
                        cur_target['boxes_orig'] = cur_target['boxes_orig'][~div_bool_2]

                        cur_target['track_ids_orig'][div_bool_1] = track_id
                        cur_target['track_ids_orig'] = cur_target['track_ids_orig'][~div_bool_2]

                        cur_target['flexible_divisions_orig'][div_bool_1] = True
                        cur_target['flexible_divisions_orig'] = cur_target['flexible_divisions_orig'][~div_bool_2]

                        is_touching_edge = cur_target['is_touching_edge_orig'][div_bool_1] or cur_target['is_touching_edge_orig'][div_bool_2]
                        cur_target['is_touching_edge'][cur_target['track_query_match_ids'][p]] = is_touching_edge

                        cur_target['is_touching_edge_orig'][div_bool_1] = is_touching_edge
                        cur_target['is_touching_edge_orig'] = cur_target['is_touching_edge_orig'][~div_bool_2]

                        assert cur_target['labels_orig'][div_bool_1,1] == 1
                        cur_target['labels_orig'] = cur_target['labels_orig'][~div_bool_2]

                        if use_masks:
                            mask = cur_target['masks'][cur_target['track_query_match_ids'][p]]
                            prev_mask = prev_target['masks'][prev_target['track_ids'] == track_id][0]
                            combined_mask = combine_div_masks(mask,prev_mask)
                            cur_target['masks'][cur_target['track_query_match_ids'][p]] = combined_mask

                            cur_target['masks_orig'][div_bool_1] = combined_mask
                            cur_target['masks_orig'] = cur_target['masks_orig'][~div_bool_2]    

                        track_id_ind = man_track[:,0] == track_id
                        div_track_id_1_ind = man_track[:,0] == div_track_id_1
                        div_track_id_2_ind = man_track[:,0] == div_track_id_2

                        man_track[track_id_ind,2] += 1                    
                        man_track[div_track_id_1_ind,1] += 1                    
                        man_track[div_track_id_2_ind,1] += 1                    

                        # Check to see if one of the daughters cells leave the FOV the frame after they are born
                        # If so, the mother cell will replace the other daugher cell since this is just tracking and division occured                        
                        if man_track[div_track_id_1_ind,1] > man_track[div_track_id_1_ind,2] or man_track[div_track_id_2_ind,1] > man_track[div_track_id_2_ind,2]:
                            man_track[track_id_ind,2] = torch.max(man_track[div_track_id_1_ind,2],man_track[div_track_id_2_ind,2])
                            man_track[div_track_id_1_ind,1:] = -1
                            man_track[div_track_id_2_ind,1:] = -1

                            # Since cell division does not exist in future frames, we need to update the fut_track_id to the mother_id
                            if div_track_id_1 in fut_target['track_ids'] and div_track_id_2 in fut_target['track_ids']: 
                                raise NotImplementedError
                            elif div_track_id_1 in fut_target['track_ids']:
                                fut_target['track_ids'][fut_target['track_ids'] == div_track_id_1] = track_id.long().to(device)
                                if fut_target_name != 'fut_target' and div_track_id_1 in target[training_method]['fut_target']['track_ids']:
                                    target[training_method]['fut_target']['track_ids'][target[training_method]['fut_target']['track_ids'] == div_track_id_1] = track_id.long().to(device)
                            elif div_track_id_2 in fut_target['track_ids']:
                                fut_target['track_ids'][fut_target['track_ids'] == div_track_id_2] = track_id.long().to(device)
                                if fut_target_name != 'fut_target' and div_track_id_2 in target[training_method]['fut_target']['track_ids']:
                                    target[training_method]['fut_target']['track_ids'][target[training_method]['fut_target']['track_ids'] == div_track_id_2] = track_id.long().to(device)

                            fut_target['track_ids_orig'] = fut_target['track_ids'].clone()
                            if fut_target_name != 'fut_target':
                                target[training_method]['fut_target']['track_ids_orig'] = target[training_method]['fut_target']['track_ids'].clone()

                            if div_track_id_1 in man_track[:,-1] and div_track_id_2 in man_track[:,-1]:
                                # error in dataset here. Cell divides two frames in a row
                                fut_div_track_id_1, fut_div_track_id_2 = man_track[(man_track[:,-1] == div_track_id_1),0]
                                fut_div_track_id_1_ind = man_track[:,0] == fut_div_track_id_1
                                fut_div_track_id_2_ind = man_track[:,0] == fut_div_track_id_2
                                man_track[fut_div_track_id_1_ind,-1] = 0
                                man_track[fut_div_track_id_2_ind,-1] = 0  
                                fut_div_track_id_1, fut_div_track_id_2 = man_track[(man_track[:,-1] == div_track_id_2),0]
                                fut_div_track_id_1_ind = man_track[:,0] == fut_div_track_id_1
                                fut_div_track_id_2_ind = man_track[:,0] == fut_div_track_id_2
                                man_track[fut_div_track_id_1_ind,-1] = 0
                                man_track[fut_div_track_id_2_ind,-1] = 0                          
                            elif div_track_id_1 in man_track[:,-1]:
                                fut_div_track_id_1, fut_div_track_id_2 = man_track[(man_track[:,-1] == div_track_id_1),0]
                                fut_div_track_id_1_ind = man_track[:,0] == fut_div_track_id_1
                                fut_div_track_id_2_ind = man_track[:,0] == fut_div_track_id_2
                                man_track[fut_div_track_id_1_ind,-1] = track_id
                                man_track[fut_div_track_id_2_ind,-1] = track_id
                            elif div_track_id_2 in man_track[:,-1]:
                                fut_div_track_id_1, fut_div_track_id_2 = man_track[(man_track[:,-1] == div_track_id_2),0]
                                fut_div_track_id_1_ind = man_track[:,0] == fut_div_track_id_1
                                fut_div_track_id_2_ind = man_track[:,0] == fut_div_track_id_2
                                man_track[fut_div_track_id_1_ind,-1] = track_id
                                man_track[fut_div_track_id_2_ind,-1] = track_id

                            assert (torch.arange(1,target[training_method]['man_track'].shape[0]+1,dtype=target[training_method]['man_track'].dtype).to(target[training_method]['man_track'].device) == target[training_method]['man_track'][:,0]).all()

                # assert cur_target['track_query_match_ids'].max() < cur_target['boxes'].shape[0]
                 

            targets = man_track_ids(targets,training_method,cur_target_name,fut_target_name)

            for p, pred_box in enumerate(pred_boxes_track):
                box = boxes[cur_target['track_query_match_ids'][p]].clone()

                if box[-1] == 0 and (pred_logits_track[p] > 0.5).all(): # gt - single-cell ; pred- division
                    # if model predcitions division, check future frame and see if there is a division

                    track_id = track_ids[cur_target['track_query_match_ids'][p]].clone()
                    track_id_ind = man_track[:,0] == track_id

                    if man_track[track_id_ind,2] == prev_target['framenb']:
                        track_id = man_track[track_id_ind,-1]
                        
                    if track_id not in fut_target['track_ids']:
                        continue  # Cell leaves chamber in future frame

                    fut_box_ind = (fut_target['track_ids'] == track_id).nonzero()[0][0]
                    fut_box = fut_target['boxes'][fut_box_ind]


                    if fut_box[-1] > 0: # If cell divides next frame, we check to see if the model is predicting an early division
                        div_box = divide_box(box,fut_box)

                        iou_div = calc_iou(div_box,pred_box)
                        iou = calc_iou(box[:4],pred_box[:4])

                        if iou_div - iou > 0 and iou_div > 0.5:
                            cur_target['boxes'][cur_target['track_query_match_ids'][p]] = div_box
                            cur_target['labels'][cur_target['track_query_match_ids'][p]] = torch.tensor([0,0]).to(device)
                            cur_target['flexible_divisions_orig'][cur_target['track_query_match_ids'][p]] = True           
                            # don't need to adjust is_touching_edge because it will remain the same but need to adjust the orig version

                            fut_track_id_1, fut_track_id_2 = man_track[man_track[:,-1] == track_id,0]
                            fut_box_1 = fut_target['boxes_orig'][fut_target['track_ids_orig'] == fut_track_id_1][0]
                            fut_box_2 = fut_target['boxes_orig'][fut_target['track_ids_orig'] == fut_track_id_2][0]

                            if (div_box[:2] - fut_box_1[:2]).square().sum() +(div_box[4:6] - fut_box_2[:2]).square().sum() > (div_box[:2] - fut_box_2[:2]).square().sum() +(div_box[4:6] - fut_box_1[:2]).square().sum():
                                fut_track_id_1, fut_track_id_2 = fut_track_id_2, fut_track_id_1
                            assert (torch.arange(1,target[training_method]['man_track'].shape[0]+1,dtype=target[training_method]['man_track'].dtype).to(target[training_method]['man_track'].device) == target[training_method]['man_track'][:,0]).all()

                            ind_tgt_orig = torch.where(cur_target['boxes_orig'].eq(box).all(-1))[0][0]
                            cur_target['boxes_orig'][ind_tgt_orig,:4] = div_box[:4]
                            cur_target['boxes_orig'] = torch.cat((cur_target['boxes_orig'], torch.cat((div_box[4:],torch.zeros_like(div_box[4:])))[None]),axis=0)

                            cur_target['track_ids_orig'][ind_tgt_orig] = fut_track_id_1 
                            cur_target['track_ids_orig'] = torch.cat((cur_target['track_ids_orig'], torch.tensor([fut_track_id_2]).to(device)))

                            cur_target['labels_orig'] = torch.cat((cur_target['labels_orig'], cur_target['labels_orig'][:1]),axis=0)                   
                            cur_target['flexible_divisions_orig'] = torch.cat((cur_target['flexible_divisions_orig'], torch.tensor([True]).to(device)))            

                            cur_target['is_touching_edge_orig'] = torch.cat((cur_target['is_touching_edge_orig'], cur_target['is_touching_edge_orig'][ind_tgt_orig][None]))                   

                            if use_masks:
                                mask = cur_target['masks'][cur_target['track_query_match_ids'][p]]
                                fut_mask = fut_target['masks'][fut_box_ind]
                                div_mask = divide_mask(mask,fut_mask)
                                cur_target['masks'][cur_target['track_query_match_ids'][p]] = div_mask
                                cur_target['masks_orig'][ind_tgt_orig,:1] = div_mask[:1]
                                cur_target['masks_orig'] = torch.cat((cur_target['masks_orig'], torch.cat((div_mask[1:],torch.zeros_like(div_mask[1:])))[None]),axis=0)

                            fut_track_id_1_ind = man_track[:,0] == fut_track_id_1
                            fut_track_id_2_ind = man_track[:,0] == fut_track_id_2

                            man_track[fut_track_id_1_ind,1] -= 1
                            man_track[fut_track_id_2_ind,1] -= 1
                            man_track[track_id_ind,2] -= 1

                            if man_track[track_id_ind,1] > man_track[track_id_ind,2]:
                                man_track[track_id_ind,1:] = -1
                                man_track[fut_track_id_1_ind,-1] = 0
                                man_track[fut_track_id_2_ind,-1] = 0

        targets[t][training_method]['man_track'] = man_track
                            

    if 'track_query_match_ids' in cur_target: # This needs to be updated for aux outputs and enc outputs because the matcher is rerun then

        targets = man_track_ids(targets,training_method,prev_target_name,cur_target_name)

        prev_track_ids = prev_target['track_ids'][cur_target['prev_ind'][1]]

        # match track ids between frames
        target_ind_match_matrix = prev_track_ids.unsqueeze(dim=1).eq(cur_target['track_ids'])
        cur_target['target_ind_matching'] = target_ind_match_matrix.any(dim=1)
        cur_target['track_query_match_ids'] = target_ind_match_matrix.nonzero()[:, 1]

        # For images with no cells in them, we reformat target_ind_matching so torch.cat works properly with zero cells
        if cur_target['target_ind_matching'].shape[0] == 0:
            cur_target['target_ind_matching'] = torch.tensor([],device=device).bool()

        track_queries_mask = torch.ones_like(cur_target['target_ind_matching']).bool()

        num_queries = (~cur_target['track_queries_mask']).sum()

        cur_target['track_queries_mask'] = torch.cat([
            track_queries_mask,
            torch.tensor([True,] * cur_target['num_FPs']).to(device),
            torch.tensor([False,] * num_queries).to(device), 
            ]).bool()

        cur_target['track_queries_TP_mask'] = torch.cat([
            cur_target['target_ind_matching'],
            torch.tensor([False,] * cur_target['num_FPs']).to(device),
            torch.tensor([False,] * num_queries).to(device), 
            ]).bool()

        cur_target['track_queries_fal_pos_mask'] = torch.cat([
            ~cur_target['target_ind_matching'],
            torch.tensor([True,] * cur_target['num_FPs']).to(device),
            torch.tensor([False,] * num_queries).to(device),
            ]).bool()

        assert cur_target['track_queries_TP_mask'].sum() == len(cur_target['track_query_match_ids'])

    return targets

def update_object_detection(
    outputs,
    targets,
    indices,
    num_queries,
    training_method,
    prev_target_name,
    cur_target_name,
    fut_target_name,):

    N = outputs['pred_logits'].shape[1]
    use_masks = 'masks' in targets[0][training_method][cur_target_name]
    device = outputs['pred_logits'].device

    # Indicies are saved in targets for calcualting object detction / tracking accuracy
    for t,(target,(ind_out,ind_tgt)) in enumerate(zip(targets,indices)):

        prev_target = target[training_method][prev_target_name]
        cur_target = target[training_method][cur_target_name]
        fut_target = target[training_method][fut_target_name]

        if cur_target['empty']:
            continue

        if training_method == 'dn_object':
            cur_target['track_queries_mask'] = torch.zeros_like(cur_target['track_queries_mask']).bool()

        man_track = target[training_method]['man_track']
        framenb = cur_target['framenb']

        skip =[] # If a GT cell is split into two cells, we want to skip the second cell
        ind_keep = torch.tensor([True for _ in range(len(ind_tgt))]).bool()

        for ind_out_i, ind_tgt_i in zip(ind_out,ind_tgt):
            # Confirm prediction is an object query, not a track query
            if ind_out_i >= (N - num_queries) and ind_tgt_i not in skip:
                if 'track_queries_mask' in cur_target:
                    assert not cur_target['track_queries_mask'][ind_out_i] 
                track_id = cur_target['track_ids'][ind_tgt_i].clone()
                # assert (man_track[:,-1] == track_id).sum() == 2 or (man_track[:,-1] == track_id).sum() == 0
                
                track_id_ind = man_track[:,0] == track_id
                # Check if cell has just divided --> the two daugheter cells will be labeled cell 1 and 2
                if man_track[track_id_ind,1] == framenb and man_track[track_id_ind,-1] > 0:
                    
                    # Get id of mother cell by using the man_track
                    mother_id = man_track[track_id_ind,-1].clone().long()
                    assert mother_id in prev_target['track_ids']
                    track_id_1 = track_id

                    # Relabel variables to be consistent with cell 1 & 2
                    ind_tgt_1 = ind_tgt_i
                    ind_out_1 = ind_out_i
                    ind_1 = torch.where(ind_out == ind_out_1)[0][0] 

                    # Get information for the other daughter cell
                    track_id_2 = man_track[(man_track[:,-1] == mother_id) * (man_track[:,0] != track_id_1),0][0].clone()
                    ind_tgt_2 = torch.where(cur_target['track_ids'] == track_id_2)[0][0].cpu()
                    ind_2 = torch.where(ind_tgt == ind_tgt_2)[0][0] 
                    ind_out_2 = ind_out[ind_2] 

                    # Get predictions for cell 1 & 2
                    pred_box_1 =  outputs['pred_boxes'][t,ind_out_1].detach()
                    pred_box_2 =  outputs['pred_boxes'][t,ind_out_2].detach()

                    # Get groundtruths for cell 1 & 2
                    box_1 =  cur_target['boxes'][ind_tgt_1]
                    box_2 =  cur_target['boxes'][ind_tgt_2]

                    assert box_1[-1] == 0 and box_2[-1] == 0, 'Cells have just divided. Each box should contain just one cell'

                    # Combine predictions and GTs 
                    # This is done for formatting issues when calculating the iou
                    boxes_1_2 = torch.cat((box_1[:4],box_2[:4]))
                    pred_boxes_1_2 = torch.cat((pred_box_1[:4],pred_box_2[:4]))
                    
                    # Calculate iou for cell 1 matching to GT 1 and cell 2 matching to GT 2
                    iou_sep,flip = calc_iou(pred_boxes_1_2,boxes_1_2,return_flip=True)

                    if flip:
                        iou_1 = calc_iou(pred_boxes_1_2[:4],boxes_1_2[4:])
                        pred_logits_1 = outputs['pred_logits'][t,ind_out_2].sigmoid()[0]
                        iou_2 = calc_iou(pred_boxes_1_2[4:],boxes_1_2[:4])
                        pred_logits_2 = outputs['pred_logits'][t,ind_out_1].sigmoid()[0]
                    else:
                        iou_1 = calc_iou(pred_boxes_1_2[:4],boxes_1_2[:4])
                        pred_logits_1 = outputs['pred_logits'][t,ind_out_1].sigmoid()[0]
                        iou_2 = calc_iou(pred_boxes_1_2[4:],boxes_1_2[4:])
                        pred_logits_2 = outputs['pred_logits'][t,ind_out_2].sigmoid()[0]

                    # Make a guess if GT 1 and GT 2 were actually 1 cell
                    # Note this does not work well when cells are on the edge of the image
                    combined_box = combine_div_boxes(boxes_1_2)

                    # Get all unused object queries that were predicted to be cells but did not match to a GT
                    # We are checking to see if the model predicted GT 1 & 2 as one cell but didn't match either
                    potential_object_query_indices = [ind_out_id for ind_out_id in torch.arange(N-num_queries,N) if ((ind_out_id not in ind_out or ind_out_id in [ind_out_1,ind_out_2]) and outputs['pred_logits'][t,ind_out_id,0].sigmoid().detach() > 0.5)]
                    
                    if len(potential_object_query_indices) == 0:
                        continue

                    potential_pred_boxes = outputs['pred_boxes'][t,potential_object_query_indices].detach() 

                    # Check iou for all unused pred boxes against the combined box
                    iou_combined = generalized_box_iou(box_cxcywh_to_xyxy(potential_pred_boxes[:,:4]),box_cxcywh_to_xyxy(combined_box[None,:4]),return_iou_only=True)

                    # Get ind of which pred box that matched best to the combined box
                    max_ind = torch.argmax(iou_combined)
                    assert iou_combined[max_ind] <= 1 and iou_combined[max_ind] >= 0 and iou_sep <= 1 and iou_sep >= 0, 'Calc_iou not working; producings numbers outside of 0 and 1'

                    # We check to see if the separate pred boxes 1 & 2 have a higher iou than the pred box combined
                    if iou_combined[max_ind] - iou_sep > 0 and iou_combined[max_ind] > 0.5 and (iou_combined[max_ind] > iou_1 or pred_logits_1 < 0.5) and (iou_combined[max_ind] > iou_2 or pred_logits_2 < 0.5):
                        # Get ind_out for the pred box combined
                        ind_out_combined = potential_object_query_indices[max_ind]

                        # Track which pred / tgt are going to get removed since two cells are being merged into one cell
                        ind_keep[ind_2] = False
                        skip += [ind_tgt_1,ind_tgt_2]
            
                        # Replace box 1 with the combined box in ind_tgt_1 and remove box 2
                        cur_target['boxes'][ind_tgt_1] = combined_box
                        cur_target['track_ids'][ind_tgt_1] = mother_id
                        cur_target['track_ids'][ind_tgt_2] = -1
                        cur_target['flexible_divisions'][ind_tgt_1] = True
                        cur_target['is_touching_edge'][ind_tgt_1] = cur_target['is_touching_edge'][ind_tgt_1] or cur_target['is_touching_edge'][ind_tgt_2] # if either of the two gt boxes were touching edge, then the new combined box is touching the edge as well

                        assert mother_id not in cur_target['track_ids_orig']
                        ind_tgt_orig_1 = cur_target['track_ids_orig'] == track_id_1
                        cur_target['track_ids_orig'][ind_tgt_orig_1] = mother_id
                        cur_target['boxes_orig'][ind_tgt_orig_1] = combined_box
                        cur_target['flexible_divisions_orig'][ind_tgt_orig_1] = True
                        cur_target['is_touching_edge_orig'][ind_tgt_orig_1] = cur_target['is_touching_edge'][ind_tgt_1]

                        ind_orig_keep = cur_target['track_ids_orig'] != track_id_2
                        
                        cur_target['track_ids_orig'] = cur_target['track_ids_orig'][ind_orig_keep]
                        cur_target['boxes_orig'] = cur_target['boxes_orig'][ind_orig_keep]
                        cur_target['labels_orig'] = cur_target['labels_orig'][ind_orig_keep]
                        cur_target['flexible_divisions_orig'] = cur_target['flexible_divisions_orig'][ind_orig_keep]
                        cur_target['is_touching_edge_orig'] = cur_target['is_touching_edge_orig'][ind_orig_keep]

                        if use_masks:
                            # Get mask of mother cell in the previous frame
                            mother_ind = torch.where(prev_target['track_ids'] == mother_id)[0][0]
                            prev_mask = prev_target['masks'][mother_ind][:1]
                            
                            # Get masks of the two daughter cells (cell 1 and cell 2)
                            mask_1 = cur_target['masks'][ind_tgt_1].detach()[:1]
                            mask_2 = cur_target['masks'][ind_tgt_2].detach()[:1]
                            sep_mask = torch.cat((mask_1,mask_2),axis=0)

                            # Make a prediction how the two daughter cell masks get combined with the mother cell mask as a reference
                            combined_mask = combine_div_masks(sep_mask,prev_mask)

                            # Replace cell 1 mask with the combined mask and remove cell 2 mask
                            cur_target['masks'][ind_tgt_1] = combined_mask
                            cur_target['masks_orig'][ind_tgt_orig_1] = combined_mask
                            cur_target['masks_orig'] = cur_target['masks_orig'][ind_orig_keep]

                        # In ind_out_1 add the object query pointing to pred box combined and get rid of ind_out_2
                        ind_out[ind_1]  = ind_out_combined
                            
                        track_id_mot_ind = man_track[:,0] == mother_id
                        track_id_1_ind = man_track[:,0] == track_id_1
                        track_id_2_ind = man_track[:,0] == track_id_2

                        man_track[track_id_mot_ind,2] += 1
                        man_track[track_id_1_ind,1] += 1
                        man_track[track_id_2_ind,1] += 1

                        # Check to see if a division occurs in the future frame. A cell could have left the FOV
                        if man_track[track_id_1_ind,2] < man_track[track_id_1_ind,1] or man_track[track_id_2_ind,2] < man_track[track_id_2_ind,1]:
                            man_track[track_id_mot_ind,2] = torch.max(man_track[track_id_1_ind,2],man_track[track_id_2_ind,2])
                            man_track[track_id_1_ind,1:] = -1
                            man_track[track_id_2_ind,1:] = -1

                            # Since cell division does not exist in future frames, we need to update the fut_track_id to the mother_id
                            if track_id_1 in fut_target['track_ids_orig']:
                                fut_target['track_ids_orig'][fut_target['track_ids_orig'] == track_id_1] = mother_id
                            elif track_id_2 in fut_target['track_ids_orig']:
                                fut_target['track_ids_orig'][fut_target['track_ids_orig'] == track_id_2] = mother_id

                            if track_id_1 in man_track[:,-1] and track_id_2 in man_track[:,-1]:
                                # error in dataset here. Cell divides two frames in a row
                                div_track_id_1, div_track_id_2 = man_track[(man_track[:,-1] == track_id_1),0]
                                div_track_id_1_ind = man_track[:,0] == div_track_id_1
                                div_track_id_2_ind = man_track[:,0] == div_track_id_2
                                man_track[div_track_id_1_ind,-1] = 0
                                man_track[div_track_id_2_ind,-1] = 0
                                div_track_id_1, div_track_id_2 = man_track[(man_track[:,-1] == track_id_2),0]
                                div_track_id_1_ind = man_track[:,0] == div_track_id_1
                                div_track_id_2_ind = man_track[:,0] == div_track_id_2
                                man_track[div_track_id_1_ind,-1] = 0
                                man_track[div_track_id_2_ind,-1] = 0
                            elif track_id_1 in man_track[:,-1]:
                                div_track_id_1, div_track_id_2 = man_track[(man_track[:,-1] == track_id_1),0]
                                div_track_id_1_ind = man_track[:,0] == div_track_id_1
                                div_track_id_2_ind = man_track[:,0] == div_track_id_2
                                man_track[div_track_id_1_ind,-1] = mother_id
                                man_track[div_track_id_2_ind,-1] = mother_id
                            elif track_id_2 in man_track[:,-1]:
                                div_track_id_1, div_track_id_2 = man_track[(man_track[:,-1] == track_id_2),0]
                                div_track_id_1_ind = man_track[:,0] == div_track_id_1
                                div_track_id_2_ind = man_track[:,0] == div_track_id_2
                                man_track[div_track_id_1_ind,-1] = mother_id
                                man_track[div_track_id_2_ind,-1] = mother_id
  
                # Check if cell is about to divide
                elif man_track[track_id_ind,2] == framenb and (man_track[:,-1] == track_id).sum() == 2 and training_method != 'dn_object': # Doesn't make sense for dn_object to detect early divisions as a FP would need to be turned into a TP which is tricky             

                        # Get groundtruths for mother cell in current frame
                        box =  cur_target['boxes'][ind_tgt_i].clone()

                        fut_track_id_1, fut_track_id_2 = man_track[(man_track[:,-1] == track_id),0]

                        # Get indices for daughter cells in the future frame
                        fut_ind_tgt_1 = torch.where(fut_target['track_ids_orig'] == fut_track_id_1)[0][0]
                        fut_ind_tgt_2 = torch.where(fut_target['track_ids_orig'] == fut_track_id_2)[0][0]

                        # Get boxes for the daughter cells
                        fut_box_1 = fut_target['boxes_orig'][fut_ind_tgt_1,:4]
                        fut_box_2 = fut_target['boxes_orig'][fut_ind_tgt_2,:4]
                        fut_box = torch.cat((fut_box_1,fut_box_2))

                        # Simulate a divided cell for the current frame
                        div_box = divide_box(box,fut_box)

                        # Get all potential pred boxes that could match the predicted divided cell
                        # potential_object_query_indices = [ind_out_id for ind_out_id in torch.arange(N-num_queries,N) if ((ind_out_id not in ind_out or ind_out_id == ind_out_i) and outputs['pred_logits'][t,ind_out_id,0].sigmoid().detach() > 0.5)]
                        potential_object_query_indices = [ind_out_id for ind_out_id in torch.arange(N-num_queries,N) if outputs['pred_logits'][t,ind_out_id,0].sigmoid().detach() > 0.5 or ind_out_id == ind_out_i] # Due to matching, if there is a early and late division adjacent to each other, this can cause issues and forces us to look at track queries as well
            
                        if len(potential_object_query_indices) > 1:

                            # Get potential pred div boxes
                            potential_pred_boxes = outputs['pred_boxes'][t,potential_object_query_indices].detach()

                            # Calculate iou for all combinations of pred div cells and the simulated div cell
                            iou_div_all = generalized_box_iou(box_cxcywh_to_xyxy(potential_pred_boxes[:,:4]),box_cxcywh_to_xyxy(torch.cat((div_box[None,:4],div_box[None,4:]),axis=0)),return_iou_only=True)

                            # Find best matching div cells
                            match_ind = torch.argmax(iou_div_all,axis=0).to('cpu')

                            if potential_object_query_indices[match_ind[0]] != ind_out_i and potential_object_query_indices[match_ind[1]] != ind_out_i:
                                continue

                            if len(torch.unique(match_ind)) == 2: # Check that two separate div cells match best to the two simulated div cells

                                # Get newly predicted cells
                                selected_pred_boxes = potential_pred_boxes[match_ind,:4]
                                iou_div = calc_iou(div_box, torch.cat((selected_pred_boxes[0],selected_pred_boxes[1])))

                                pred_box = outputs['pred_boxes'][t,ind_out_i,:4].detach()
                                iou = calc_iou(box,torch.cat((pred_box,torch.zeros_like(pred_box))))

                                assert iou_div <= 1 and iou_div >= 0 and iou <= 1 and iou >= 0

                                if iou_div - iou > 0 and iou_div > 0.5:

                                    if calc_iou(div_box[:4], selected_pred_boxes[0]) + calc_iou(div_box[4:], selected_pred_boxes[1]) < calc_iou(div_box[4:], selected_pred_boxes[0]) + calc_iou(div_box[:4], selected_pred_boxes[1]):
                                        fut_track_id_1, fut_track_id_2 = fut_track_id_2, fut_track_id_1

                                    cur_target['boxes'][ind_tgt_i] = torch.cat((div_box[:4],torch.zeros_like(div_box[:4])))
                                    cur_target['boxes'] = torch.cat((cur_target['boxes'],torch.cat((div_box[4:],torch.zeros_like(div_box[:4])))[None]))

                                    assert cur_target['labels'][ind_tgt_i,1] == 1
                                    cur_target['labels'] = torch.cat((cur_target['labels'],torch.tensor([0,1])[None,].to(device)))
                                    cur_target['track_ids'][ind_tgt_i] = fut_track_id_1
                                    cur_target['track_ids'] = torch.cat((cur_target['track_ids'],torch.tensor([fut_track_id_2]).to(device)))
                                    cur_target['flexible_divisions'][ind_tgt_i] = True
                                    cur_target['flexible_divisions'] = torch.cat((cur_target['flexible_divisions'],torch.tensor([True]).to(device)))
                                    cur_target['is_touching_edge'] = torch.cat((cur_target['is_touching_edge'],cur_target['is_touching_edge'][ind_tgt_i][None]))

                                    ind_keep = torch.cat((ind_keep,torch.tensor([True])))

                                    ind_tgt_orig_i = torch.where(cur_target['boxes_orig'].eq(box).all(-1))[0][0]

                                    cur_target['boxes_orig'][ind_tgt_orig_i] = torch.cat((div_box[:4],torch.zeros_like(div_box[:4])))
                                    cur_target['boxes_orig'] = torch.cat((cur_target['boxes_orig'],torch.cat((div_box[4:],torch.zeros_like(div_box[:4])))[None]))

                                    cur_target['labels_orig'] = torch.cat((cur_target['labels_orig'],torch.tensor([0,1])[None,].to(device)))
                                    cur_target['track_ids_orig'][ind_tgt_orig_i] = fut_track_id_1
                                    cur_target['track_ids_orig'] = torch.cat((cur_target['track_ids_orig'],torch.tensor([fut_track_id_2]).to(device)))
                                    cur_target['flexible_divisions_orig'][ind_tgt_orig_i] = True
                                    cur_target['flexible_divisions_orig'] = torch.cat((cur_target['flexible_divisions_orig'],torch.tensor([True]).to(device)))
                                    cur_target['is_touching_edge_orig'] = torch.cat((cur_target['is_touching_edge_orig'],cur_target['is_touching_edge'][ind_tgt_orig_i][None]))


                                    if use_masks:
                                        mask = cur_target['masks'][ind_tgt_i]
                                        fut_mask_1 = fut_target['masks_orig'][fut_ind_tgt_1][:1]
                                        fut_mask_2 = fut_target['masks_orig'][fut_ind_tgt_2][:1]
                                        fut_mask = torch.cat((fut_mask_1,fut_mask_2))
                                        div_mask = divide_mask(mask,fut_mask)

                                        cur_target['masks'][ind_tgt_i] = torch.cat((div_mask[:1],torch.zeros_like(div_mask[:1])))
                                        cur_target['masks'] = torch.cat((cur_target['masks'],torch.cat((div_mask[1:],torch.zeros_like(div_mask[:1])))[None]))

                                        cur_target['masks_orig'][ind_tgt_orig_i] = torch.cat((div_mask[:1],torch.zeros_like(div_mask[:1])))
                                        cur_target['masks_orig'] = torch.cat((cur_target['masks_orig'],torch.cat((div_mask[1:],torch.zeros_like(div_mask[:1])))[None]))

                                    ind_out_copy = torch.cat((ind_out,torch.tensor([-10])))

                                    if (potential_object_query_indices[match_ind[1]] != ind_out_i and potential_object_query_indices[match_ind[1]] in ind_out):
                                        ind_out[ind_out == potential_object_query_indices[match_ind[1]]] = -1
                                    elif (potential_object_query_indices[match_ind[0]] != ind_out_i and potential_object_query_indices[match_ind[0]] in ind_out):
                                        ind_out[ind_out == potential_object_query_indices[match_ind[0]]] = -1

                                    ind_out = torch.cat((ind_out,torch.tensor([potential_object_query_indices[match_ind[1]]])))
                                    ind_tgt = torch.cat((ind_tgt,torch.tensor([cur_target['boxes'].shape[0]-1])))        

                                    ind_out[ind_out_copy == ind_out_i] = torch.tensor([potential_object_query_indices[match_ind[0]]])

                                    if -1 in ind_out:
                                        unmatched_box = cur_target['boxes'][ind_out == -1]
                                        potential_object_query_indices = [ind_out_id for ind_out_id in torch.arange(N-num_queries,N) if ind_out_id not in ind_out and outputs['pred_logits'][t,ind_out_id,0].sigmoid().detach() > 0.5]

                                        if len(potential_object_query_indices) == 0:
                                            potential_object_query_indices = [ind_out_id for ind_out_id in torch.arange(N-num_queries,N) if ind_out_id not in ind_out]

                                        if len(potential_object_query_indices) == 0:
                                            continue

                                        # Get potential pred div boxes
                                        potential_pred_boxes = outputs['pred_boxes'][t,potential_object_query_indices].detach()

                                        # Calculate iou for all combinations of pred div cells and the simulated div cell
                                        iou_div_all = generalized_box_iou(box_cxcywh_to_xyxy(potential_pred_boxes[:,:4]),box_cxcywh_to_xyxy(unmatched_box[:,:4]),return_iou_only=True)

                                        if iou_div_all.sum() == 0:
                                            match_ind = torch.randint(low=0, high=len(potential_object_query_indices), size=(1,), dtype=torch.int)
                                        else:
                                            match_ind = torch.argmax(iou_div_all,axis=0).to('cpu')

                                        potential_object_query_ind = potential_object_query_indices[match_ind]
                                        assert potential_object_query_ind not in ind_out
                                        ind_out[ind_out == -1] = potential_object_query_ind

                                    assert -1 not in ind_out
                                    assert len(ind_out) == len(ind_tgt)
                                    assert len(cur_target['boxes']) == len(cur_target['labels'])

                                    fut_track_id_1_ind = man_track[:,0] == fut_track_id_1
                                    fut_track_id_2_ind = man_track[:,0] == fut_track_id_2

                                    man_track[track_id_ind,2] -= 1
                                    man_track[fut_track_id_1_ind,1] -= 1
                                    man_track[fut_track_id_2_ind,1] -= 1

                                    if man_track[track_id_ind,1] > man_track[track_id_ind,2]:
                                        man_track[track_id_ind,1:] = -1
                                        man_track[fut_track_id_1_ind,-1] = 0
                                        man_track[fut_track_id_2_ind,-1] = 0

        if training_method == 'dn_object':
            if cur_target['num_FPs'] > 0:
                cur_target['track_queries_fal_pos_mask'][:-cur_target['num_FPs']][cur_target['track_ids'] == -1] = True
            else:
                cur_target['track_queries_fal_pos_mask'][cur_target['track_ids'] == -1] = True

        cur_target['boxes'] = cur_target['boxes'][ind_tgt[ind_keep].sort()[0]]
        cur_target['labels'] = cur_target['labels'][ind_tgt[ind_keep].sort()[0]]
        cur_target['track_ids'] = cur_target['track_ids'][ind_tgt[ind_keep].sort()[0]]
        cur_target['flexible_divisions'] = cur_target['flexible_divisions'][ind_tgt[ind_keep].sort()[0]]
        cur_target['is_touching_edge'] = cur_target['is_touching_edge'][ind_tgt[ind_keep].sort()[0]]

        if use_masks:
            cur_target['masks'] = cur_target['masks'][ind_tgt[ind_keep].sort()[0]]

        if 'track_query_match_ids' in cur_target and training_method != 'dn_object': # This needs to be updated for aux outputs and enc outputs because the matcher is rerun then
            prev_track_ids = prev_target['track_ids'][cur_target['prev_ind'][1]]

            # match track ids between frames
            target_ind_match_matrix = prev_track_ids.unsqueeze(dim=1).eq(cur_target['track_ids'])
            cur_target['target_ind_matching'] = target_ind_match_matrix.any(dim=1)
            cur_target['track_query_match_ids'] = target_ind_match_matrix.nonzero()[:, 1]

            # For images with no cells in them, we reformat target_ind_matching so torch.cat works properly with zero cells
            if cur_target['target_ind_matching'].shape[0] == 0:
                cur_target['target_ind_matching'] = torch.tensor([],device=device).bool()

            track_queries_mask = torch.ones_like(cur_target['target_ind_matching']).bool()

            cur_target['track_queries_mask'] = torch.cat([
                track_queries_mask,
                torch.tensor([True,] * cur_target['num_FPs']).to(device),
                torch.tensor([False,] * num_queries).to(device), 
                ]).bool()

            cur_target['track_queries_TP_mask'] = torch.cat([
                cur_target['target_ind_matching'],
                torch.tensor([False,] * cur_target['num_FPs']).to(device),
                torch.tensor([False,] * num_queries).to(device), 
                ]).bool()

            cur_target['track_queries_fal_pos_mask'] = torch.cat([
                ~cur_target['target_ind_matching'],
                torch.tensor([True,] * cur_target['num_FPs']).to(device),
                torch.tensor([False,] * num_queries).to(device),
                ]).bool()

            assert cur_target['track_queries_TP_mask'].sum() == len(cur_target['track_query_match_ids'])

        ind_out = ind_out[ind_keep]
        ind_tgt = ind_tgt[ind_keep]

        while not torch.arange(len(ind_tgt))[:,None].eq(ind_tgt[None]).any(0).all():
            for i in range(len(ind_tgt)):
                if i not in ind_tgt:
                    ind_tgt[ind_tgt > i] = ind_tgt[ind_tgt > i] - 1

        indices[t] = (ind_out,ind_tgt)
        targets[t][training_method]['man_track'] = man_track

    return targets, indices