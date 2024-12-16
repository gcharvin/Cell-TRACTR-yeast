# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Data visualizaiton functions.

"""
import cv2 
import numpy as np
import torch
from .box_ops import box_cxcy_to_xyxy

class data_visualizer():

    def __init__(self,height,width,args,colors,fontscale=0.4,alpha=0.3,mask_threshold=0.5):
        self.height = height
        self.width = width
        self.num_queries = args.num_queries
        self.dataset = args.dataset
        self.fontscale = fontscale
        self.alpha = alpha
        self.display_all = args.display_all
        self.cls_threshold = args.cls_threshold
        self.mask_threshold = mask_threshold
        self.colors = colors
        self.display_edge_cells = False

        self.enc_thresholds = [0, 0.1 ,0.3 ,0.5 , 0.8, 1.0]

    def draw_bbox(self,img,bounding_box,color=None,pred_logit=None,thickness=1,flex_div=False,edge=False):

        if color is None:
            color = (128,128,128)

        x0 = int(np.clip(bounding_box[0],0,self.width))
        y0 = int(np.clip(bounding_box[1],0,self.height))

        x1 = int(np.clip(bounding_box[0] + bounding_box[2],0,self.width))
        y1 = int(np.clip(bounding_box[1] + bounding_box[3],0,self.height))

        img = cv2.rectangle(
            img,
            (x0,y0),
            (x1,y1),
            color=color,
            thickness=thickness
            )
        
        if flex_div:
            centroid = (int(np.clip(bounding_box[0] + bounding_box[2] / 2,0,self.width)), int(np.clip(bounding_box[1] + bounding_box[3] / 2,0,self.height)))
            img = cv2.circle(img, centroid, radius=2, color=(255,255,255), thickness=-1)

        if edge and self.display_edge_cells:
            centroid = (int(np.clip(bounding_box[0] + bounding_box[2] / 2,0,self.width)), int(np.clip(bounding_box[1] + bounding_box[3] / 2,0,self.height)))
            img = cv2.circle(img, centroid, radius=4, color=(0,0,0), thickness=1)            
        
        if pred_logit is not None:
            org = (int(bounding_box[0]) - 5, int(bounding_box[1] + bounding_box[3] // 4 + int(self.fontscale*30))) if self.dataset == 'moma' else (int(bounding_box[0]), int(bounding_box[1]))
            img = cv2.putText(
                img,
                text = f'{pred_logit}', 
                org=org, 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale = self.fontscale,
                color = color,
                thickness=1,
            )
        
        return img
    
    def draw_mask(self,img,mask,color=None):

        if mask.shape != (self.height,self.width):
            mask = cv2.resize(mask,(self.width,self.height))
        mask = np.repeat(mask[...,None],3,axis=-1)

        if np.min(mask) < 0: 
            mean = 0
        elif np.max(mask) <= 1:
            mean = 0.5
        else:
            mean = 127

        mask_color = mask.copy()
            
        mask_color[mask[...,0]>mean] = color

        img[mask>mean] = img[mask>mean]*(1-self.alpha) + mask_color[mask>mean]*(self.alpha)

        return img

    def filter_pred_masks(self,pred_masks):

        if pred_masks.shape[0] > 0:
            masks_filt = np.zeros((pred_masks.shape))
            argmax = np.argmax(pred_masks,axis=0)
            for m in range(pred_masks.shape[0]):
                masks_filt[m,argmax==m] = pred_masks[m,argmax==m]
            masks_filt = masks_filt > self.mask_threshold
            pred_masks = (masks_filt*255).astype(np.uint8)

        return pred_masks

    def bbox_cxcy_to_xyxy(self,boxes):
        return box_cxcy_to_xyxy(boxes,self.height,self.width)
    
    def preprocess_img(self,target,image_name):

        img = target[image_name].permute(1,2,0)
        img = img.detach().cpu().numpy().copy()
        img = np.repeat(np.mean(img,axis=-1)[:,:,np.newaxis],3,axis=-1)
        img = (255*(img - np.min(img)) / np.ptp(img)).astype(np.uint8)

        return img

    def get_min_max_track_id(self,targets):

        min_track_ids = []
        max_track_ids = []

        for target in targets:

            target_names = ['prev_prev_target','prev_target','cur_target']
            if 'fut_target' in target:
                target_names += 'fut_target'
            
            empty = torch.tensor([target['main'][name]['empty'] for name in target_names]).sum().item()

            if empty < len(target_names):
                all_track_ids = torch.cat([target['main'][name]['track_ids_orig'] for name in target_names if target['main'][name]['track_ids_orig'].ndim == 1])
                min_track_id = all_track_ids.min().item()
                max_track_id = all_track_ids.max().item() + 1
            else:
                min_track_id = 0
                max_track_id = self.num_queries + 1

            min_track_ids.append(min_track_id)
            max_track_ids.append(max_track_id)

        self.min_track_ids = min_track_ids
        self.max_track_ids = max_track_ids

        return min_track_ids, max_track_ids
        
    def stack_enc_preds(self,enc_preds,enc_pred_logits, return_indices=False):
        '''
        enc_pred: bboxes or masks
        '''
                    
        enc_list = []

        if return_indices:
            ind_list = []

        for e in range(len(self.enc_thresholds)-1):
            low_thresh = self.enc_thresholds[e]
            high_thresh = self.enc_thresholds[e+1]
            enc_list.append(enc_preds[(enc_pred_logits > low_thresh) * (enc_pred_logits < high_thresh)])

            if return_indices:
                ind_list.append([ind for ind in np.where((enc_pred_logits > low_thresh) * (enc_pred_logits < high_thresh))[0]])
    
        if return_indices:
            return enc_list,ind_list
        
        return enc_list

    def display_main_pred(self,out,target,i,target_name, prev_target_name=''):

        min_track_id, max_track_id = self.min_track_ids[i], self.max_track_ids[i]

        img_pred_raw = self.preprocess_img(target,target_name.replace('target','image')) 

        use_masks = 'pred_masks' in out

        pred_boxes_all = out['pred_boxes'][i].detach().cpu().numpy()
        pred_logits_all = out['pred_logits'][i].sigmoid().detach().cpu().numpy()

        indices = target['main'][target_name]['indices']
        
        if 'track_ids_track' in target['main'][target_name]:
            track_ids = target['main'][target_name]['track_ids_track'][indices[1]].cpu().numpy()
        else:
            track_ids = target['main'][target_name]['track_ids'][indices[1]].cpu().numpy()

        pred_boxes_all = self.bbox_cxcy_to_xyxy(pred_boxes_all)

        keep = pred_logits_all[:,0] > self.cls_threshold
        pred_boxes = pred_boxes_all[keep]
        pred_logits = pred_logits_all[keep]

        div_keep = keep * (pred_logits_all[:,1] > self.cls_threshold) * (target['main'][target_name]['track_queries_mask'].cpu().numpy())

        if use_masks:
            pred_masks_all = out['pred_masks'][i].sigmoid().detach().cpu().numpy()

            filter_pred_masks = np.concatenate((pred_masks_all[keep,0],pred_masks_all[div_keep,1]))
            filter_pred_masks = self.filter_pred_masks(filter_pred_masks)

            pred_masks_all[keep,0] = filter_pred_masks[:keep.sum()]
            pred_masks_all[div_keep,1] = filter_pred_masks[keep.sum():]

            pred_masks = pred_masks_all[keep]

        # Display all boxes with a cls pred over the threshold (0.5)
        for idx,pred_bbox in enumerate(pred_boxes):

            flex_div = edge = False
            ind = np.where(keep)[0][idx]

            if ind in indices[0]: # Check if cell was matched with a gt
                if target['main'][target_name]['flexible_divisions'][indices[1]][indices[0] == ind].sum() > 0:
                    flex_div = True
                elif target['main'][target_name]['is_touching_edge'][indices[1]][indices[0] == ind].sum() > 0:
                    edge = True

            # Checks if pred_bbox is a match with a gt - then assign it a color based on the cell number
            if ind in indices[0]:
                #TODO - hard to debug - when prev_track_ids is len of 1, it is not an array 
                if len(track_ids) == 1:
                    color = self.colors[track_ids[0] - min_track_id]
                elif len(indices[0]) == 1:
                    color = self.colors[track_ids - min_track_id]
                else:
                    color = self.colors[track_ids[indices[0] == ind][0] - min_track_id]
            # Assign random color is pred_bbox does not match with any gt
            else:
                color = np.array([0.,0.,0.])

            # Display all predictions above cls threshold regardless of matching
            pred_logit = f'{pred_logits[idx,0]:.2f}' if self.display_all else None
            img_pred_raw = self.draw_bbox(img_pred_raw,pred_bbox[:4],color,pred_logit,flex_div=flex_div,edge=edge)

            if use_masks:
                img_pred_raw = self.draw_mask(img_pred_raw,pred_masks[idx,0],color)    

            if pred_logits[idx,1] > self.cls_threshold and target['main'][target_name]['track_queries_mask'][ind]:
                pred_logit = f'{pred_logits[idx,1]:.2f}' if self.display_all else None
                img_pred_raw = self.draw_bbox(img_pred_raw,pred_bbox[4:],color,pred_logit,flex_div=flex_div,edge=edge)

                if use_masks:
                    img_pred_raw = self.draw_mask(img_pred_raw,pred_masks[idx,1],color)

        res = img_pred_raw

        if 'updated_indices' in target['main'][target_name]:
            indices = target['main'][target_name]['updated_indices'] 
            track_ids = target['main'][target_name]['track_ids_orig'][indices[1]].cpu().numpy()

        if len(indices[0]) > 0:

            img_pred_match = self.preprocess_img(target,target_name.replace('target','image')) 

            pred_match_bboxes = np.atleast_2d(pred_boxes_all[indices[0]])
            pred_match_logits = np.atleast_2d(pred_logits_all[indices[0]])

            if use_masks:
                pred_match_masks = pred_masks_all[indices[0]]
                pred_match_masks = pred_match_masks[None] if pred_match_masks.ndim ==3 else pred_match_masks

            indices_unique = np.unique(indices[0].numpy())

            for out_ind in indices_unique:
                if (indices[0] == out_ind).sum() == 2:
                    inds = np.where(indices[0] == out_ind)[0]
                    assert len(inds) == 2 and (inds[0] == inds[1]+1 or inds[0]+1 == inds[1])
                    ind = inds.max() # match the order the indices of the divided cells are split
                    pred_match_bboxes[ind,:4] = pred_match_bboxes[ind,4:]
                    pred_match_logits[ind,0] = pred_match_logits[ind,1]

                    if use_masks:
                        pred_match_masks[ind,:1] = pred_match_masks[ind,1:]

            for idx,pred_match_bbox in enumerate(pred_match_bboxes):

                flex_div = edge = False

                if target['main'][target_name]['flexible_divisions'][indices[1]][idx]:
                    flex_div = True
                elif target['main'][target_name]['is_touching_edge'][indices[1]][idx]:
                    edge = True

                pred_logit = f'{pred_match_logits[idx,0]:.2f}'
                color = self.colors[track_ids[idx] - min_track_id]
                img_pred_match = self.draw_bbox(img_pred_match,pred_match_bbox[:4],color,pred_logit,flex_div=flex_div,edge=edge)
                
                if use_masks:
                    img_pred_match = self.draw_mask(img_pred_match,pred_match_masks[idx,0],color)   

                # For current target, we don't separate divided cells but for prev target, we do
                if target_name == 'cur_target' and pred_match_logits[idx,1] > self.cls_threshold and target['main'][target_name]['boxes'][indices[1][idx],-1] > 0:
                    pred_logit = f'{pred_match_logits[idx,1]:.2f}' if self.display_all else None
                    img_pred_match = self.draw_bbox(img_pred_match,pred_match_bbox[4:],color,pred_logit,flex_div=flex_div,edge=edge)

                    if use_masks:
                        img_pred_match = self.draw_mask(img_pred_match,pred_match_masks[idx,1],color)

            res = np.concatenate((res,img_pred_match),1)

        if 'track_query_boxes' in target['main'][target_name]:

            img_track_query_boxes = self.preprocess_img(target,prev_target_name.replace('target','image')) 

            track_query_boxes = target['main'][target_name]['track_query_boxes'].cpu().numpy()
            track_query_boxes = self.bbox_cxcy_to_xyxy(track_query_boxes)

            indices = target['main'][target_name]['prev_ind']
            track_query_ids = target['main'][prev_target_name]['track_ids'][indices[1]].cpu().numpy()

            if len(track_query_ids) > 0:
                track_query_ids = np.concatenate((track_query_ids,np.array([max_track_id+idx for idx in range(track_query_boxes.shape[0]-len(track_query_ids))]).astype(int))) # account for FPs
            else:
                track_query_ids = np.array([max_track_id+idx for idx in range(track_query_boxes.shape[0]-len(track_query_ids))])

            for idx,track_query_box in enumerate(track_query_boxes):

                flex_div = edge = False

                if idx < len(indices[1]):
                    if target['main'][prev_target_name]['flexible_divisions'][indices[1]][idx]:
                        flex_div = True
                    elif target['main'][prev_target_name]['is_touching_edge'][indices[1]][idx]:
                        edge = True

                color = self.colors[track_query_ids[idx] - min_track_id] if idx < len(track_query_ids) else self.colors[-idx]
                img_track_query_boxes = self.draw_bbox(img_track_query_boxes,track_query_box,color,flex_div=flex_div,edge=edge)

            res = np.concatenate((img_track_query_boxes,res),1)

        return res

    def display_two_stage(self,out,target,i,target_name):

        img = self.preprocess_img(target,target_name.replace('target','image')) 

        two_stage_outputs = out['two_stage']
        use_masks = 'pred_masks' in two_stage_outputs
        enc_pred_logits_topk = two_stage_outputs['pred_logits'][i,:,0].detach().cpu().sigmoid().numpy()
        enc_pred_boxes_topk = two_stage_outputs['pred_boxes'][i,:,:4].detach().cpu().numpy()

        keep = enc_pred_logits_topk > self.cls_threshold

        enc_pred_logits = enc_pred_logits_topk[keep]
        enc_pred_boxes = enc_pred_boxes_topk[keep]
        enc_pred_boxes = self.bbox_cxcy_to_xyxy(enc_pred_boxes)

        if use_masks:
            enc_pred_masks_topk = two_stage_outputs['pred_masks'][i,:,0].detach().cpu().numpy()
            enc_pred_masks = enc_pred_masks_topk[keep]
            enc_pred_masks = self.filter_pred_masks(enc_pred_masks)

        indices = target['main'][target_name]['indices']
        track_ids = target['main'][target_name]['track_ids'][indices[1]].cpu().numpy()

        for idx, enc_pred_box in enumerate(enc_pred_boxes):

            # If two-stage enc query is used for object detection, we use the color corresponding with hte gt otherwise, we use a random color
            ind = np.where(keep)[0][idx]
            num_TQs = target['main'][target_name]['track_queries_mask'][ind].sum().item()
            
            if ind + num_TQs in indices[0] and not target['main'][target_name]['track_queries_mask'][ind + num_TQs]:
                color = self.colors[track_ids[indices[0].numpy() == (ind+num_TQs)] - self.min_track_ids[i]][0]
            else:
                color = self.colors[-idx]

            enc_pred_logit = f'{enc_pred_logits[idx]:.1f}'
            img = self.draw_bbox(img,enc_pred_box[:4],color,enc_pred_logit)

            if use_masks:
                img = self.draw_mask(img,enc_pred_masks[idx],color)

        if target['main'][target_name]['track_queries_mask'].sum() == 0:

            img_gt = self.preprocess_img(target,target_name.replace('target','image')) 

            if not target['main'][target_name]['empty']:
                boxes_gt = target['main'][target_name]['boxes'].cpu().numpy()
                track_ids = target['main'][target_name]['track_ids'].cpu().numpy()
                boxes_gt = self.bbox_cxcy_to_xyxy(boxes_gt)

                if use_masks:
                    masks_gt = target['main'][target_name]['masks'].cpu().numpy()

                for idx,bounding_box in enumerate(boxes_gt):
                    color=self.colors[track_ids[idx]-self.min_track_ids[i]]
                    img_gt = self.draw_bbox(img_gt,bounding_box[:4],color)

                    if use_masks:
                        img_gt = self.draw_mask(img_gt,masks_gt[idx,0],color)

            img = np.concatenate((img,img_gt),1)

        return img
    
    def display_CoMOT(self,aux_outputs,target,i):

        min_track_id, max_track_id = self.min_track_ids[i], self.max_track_ids[i]
        CoMOT_raw = self.preprocess_img(target,'cur_image')
        CoMOT_match = self.preprocess_img(target,'cur_image')

        use_masks = 'pred_masks' in aux_outputs[-1]
        CoMOT_res = np.zeros((self.height,0,3))

        pred_logits_all = aux_outputs[-1]['pred_logits'].detach().cpu().sigmoid().numpy()[i,-self.num_queries:]
        pred_boxes_all = aux_outputs[-1]['pred_boxes'].detach().cpu().numpy()[i,-self.num_queries:]
        CoMOT_indices = aux_outputs[-1]['CoMOT_indices'][i]
        CoMOT_track_ids = target['CoMOT']['cur_target']['track_ids'][CoMOT_indices[1]].cpu().numpy()

        keep = pred_logits_all[:,0] > self.cls_threshold
        keep_ind = np.where(keep)[0]

        if use_masks and 'pred_masks' in aux_outputs[-1]:
            pred_masks_all = aux_outputs[-1]['pred_masks'].detach().cpu().numpy()[i,-self.num_queries:]
            pred_masks = pred_masks_all[keep]
            pred_masks = self.filter_pred_masks(pred_masks)
            pred_masks_all[keep] = pred_masks

        pred_boxes_all = self.bbox_cxcy_to_xyxy(pred_boxes_all)

        for idx, pred_box in enumerate(pred_boxes_all):

            if idx in CoMOT_indices[0]:
                if len(CoMOT_track_ids) == 1:
                    color = self.colors[CoMOT_track_ids[0] - min_track_id]
                elif len(CoMOT_indices[0]) == 1:
                    color = self.colors[CoMOT_track_ids - min_track_id]
                else:
                    color = self.colors[CoMOT_track_ids[CoMOT_indices[0] == idx][0] - min_track_id]
            else:
                color = self.colors[-idx]

            pred_logit = f'{pred_logits_all[idx,0]:.2f}' if self.display_all and aux_outputs[-1]['CoMOT_loss_ce'] else None
            
            if aux_outputs[-1]['CoMOT_loss_ce'] and idx in keep_ind:
                CoMOT_raw = self.draw_bbox(CoMOT_raw,pred_box[:4],color,pred_logit)
                if use_masks and 'pred_masks' in aux_outputs[-1]:
                    CoMOT_raw = self.draw_mask(CoMOT_raw,pred_masks_all[idx,0],color)

            if idx in CoMOT_indices[0]:
                CoMOT_match = self.draw_bbox(CoMOT_match,pred_box[:4],color,pred_logit)
                if use_masks and 'pred_masks' in aux_outputs[-1]:
                    CoMOT_match = self.draw_mask(CoMOT_match,pred_masks_all[idx,0],color)        
            
        CoMOT_res = np.concatenate((CoMOT_res,CoMOT_match),1)
        
        if self.display_all and aux_outputs[-1]['CoMOT_loss_ce']:
            CoMOT_res = np.concatenate((CoMOT_raw,CoMOT_res),1)

        return CoMOT_res

def plot_results(outputs,targets,samples,savepath,filename,folder,args):

    if 'prev_outputs' in outputs:
        prev_outputs = outputs.pop('prev_outputs')
    else:
        prev_outputs = None

    if 'prev_prev_outputs' in outputs:
        prev_prev_outputs = outputs.pop('prev_prev_outputs')
    else:
        prev_prev_outputs = None

    print_gt = True
    bs,_,height,width = samples.shape
    # spacer = 10
    cls_threshold = args.cls_threshold
    mask_threshold = 0.5
    fontscale = 0.4
    alpha = 0.3
    use_masks = 'pred_masks' in outputs['main']
    # box_converter = box_cxcy_to_xyxy(height,width)
    colors = np.array([tuple((255*np.random.random(3))) for _ in range(10000)]) # Assume max 1000 cells in one chamber
    colors[:6] = np.array([[0.,0.,255.],[0.,255.,0.],[255.,0.,0.],[255.,0.,255.],[0.,255.,255.],[255.,255.,0.]])
    # colors = [tuple((255*np.random.random(3))) for _ in range(600)]
    # colors[:6] = [np.array(255.,0.,0.),np.array(0.,255.,0.),np.array(0.,0.,255.),np.array(255.,255.,0.),np.array(255.,0.,255.),np.array(0.,255.,255.),]

    viz = data_visualizer(height,width,args,colors,fontscale=fontscale,alpha=alpha,mask_threshold=mask_threshold)
    min_track_ids, max_track_ids = viz.get_min_max_track_id(targets)
    
    training_methods = list(outputs.keys())
    training_methods = [training_method for training_method in training_methods if training_method not in ['main','two_stageSS']]

    for training_method in training_methods:

        outputs_TM = outputs[training_method]
        targets_TM = [target[training_method]['cur_target'] for target in targets]

        if training_method == 'dn_track' or training_method == 'dn_track_group':

            dn_track_frame = np.zeros((height,0,3))

            for i,target in enumerate(targets):

                min_track_id, max_track_id = min_track_ids[i], max_track_ids[i]

                prev_track_ids = target['main']['prev_target']['track_ids'].cpu().numpy()
                dn_track_prev_ind = target['dn_track']['cur_target']['prev_ind'][1].cpu().numpy()
                
                prev_track_ids = prev_track_ids[dn_track_prev_ind]

                img = viz.preprocess_img(target,'cur_image')              
                prev_img = viz.preprocess_img(target,'prev_image')              
                
                unoised_boxes = targets_TM[i]['track_query_boxes_gt'].cpu().numpy()
                noised_boxes = targets_TM[i]['track_query_boxes'].cpu().numpy()
                gt_boxes = targets_TM[i]['boxes'].cpu().numpy()

                if use_masks:
                    gt_masks = targets_TM[i]['masks'].cpu().numpy()
                    gt_masks = viz.filter_pred_masks(gt_masks)

                unoised_boxes = viz.bbox_cxcy_to_xyxy(unoised_boxes)
                noised_boxes = viz.bbox_cxcy_to_xyxy(noised_boxes)
                gt_boxes = viz.bbox_cxcy_to_xyxy(gt_boxes)

                TP_mask = targets_TM[i]['track_queries_TP_mask']

                previmg_unoised_boxes = prev_img.copy()
                previmg_noised_boxes = prev_img.copy()

                indices = target[training_method]['cur_target']['indices']

                for idx,noised_box in enumerate(noised_boxes):

                    flex_div = edge = False
                        
                    if idx in indices[0]:
                        if target[training_method]['cur_target']['flexible_divisions'][indices[1]][[indices[0] == idx]]:
                            flex_div = True
                        elif target[training_method]['cur_target']['is_touching_edge'][indices[1]][[indices[0] == idx]]:
                            edge = True

                    color=colors[prev_track_ids[idx] - min_track_id] if TP_mask[idx] else (0,0,0)
                    previmg_noised_boxes = viz.draw_bbox(previmg_noised_boxes,noised_box[:4],color,flex_div=flex_div,edge=edge)

                pred_boxes_all = outputs_TM['pred_boxes'][i].detach().cpu().numpy()
                pred_logits_all = outputs_TM['pred_logits'][i].detach().sigmoid().cpu().numpy()

                pred_boxes_all = viz.bbox_cxcy_to_xyxy(pred_boxes_all)

                keep = pred_logits_all[:,0] > cls_threshold
                where = np.where(pred_logits_all[:,0] > cls_threshold)[0]
                pred_logits = pred_logits_all[keep]
                pred_boxes = pred_boxes_all[keep]

                img_pred = img.copy()
                img_pred_all = img.copy()

                if use_masks:
                    pred_masks_all = outputs_TM['pred_masks'][i].detach().cpu().numpy()
                    pred_masks = pred_masks_all[keep]
                    pred_masks = viz.filter_pred_masks(pred_masks)
                    pred_masks_all[keep] = pred_masks

                indices = targets_TM[i]['indices']

                for idx,pred_box in enumerate(pred_boxes_all):
                    color=colors[prev_track_ids[idx] - min_track_id] if TP_mask[idx] else (0,0,0)

                    flex_div = edge = False
                    if idx in indices[0]:
                        if target[training_method]['cur_target']['flexible_divisions'][indices[1]][indices[0] == idx].sum() > 0:
                            flex_div = True
                        elif target[training_method]['cur_target']['is_touching_edge'][indices[1]][indices[0] == idx].sum() > 0:
                            edge = True

                    img_pred_all = viz.draw_bbox(img_pred_all,pred_box[:4],color,flex_div=flex_div,edge=edge)

                    if pred_logits_all[idx,1] > cls_threshold:
                        img_pred_all = viz.draw_bbox(img_pred_all,pred_box[4:],color,flex_div=flex_div,edge=edge)
  
                    if idx in where:
                        pred_logit = f'{pred_logits_all[idx,0]:.2f}' if args.display_all else None
                        img_pred = viz.draw_bbox(img_pred,pred_box[:4],color,pred_logit,flex_div=flex_div,edge=edge)

                        if use_masks:
                            img_pred = viz.draw_mask(img_pred,pred_masks_all[idx,0],color)

                        if pred_logits_all[idx,1] > cls_threshold:

                            pred_logit = f'{pred_logits_all[idx,1]:.2f}' if args.display_all else None
                            img_pred = viz.draw_bbox(img_pred,pred_box[4:],color,pred_logit,flex_div=flex_div,edge=edge)

                            if use_masks:
                                img_pred = viz.draw_mask(img_pred,pred_masks_all[idx,1],color)     

                img_gt = img.copy()

                track_ids = targets_TM[i]['track_ids'].cpu().numpy()

                for idx,gt_box in enumerate(gt_boxes):

                    flex_div = edge = False
                    if target[training_method]['cur_target']['flexible_divisions'][idx]:
                        flex_div = True
                    elif target[training_method]['cur_target']['is_touching_edge'][idx]:
                        edge = True
                        
                    color = colors[track_ids[idx] - min_track_id]
                    img_gt = viz.draw_bbox(img_gt,gt_box[:4],color,flex_div=flex_div,edge=edge)       

                    if use_masks:
                        img_gt = viz.draw_mask(img_gt,gt_masks[idx,0],color)              

                    if gt_box[-1] > 0:
                        img_gt = viz.draw_bbox(img_gt,gt_box[4:],color,flex_div=flex_div,edge=edge)  

                        if use_masks:
                            img_gt = viz.draw_mask(img_gt,gt_masks[idx,1],color)  

                if args.display_all:

                    orig_gt_boxes = target['target_og']['cur_target']['boxes'].cpu().numpy()
                    orig_gt_boxes = viz.bbox_cxcy_to_xyxy(orig_gt_boxes)

                    if use_masks:
                        orig_gt_masks = target['target_og']['cur_target']['masks'].cpu().numpy()
                        orig_gt_masks = viz.filter_pred_masks(orig_gt_masks)

                    orig_track_ids = target['target_og']['cur_target']['track_ids'].cpu().numpy()

                    orig_img_gt = img.copy()
                    man_track = target['target_og']['man_track']

                    for idx,orig_gt_box in enumerate(orig_gt_boxes):

                        edge = False
                        if target['target_og']['cur_target']['is_touching_edge'][idx]:
                            edge = True
                            
                        orig_track_id = orig_track_ids[idx]

                        if man_track[orig_track_id-1,1] == target['target_og']['cur_target']['framenb'] and  man_track[orig_track_id-1,-1] > 0:
                            orig_track_id = man_track[orig_track_id-1,-1]

                        color = colors[orig_track_id - min_track_id]

                        orig_img_gt = viz.draw_bbox(orig_img_gt,orig_gt_box[:4],color,edge=edge)       

                        if use_masks:
                            orig_img_gt = viz.draw_mask(orig_img_gt,orig_gt_masks[idx,0],color)              

                        if orig_gt_box[-1] > 0:
                            orig_img_gt = viz.draw_bbox(orig_img_gt,orig_gt_box[4:],color,edge=edge)  

                            if use_masks:
                                orig_img_gt = viz.draw_mask(orig_img_gt,orig_gt_masks[idx,1],color)  

                if args.display_all:
                    dn_track_frame = np.concatenate((dn_track_frame,previmg_unoised_boxes,previmg_noised_boxes,img_pred,img,img_pred_all,img_gt,orig_img_gt),axis=1)
                else:
                    dn_track_frame = np.concatenate((dn_track_frame,previmg_noised_boxes,img_pred,img_gt),axis=1)

                if i < len(targets) - 1:
                    dn_track_frame = np.concatenate((dn_track_frame,np.zeros((img.shape[0],20,3),dtype=img.dtype)),axis=1)

            cv2.imwrite(str(savepath / folder / training_method / filename),dn_track_frame)

        elif training_method == 'OD':

            OD_frames = []

            for t,target in enumerate(targets):

                if target['main']['cur_target']['empty']:
                    continue

                min_track_id, max_track_id = min_track_ids[t], max_track_ids[t]

                OD_indices = outputs[training_method]['indices'][t]
                
                OD_track_ids = target[training_method]['cur_target']['track_ids'][OD_indices[1]].cpu().numpy()
                OD_gt_boxes = target[training_method]['cur_target']['boxes'][OD_indices[1]].cpu().numpy()
                OD_gt_boxes = viz.bbox_cxcy_to_xyxy(OD_gt_boxes)

                if use_masks:
                    OD_gt_masks = target[training_method]['cur_target']['masks'][OD_indices[1]].cpu().numpy()

                pred_logits_all = outputs[training_method]['pred_logits'][t,:,0].sigmoid().detach().cpu().numpy()
                pred_boxes_all = outputs[training_method]['pred_boxes'][t,:,:4].detach().cpu().numpy()

                pred_boxes_all = viz.bbox_cxcy_to_xyxy(pred_boxes_all)
                keep = pred_logits_all > cls_threshold

                if use_masks:
                    pred_masks_all = outputs[training_method]['pred_masks'][t,:,0].detach().cpu().numpy()
                
                OD_img = viz.preprocess_img(target,'cur_image')  

                for idx,pred_box in enumerate(pred_boxes_all):

                    if keep[idx]:
                        flex_div = edge = False

                        if idx in OD_indices[0]:
                            ind = torch.where(OD_indices[0] == idx)[0][0]
                            color=colors[OD_track_ids[ind] - min_track_id]
                            if target[training_method]['cur_target']['flexible_divisions'][OD_indices[1]][ind].sum() > 0:
                                flex_div = True
                            elif target[training_method]['cur_target']['is_touching_edge'][OD_indices[1]][ind].sum() > 0:
                                edge = True
                        else:
                            color=(0,0,0)

                        pred_logit = f'{pred_logits_all[idx]:.1f}'

                        OD_img = viz.draw_bbox(OD_img,pred_box,color,pred_logit,flex_div=flex_div,edge=edge)

                        if use_masks:
                            OD_img = viz.draw_mask(OD_img,pred_masks_all[idx],color)                           

                OD_frames.append(OD_img)

                OD_gt_img = viz.preprocess_img(target,'cur_image')  

                for idx, OD_gt_box in enumerate(OD_gt_boxes):

                    flex_div = edge = False

                    color=colors[OD_track_ids[idx] - min_track_id]

                    if target[training_method]['cur_target']['flexible_divisions'][OD_indices[1]][idx].sum() > 0:
                        flex_div = True
                    elif target[training_method]['cur_target']['is_touching_edge'][OD_indices[1]][idx].sum() > 0:
                        edge = True

                    OD_gt_img = viz.draw_bbox(OD_gt_img,OD_gt_box[:4],color,flex_div=flex_div,edge=edge)

                    if use_masks:
                        OD_gt_img = viz.draw_mask(OD_gt_img,OD_gt_masks[idx,0],color)  

                OD_frames.append(OD_gt_img)

            if len(OD_frames) > 0:
                OD_frames = np.concatenate((OD_frames),axis=1)

                cv2.imwrite(str(savepath / folder / training_method / filename),OD_frames)

        elif training_method == 'dn_enc':
            dn_encs = []
            for i,target in enumerate(targets):
                
                min_track_id, max_track_id = min_track_ids[i], max_track_ids[i]

                enc_boxes_noised = targets_TM[i]['enc_boxes_noised'].cpu().numpy()
                enc_boxes = targets_TM[i]['enc_boxes'].cpu().numpy()
                gt_boxes = targets_TM[i]['boxes'].cpu().numpy()
                track_ids = targets_TM[i]['track_ids'].cpu().numpy()
                indices = [ind.cpu().numpy() for ind in targets_TM[i]['indices']]

                if use_masks:
                    gt_masks = targets_TM[i]['masks'][:,0].cpu().numpy()

                enc_boxes_noised = viz.bbox_cxcy_to_xyxy(enc_boxes_noised)
                enc_boxes = viz.bbox_cxcy_to_xyxy(enc_boxes)
                gt_boxes = viz.bbox_cxcy_to_xyxy(gt_boxes)

                img = viz.preprocess_img(target,'cur_image')

                img_gt = img.copy()
                img_enc_boxes_noised = img.copy()
                img_enc_boxes_noised_match = img.copy()
                img_enc_boxes = img.copy()
                img_enc_boxes_match = img.copy()
                img_pred_thresh = img.copy()
                img_pred_all = img.copy()
                img_pred_match = img.copy()

                for idx, gt_box in enumerate(gt_boxes):

                    color = colors[track_ids[idx] - min_track_id]
                    img_gt = viz.draw_bbox(img_gt,gt_box[:4],color)

                    if use_masks:
                        img_gt = viz.draw_mask(img_gt,gt_masks[idx],color)

                pred_boxes_all = outputs_TM['pred_boxes'][i].detach().cpu().numpy()
                pred_logits_all = outputs_TM['pred_logits'][i].detach().sigmoid().cpu().numpy()

                pred_boxes_all = viz.bbox_cxcy_to_xyxy(pred_boxes_all)

                keep = pred_logits_all[:,0] > cls_threshold
                where = np.where(keep)[0]
                pred_boxes = pred_boxes_all[keep]
                pred_logits = pred_logits_all[keep]

                if use_masks:
                    pred_masks_all = outputs_TM['pred_masks'][i,:,0].detach().sigmoid().cpu().numpy()
                    pred_masks = pred_masks_all[keep]

                    if sum(keep) > 0:
                        pred_masks = viz.filter_pred_masks(pred_masks)

                    pred_masks_all[keep] = pred_masks

                for idx,(enc_box,enc_box_noised,pred_box) in enumerate(zip(enc_boxes,enc_boxes_noised,pred_boxes_all)):
                        
                    if idx in indices[0]:
                        color = colors[track_ids[indices[1]][indices[0] == idx][0] - min_track_id]
                    else:
                        color = colors[-idx]

                    img_enc_boxes_noised = viz.draw_bbox(img_enc_boxes_noised,enc_box_noised[:4],color)
                    img_pred_all = viz.draw_bbox(img_pred_all,pred_box[:4],color)

                    img_enc_boxes = viz.draw_bbox(img_enc_boxes,enc_box[:4],color)

                    if idx in indices[0]:
                        pred_logit = f'{targets_TM[i]["enc_logits"][idx]:.2f}'
                        img_enc_boxes_match = viz.draw_bbox(img_enc_boxes_match,enc_box[:4],color,pred_logit)

                        img_enc_boxes_noised_match = viz.draw_bbox(img_enc_boxes_noised_match,enc_box_noised[:4],color)

                    pred_logit = f'{pred_logits_all[idx,0]:.2f}' if args.display_all else None
                    if idx in where:
                        img_pred_thresh = viz.draw_bbox(img_pred_thresh,pred_box[:4],color,pred_logit)

                        if use_masks:
                            img_pred_thresh = viz.draw_mask(img_pred_thresh,pred_masks_all[idx],color)

                    if idx in indices[0]:
                        img_pred_match = viz.draw_bbox(img_pred_match,pred_box[:4],color,pred_logit)

                        if use_masks:
                            img_pred_match = viz.draw_mask(img_pred_match,pred_masks_all[idx],color)

                if args.display_all:
                    dn_encs.append(np.concatenate((img,img_enc_boxes,img_enc_boxes_match,img_enc_boxes_noised,img_enc_boxes_noised_match,img_pred_thresh,img_pred_match,img_gt),axis=1))
                else:
                    dn_encs.append(np.concatenate((img,img_enc_boxes_noised_match,img_pred_thresh,img_gt),axis=1))

            if len(dn_encs) == 1:
                dn_enc = dn_encs[0]
            else:
                dn_enc = np.concatenate(dn_encs,1)

            cv2.imwrite(str(savepath / folder / 'dn_enc' / filename),dn_enc)

        elif training_method == 'dn_object':

            dn_object_frame = np.zeros((height,0,3))

            for i,target_TM in enumerate(targets_TM):

                min_track_id, max_track_id = min_track_ids[i], max_track_ids[i]

                noised_boxes = target_TM['noised_boxes'].cpu().numpy()
                gt_boxes = target_TM['boxes'].cpu().numpy()
                TP_mask = ~target_TM['track_queries_fal_pos_mask'].cpu().numpy()
                where_TP_mask = np.where(TP_mask)[0]
                track_ids = target_TM['track_ids'].cpu().numpy()
                indices = (target_TM['indices'][0].cpu().numpy(),target_TM['indices'][1].cpu().numpy())
            
                gt_boxes = viz.bbox_cxcy_to_xyxy(gt_boxes)
                noised_boxes = viz.bbox_cxcy_to_xyxy(noised_boxes)

                cur_img = viz.preprocess_img(targets[i],'cur_image')

                cur_img_gt = cur_img.copy()
                cur_img_noised = cur_img.copy()

                for idx,gt_box in enumerate(gt_boxes):

                    flex_div = edge = False
                        
                    if target_TM['flexible_divisions'][idx]:
                        flex_div = True
                    elif target_TM['is_touching_edge'][idx]:
                        edge = True

                    color = colors[track_ids[idx] - min_track_id]
                    cur_img_gt = viz.draw_bbox(cur_img_gt,gt_box[:4],color,flex_div=flex_div,edge=edge)

                TP_noised_boxes = noised_boxes[TP_mask]
                for idx,noised_box in enumerate(TP_noised_boxes):

                    flex_div = edge = False
                        
                    if target_TM['flexible_divisions'][idx]:
                        flex_div = True
                    elif target_TM['is_touching_edge'][idx]:
                        edge = True

                    color = colors[track_ids[idx] - min_track_id]

                    cur_img_noised = viz.draw_bbox(cur_img_noised,noised_box[:4],color,flex_div=flex_div,edge=edge)

                FP_noised_boxes = noised_boxes[~TP_mask]
                
                for idx,noised_box in enumerate(FP_noised_boxes):
                    color = (0,0,0)
                    cur_img_noised = viz.draw_bbox(cur_img_noised,noised_box[:4],color,flex_div=flex_div,edge=edge)


                pred_boxes_all = outputs_TM['pred_boxes'][i].detach().cpu().numpy()
                pred_logits_all = outputs_TM['pred_logits'][i].detach().cpu().sigmoid().numpy()

                pred_boxes_all = viz.bbox_cxcy_to_xyxy(pred_boxes_all)

                keep = pred_logits_all[:,0] > cls_threshold
                where = np.where(keep)[0]
                pred_boxes = pred_boxes_all[keep]
                pred_logits = pred_logits_all[keep]

                if use_masks:
                    pred_masks_all = outputs_TM['pred_masks'][i].detach().cpu().sigmoid().numpy()
                    pred_masks = pred_masks_all[keep]

                    if sum(keep) > 0:
                        pred_masks = viz.filter_pred_masks(pred_masks)

                    pred_masks_all[keep] = pred_masks

                cur_img_pred_all = cur_img.copy()
                cur_img_pred = cur_img.copy()


                pred_boxes_TP = pred_boxes_all[indices[0]]
                pred_masks_TP = pred_masks_all[indices[0]]
                pred_logits_TP = pred_logits_all[indices[0]]

                for idx,pred_box in enumerate(pred_boxes_TP):
                    
                    flex_div = edge = False

                    if target_TM['flexible_divisions'][indices[1]][idx]:
                        flex_div = True
                    elif target_TM['is_touching_edge'][indices[1]][idx]:
                        edge = True

                    color = colors[track_ids[idx] - min_track_id]

                    cur_img_pred_all = viz.draw_bbox(cur_img_pred_all,pred_box[:4],color,flex_div=flex_div,edge=edge)

                    if keep[indices[0]][idx]:
                        pred_logit = f'{pred_logits_TP[idx,0]:.2f}' if args.display_all else None
                        cur_img_pred = viz.draw_bbox(cur_img_pred,pred_box[:4],color,pred_logit,flex_div=flex_div,edge=edge)

                        if use_masks:
                            cur_img_pred = viz.draw_mask(cur_img_pred,pred_masks_TP[idx,0].squeeze(),color)

                indices_FP = np.array([idx for idx in range(pred_boxes_all.shape[0]) if idx not in indices[0]])

                if len(indices_FP) > 0:
                    pred_boxes_FP = pred_boxes_all[indices_FP]
                    pred_masks_FP = pred_masks_all[indices_FP]
                    pred_logits_FP = pred_logits_all[indices_FP]

                    for idx,pred_box in enumerate(pred_boxes_FP):
                        color = (0,0,0)
                        cur_img_pred_all = viz.draw_bbox(cur_img_pred_all,pred_box[:4],color,flex_div=flex_div,edge=edge)

                        if keep[indices_FP][idx]:
                            pred_logit = f'{pred_logits_FP[idx,0]:.2f}' if args.display_all else None
                            cur_img_pred = viz.draw_bbox(cur_img_pred,pred_box[:4],color,pred_logit,flex_div=flex_div,edge=edge)

                            if use_masks:
                                cur_img_pred = viz.draw_mask(cur_img_pred,pred_masks_FP[idx,0].squeeze(),color)

                if args.display_all:
                    orig_gt_boxes = targets[i]['target_og']['cur_target']['boxes'].cpu().numpy()
                    orig_gt_boxes = viz.bbox_cxcy_to_xyxy(orig_gt_boxes)

                    if use_masks:
                        orig_gt_masks = targets[i]['target_og']['cur_target']['masks'].cpu().numpy()
                        orig_gt_masks = viz.filter_pred_masks(orig_gt_masks)

                    orig_track_ids = targets[i]['target_og']['cur_target']['track_ids'].cpu().numpy()

                    orig_img_gt = cur_img.copy()
                    man_track = targets[i]['target_og']['man_track']

                    for idx,orig_gt_box in enumerate(orig_gt_boxes):

                        edge = False
                        if targets[i]['target_og']['cur_target']['is_touching_edge'][idx]:
                            edge = True
                            
                        orig_track_id = orig_track_ids[idx]

                        color = colors[orig_track_id - min_track_id]

                        orig_img_gt = viz.draw_bbox(orig_img_gt,orig_gt_box[:4],color,edge=edge)       

                        if use_masks:
                            orig_img_gt = viz.draw_mask(orig_img_gt,orig_gt_masks[idx,0],color)  

                if args.display_all:
                    dn_object_frame = np.concatenate((dn_object_frame,cur_img_noised,cur_img_pred,cur_img,cur_img_gt,orig_img_gt),axis=1)
                else:
                    dn_object_frame = np.concatenate((dn_object_frame,cur_img_noised,cur_img_pred,cur_img_gt),axis=1)

                if i < len(targets_TM) - 1:
                    dn_object_frame = np.concatenate((dn_object_frame,np.zeros((height,10,3))),axis=1)

            cv2.imwrite(str(savepath / folder / 'dn_object' / filename),dn_object_frame)


    for i,target in enumerate(targets):

        res = np.zeros((height,0,3))
        two_stage_res = np.zeros((height,0,3))

        if prev_prev_outputs is not None:
            prev_prev_res = viz.display_main_pred(prev_prev_outputs,target,i,'prev_prev_target')
            res = np.concatenate((res,prev_prev_res),1)

            if 'two_stage' in prev_prev_outputs:
                prev_prev_two_stage_res = viz.display_two_stage(prev_prev_outputs,target,i,'prev_prev_target')
                two_stage_res = np.concatenate((two_stage_res,prev_prev_two_stage_res),1)

        if prev_outputs is not None:
            prev_res = viz.display_main_pred(prev_outputs,target,i,'prev_target', 'prev_prev_target')
            res = np.concatenate((res,prev_res),1)

            if 'two_stage' in prev_outputs:
                prev_two_stage_res = viz.display_two_stage(prev_outputs,target,i,'prev_target')
                two_stage_res = np.concatenate((two_stage_res,prev_two_stage_res),1)

        cur_res = viz.display_main_pred(outputs['main'],target,i,'cur_target', 'prev_target')
        res = np.concatenate((res,cur_res),1)

        if 'two_stage' in outputs:
            cur_two_stage_res = viz.display_two_stage(outputs,target,i,'cur_target')
            two_stage_res = np.concatenate((two_stage_res,cur_two_stage_res),1)
                
        aux_outputs = outputs['main']['aux_outputs']
        if 'CoMOT' in aux_outputs[-1]:
            CoMOT_res = viz.display_CoMOT(aux_outputs,target,i)

        min_track_id, max_track_id = min_track_ids[i], max_track_ids[i]

        if print_gt:
    
            gt_dict = {}
            target_names = ['prev_prev','prev','cur','prev_flex','cur_flex']
            if 'fut_target' in target:
                target_names += ['fut']

            for t in target_names:

                if '_flex' in t:
                    target_holder = target['main']
                    target_name = t[:-5] + '_target'
                    image_name = t[:-5] + '_image'                                
                else:
                    if args.display_all:
                        target_holder = target['target_og']
                        target_name = t + '_target'
                        image_name = t + '_image'
                    else:
                        continue

                man_track = target_holder['man_track']

                frame_gt = viz.preprocess_img(target,image_name)

                if not target['main'][target_name]['empty']:

                    # If just object detection, then just current frame is used so two frames back gives us no valuable information
                    boxes = target_holder[target_name]['boxes'].cpu().numpy()
                    boxes = viz.bbox_cxcy_to_xyxy(boxes)

                    track_ids = target_holder[target_name]['track_ids'].cpu().numpy()

                    if use_masks:
                        masks = target_holder[target_name]['masks'].cpu().numpy()

                    for idx, box in enumerate(boxes):

                        flex_div = edge = False
                        if 'flexible_divisions' in target_holder[target_name] and target_holder[target_name]['flexible_divisions'][idx]:
                            flex_div = True

                        if 'is_touching_edge' in target_holder[target_name] and target_holder[target_name]['is_touching_edge'][idx]:
                            edge = True
                        
                        color = colors[track_ids[idx]-min_track_id]
                        frame_gt = viz.draw_bbox(frame_gt,box[:4],color,flex_div=flex_div,edge=edge)

                        if use_masks:
                            frame_gt = viz.draw_mask(frame_gt,masks[idx,0],color)

                        if box[-1] > 0:
                            frame_gt = viz.draw_bbox(frame_gt,box[4:],color,flex_div=flex_div,edge=edge)

                            if use_masks:
                                frame_gt = viz.draw_mask(frame_gt,masks[idx,1],color)

                        track_id_ind = man_track[:,0] == track_ids[idx]

                        if man_track[track_id_ind,1] == target['main'][target_name]['framenb'] and  man_track[track_id_ind,-1] > 0:
                            mother_id = man_track[track_id_ind,-1]
                            color = colors[mother_id - min_track_id]
                            centroid = (int(np.clip(box[0] + box[2] / 2,0,width)), int(np.clip(box[1] + box[3] / 2,0,height)))
                            frame_gt = cv2.circle(frame_gt, centroid, radius=2, color=color, thickness=-1)

                gt_dict[t] = frame_gt

            if args.display_all:

                flex_div = ''
                if args.flex_div:
                    flex_div='_flex'

                gts = gt_dict['cur'+flex_div]

                if prev_outputs is not None:
                    gts = np.concatenate((gt_dict['prev'+flex_div],gts),axis=1)

                if prev_prev_outputs is not None:
                    gts = np.concatenate((gt_dict['prev_prev'],gts),axis=1)

                if 'fut_target' in target:
                    gts = np.concatenate((gts,gt_dict['fut']),axis=1)
            else:
                gts = gt_dict['cur_flex']

        raw_images = viz.preprocess_img(target,'cur_image') 

        if args.display_all:

            if prev_outputs is not None:
                prev_img = viz.preprocess_img(target,'prev_image')
                raw_images = np.concatenate((prev_img,raw_images),axis=1)

            if prev_prev_outputs is not None:
                prev_prev_img = viz.preprocess_img(target,'prev_prev_image')
                raw_images = np.concatenate((prev_prev_img,raw_images),axis=1)

            if 'fut_target' in target:
                fut_img = viz.preprocess_img(target,'fut_image')    
                raw_images = np.concatenate((raw_images,fut_img),axis=1)
        
        if i == 0:
            res_batch = np.concatenate((res,np.zeros((height,10,3)),gts,np.zeros((height,10,3)),raw_images),axis=1)
            if 'two_stage' in outputs:
                two_stage_res_batch = two_stage_res.copy()
            if 'CoMOT' in aux_outputs[-1]:
                CoMOT_res_batch = CoMOT_res.copy()

        else:
            res_batch = np.concatenate((res_batch,np.concatenate((res,np.zeros((height,10,3)),gts,np.zeros((height,10,3)),raw_images),axis=1)),axis=0)
            if 'two_stage' in outputs:
                two_stage_res_batch = np.concatenate((two_stage_res_batch,two_stage_res),1)
            if 'CoMOT' in aux_outputs[-1]:
                CoMOT_res_batch = np.concatenate((CoMOT_res_batch,CoMOT_res),1)          

    cv2.imwrite(str(savepath / folder / 'standard' / filename),res_batch)

    if 'two_stage' in outputs:
        cv2.imwrite(str(savepath / folder / 'two_stage' / filename), two_stage_res_batch)

    if 'CoMOT' in aux_outputs[-1]:
        cv2.imwrite(str(savepath / folder / 'CoMOT' / filename), CoMOT_res_batch)

def plot_tracking_results(img,bbs,masks,colors,div_track=None,new_cells=None,use_new_img=True):

    if use_new_img:
        img = np.copy(np.array(img))
    height = img.shape[0]
    width = img.shape[1]
    alpha = 0.4

    if bbs is not None:
        # Convert bboxes and masks into correct format
        bbs = box_cxcy_to_xyxy(bbs.copy(),height,width)

        # Draw bboxes and masks onto image
        for idx,bounding_box in enumerate(bbs):

            # Add bbox to image
            thickness = 2 if new_cells is not None and new_cells[idx] == True else 1
            img = cv2.rectangle(
                img,
                (int(np.clip(bounding_box[0],0,width)), int(np.clip(bounding_box[1],0,height))),
                (int(np.clip(bounding_box[0] + bounding_box[2],0,width)), int(np.clip(bounding_box[1] + bounding_box[3],0,height))),
                color=(colors[idx]),
                thickness = thickness)

    if masks is not None:
        assert masks.dtype == bool
        masks = (masks.copy() * 255).astype(np.uint8)

        for idx, mask in enumerate(masks):
            # Color mask onto image
            if masks is not None:
                mask = np.repeat(masks[idx,:,:,None],3,axis=-1)
                mask[mask[...,0]>0] = colors[idx]
                img[mask>0] = img[mask>0]*(1-alpha) + mask[mask>0]*(alpha)

    # Draw line connecting cell divisions
    if div_track is not None and (div_track != -1).sum() > 0 and bbs is not None:
        div_track_nbs = np.unique(div_track[div_track != -1])

        for div_track_nb in div_track_nbs:
            div_loc = (div_track == div_track_nb).nonzero()[0]
            cell_1 = bbs[div_loc[0]]
            cell_2 = bbs[div_loc[1]]

            img = cv2.arrowedLine(
                img,
                (int(cell_1[0] + cell_1[2] // 2), int(cell_1[1] + cell_1[3] // 2)),
                (int(cell_2[0] + cell_2[2] // 2), int(cell_2[1] + cell_2[3] // 2)),
                color=(1, 1, 1),
                thickness=1,
            )

    return img
 