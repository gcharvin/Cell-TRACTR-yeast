# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Data visualizaiton functions.

"""
import cv2 
import numpy as np
import torch
from pathlib import Path

class data_visualizer():

    def __init__(self,height,width,num_queries,fontscale=0.4,alpha=0.3,mask_threshold=0.5):
        self.height = height
        self.width = width
        self.num_queries = num_queries
        self.fontscale = fontscale
        self.alpha = alpha
        self.mask_threshold = mask_threshold

        self.enc_thresholds = [0, 0.1 ,0.3 ,0.5 , 0.8, 1.0]

    def draw_bbox(self,img,bounding_box,color=None,pred_logit=None,thickness=1):

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
        
        if pred_logit is not None:
            img = cv2.putText(
                img,
                text = f'{pred_logit}', 
                org=(int(bounding_box[0]) - 5, int(bounding_box[1] + bounding_box[3] // 4 + int(self.fontscale*30))), 
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
        mask[mask[...,0]>0] = color
        img[mask>0] = img[mask>0]*(1-self.alpha) + mask[mask>0]*(self.alpha)

        return img

    def filter_pred_masks(self,pred_masks):

        masks_filt = np.zeros((pred_masks.shape))
        argmax = np.argmax(pred_masks,axis=0)
        for m in range(pred_masks.shape[0]):
            masks_filt[m,argmax==m] = pred_masks[m,argmax==m]
        masks_filt = masks_filt > self.mask_threshold
        pred_masks = (masks_filt*255).astype(np.uint8)

        return pred_masks

    def bbox_cxcy_to_xyxy(self,boxes):

        boxes[:,1::2] = boxes[:,1::2] * self.height
        boxes[:,::2] = boxes[:,::2] * self.width

        boxes[:,0] = boxes[:,0] - boxes[:,2] // 2
        boxes[:,1] = boxes[:,1] - boxes[:,3] // 2

        if boxes.shape[1] > 4:
            boxes[:,4] = boxes[:,4] - boxes[:,6] // 2
            boxes[:,5] = boxes[:,5] - boxes[:,7] // 2

        return boxes
    
    def preprocess_img(self,target,image_name):

        img = target[image_name].permute(1,2,0)
        img = img.detach().cpu().numpy().copy()
        img = np.repeat(np.mean(img,axis=-1)[:,:,np.newaxis],3,axis=-1)
        img = (255*(img - np.min(img)) / np.ptp(img)).astype(np.uint8)

        return img

    def get_min_max_track_id(self,target):

        empty = torch.tensor([target[name]['empty'] for name in ['prev_prev_target','prev_target','cur_target','fut_target']]).sum().item()

        if empty < 4:
            all_track_ids = torch.cat([target[name]['track_ids'] for name in ['prev_prev_target','prev_target','cur_target','fut_target'] if target[name]['track_ids'].ndim == 1])
            min_track_id = all_track_ids.min().item()
            max_track_id = all_track_ids.max().item() + 1
        else:
            min_track_id = 0
            max_track_id = self.num_queries + 1

        return min_track_id, max_track_id
    
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

def plot_results(outputs,prev_outputs,targets,samples,savepath,filename,folder,args,meta_data=None):

    print_gt = True
    bs,_,height,width = samples.shape
    # spacer = 10
    cls_threshold = 0.5 
    mask_threshold = 0.5
    fontscale = 0.4
    alpha = 0.3
    use_masks = 'pred_masks' in outputs
    # box_converter = box_cxcy_to_xyxy(height,width)
    batch_enc_frames = []
    batch_CoMOT_frames = []
    colors = np.array([tuple((255*np.random.random(3))) for _ in range(10000)]) # Assume max 1000 cells in one chamber
    colors[:6] = np.array([[0.,0.,255.],[0.,255.,0.],[255.,0.,0.],[255.,0.,255.],[0.,255.,255.],[255.,255.,0.]])
    # colors = [tuple((255*np.random.random(3))) for _ in range(600)]
    # colors[:6] = [np.array(255.,0.,0.),np.array(0.,255.,0.),np.array(0.,0.,255.),np.array(255.,255.,0.),np.array(255.,0.,255.),np.array(0.,255.,255.),]

    viz = data_visualizer(height,width,args.num_queries,fontscale=fontscale,alpha=alpha,mask_threshold=mask_threshold)
    
    if meta_data is not None:
        meta_data_keys = list(meta_data.keys())

        for meta_data_key in meta_data_keys:

            outputs_TM = meta_data[meta_data_key]['outputs']
            targets_TM = [target[meta_data_key] for target in targets]

            if meta_data_key == 'dn_track' or meta_data_key == 'dn_track_group':

                dn_track_frame = np.zeros((height,0,3))

                for i,target in enumerate(targets):

                    min_track_id, max_track_id = viz.get_min_max_track_id(target)

                    prev_track_ids = target['prev_target']['track_ids'].cpu().numpy()
                    dn_track_prev_ind = target['dn_track']['prev_ind'][1].cpu().numpy()
                    
                    prev_track_ids = prev_track_ids[dn_track_prev_ind]

                    img = viz.preprocess_img(target,'cur_image')              
                    prev_img = viz.preprocess_img(target,'prev_image')              
                    
                    unoised_boxes = targets_TM[i]['track_query_boxes_gt'].cpu().numpy()
                    noised_boxes = targets_TM[i]['track_query_boxes'].cpu().numpy()
                    gt_boxes = targets_TM[i]['boxes'].cpu().numpy()
                    gt_masks = targets_TM[i]['masks'].cpu().numpy()

                    unoised_boxes = viz.bbox_cxcy_to_xyxy(unoised_boxes)
                    noised_boxes = viz.bbox_cxcy_to_xyxy(noised_boxes)
                    gt_boxes = viz.bbox_cxcy_to_xyxy(gt_boxes)

                    gt_masks = viz.filter_pred_masks(gt_masks)

                    TP_mask = targets_TM[i]['track_queries_TP_mask']

                    img_unoised_boxes = img.copy()
                    img_noised_boxes = img.copy()
                    previmg_unoised_boxes = prev_img.copy()
                    previmg_noised_boxes = prev_img.copy()

                    for idx,(unoised_box, noised_box) in enumerate(zip(unoised_boxes,noised_boxes)):

                        color=colors[prev_track_ids[idx] - min_track_id] if TP_mask[idx] else (0,0,0)
                        img_unoised_boxes = viz.draw_bbox(img_unoised_boxes,unoised_box[:4],color)
                        previmg_unoised_boxes = viz.draw_bbox(previmg_unoised_boxes,unoised_box[:4],color)
                        
                        img_noised_boxes = viz.draw_bbox(img_noised_boxes,noised_box[:4],color)
                        previmg_noised_boxes = viz.draw_bbox(previmg_noised_boxes,noised_box[:4],color)

                    pred_boxes_all = outputs_TM['pred_boxes'][i].detach().cpu().numpy()
                    pred_logits_all = outputs_TM['pred_logits'][i].detach().sigmoid().cpu().numpy()

                    pred_boxes_all = viz.bbox_cxcy_to_xyxy(pred_boxes_all)

                    num_object_queris = (~targets_TM[i]['track_queries_mask']).sum()

                    keep = pred_logits_all[:,0] > cls_threshold
                    where = np.where(pred_logits_all[:,0] > cls_threshold)[0]
                    pred_logits = pred_logits_all[keep]
                    pred_boxes = pred_boxes_all[keep]

                    img_pred = img.copy()
                    img_pred_all = img.copy()

                    if use_masks:
                        pred_masks_all = outputs_TM['pred_masks'][i].detach().cpu().numpy()
                        pred_masks = pred_masks_all[keep]

                        if keep.sum() > 0:
                            pred_masks = viz.filter_pred_masks(pred_masks)

                        pred_masks_all[keep] = pred_masks

                    for idx,pred_box in enumerate(pred_boxes_all):
                        color=colors[prev_track_ids[idx] - min_track_id] if TP_mask[idx] else (0,0,0)
                        # if idx >= pred_logits_all.shape[0] - num_object_queris:
                        #     color = (128,128,128)

                        img_pred_all = viz.draw_bbox(img_pred_all,pred_box[:4],color)

                        if idx in where:

                            pred_logit = f'{pred_logits_all[idx,0]:.2f}' if args.display_all else None
                            img_pred = viz.draw_bbox(img_pred,pred_box[:4],color,pred_logit)

                            if use_masks:
                                img_pred = viz.draw_mask(img_pred,pred_masks_all[idx,0],color)

                            if pred_logits_all[idx,1] > cls_threshold:

                                pred_logit = f'{pred_logits_all[idx,1]:.2f}' if args.display_all else None
                                img_pred = viz.draw_bbox(img_pred,pred_box[4:],color,pred_logit)

                                if use_masks:
                                    img_pred = viz.draw_mask(img_pred,pred_masks_all[idx,1],color)     

                    img_gt = img.copy()

                    track_ids = targets_TM[i]['track_ids'].cpu().numpy()

                    for idx,gt_box in enumerate(gt_boxes):
                         
                        color = colors[track_ids[idx] - min_track_id]
                        img_gt = viz.draw_bbox(img_gt,gt_box[:4],color)       

                        if use_masks:
                            img_gt = viz.draw_mask(img_gt,gt_masks[idx,0],color)              

                        if gt_box[-1] > 0:
                            img_gt = viz.draw_bbox(img_gt,gt_box[4:],color)  

                            if use_masks:
                                img_gt = viz.draw_mask(img_gt,gt_masks[idx,1],color)  

                    if args.display_all:
                        dn_track_frame = np.concatenate((dn_track_frame,previmg_unoised_boxes,previmg_noised_boxes,img_pred,img,img_pred_all,img_gt),axis=1)
                    else:
                        dn_track_frame = np.concatenate((dn_track_frame,previmg_noised_boxes,img_pred),axis=1)

                    if i < len(targets) - 1:
                        dn_track_frame = np.concatenate((dn_track_frame,np.zeros((img.shape[0],20,3),dtype=img.dtype)),axis=1)

                cv2.imwrite(str(savepath / folder / meta_data_key / filename),dn_track_frame)

            elif meta_data_key == 'dn_enc':
                dn_encs = []
                for i,target in enumerate(targets):

                    min_track_id, max_track_id = viz.get_min_max_track_id(targets[i])
                    
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

            elif meta_data_key == 'dn_object':

                dn_object_frame = np.zeros((height,0,3))

                for i,target_TM in enumerate(targets_TM):

                    min_track_id, max_track_id = viz.get_min_max_track_id(targets[i])
                    
                    noised_boxes = target_TM['noised_boxes'].cpu().numpy()
                    gt_boxes = target_TM['boxes'].cpu().numpy()
                    TP_mask = ~target_TM['track_queries_fal_pos_mask'].cpu().numpy()
                    track_ids = target_TM['track_ids'].cpu().numpy()
                
                    gt_boxes = viz.bbox_cxcy_to_xyxy(gt_boxes)
                    noised_boxes = viz.bbox_cxcy_to_xyxy(noised_boxes)

                    cur_img = viz.preprocess_img(targets[i],'cur_image')

                    cur_img_gt = cur_img.copy()
                    cur_img_noised = cur_img.copy()
                    cur_img_pred_all = cur_img.copy()
                    cur_img_pred = cur_img.copy()

                    for idx,(gt_box,noised_box) in enumerate(zip(gt_boxes,noised_boxes)):

                        color = colors[track_ids[idx] - min_track_id] if TP_mask[idx] else (0,0,0)
                        cur_img_gt = viz.draw_bbox(cur_img_gt,gt_box[:4],color)
                        cur_img_noised = viz.draw_bbox(cur_img_noised,noised_box[:4],color)

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

                    for idx,pred_box in enumerate(pred_boxes_all):
                        
                        color = colors[track_ids[idx] - min_track_id] if TP_mask[idx] else (0,0,0)
                        cur_img_pred_all = viz.draw_bbox(cur_img_pred_all,pred_box[:4],color)

                        if idx in where:
                            pred_logit = f'{pred_logits_all[idx,0]:.2f}' if args.display_all else None
                            cur_img_pred = viz.draw_bbox(cur_img_pred,pred_box[:4],color,pred_logit)

                            if use_masks:
                                cur_img_pred = viz.draw_mask(cur_img_pred,pred_masks_all[idx,0].squeeze(),color)

                    dn_object_frame = np.concatenate((dn_object_frame,cur_img_noised,cur_img_pred,cur_img),axis=1)
                    
                    if args.display_all:
                        dn_object_frame = np.concatenate((dn_object_frame,cur_img_pred_all),axis=1)

                    if i < len(targets_TM) - 1:
                        dn_object_frame = np.concatenate((dn_object_frame,np.zeros((height,10,3))),axis=1)

                cv2.imwrite(str(savepath / folder / 'dn_object' / filename),dn_object_frame)


    for i,target in enumerate(targets):

        min_track_id, max_track_id = viz.get_min_max_track_id(target)    

        prev_prev_img = viz.preprocess_img(target,'prev_prev_image')    
        prev_img = viz.preprocess_img(target,'prev_image')    
        cur_img = viz.preprocess_img(target,'cur_image')    
        fut_img = viz.preprocess_img(target,'fut_image')    

        previmg_pred_raw = prev_img.copy()
        previmg_pred_match = prev_img.copy()

        img_pred = cur_img.copy()
        img_masks = cur_img.copy()

        if not args.display_all:
            img_pred = img_masks
        
        if prev_outputs is not None:
            prev_pred_boxes_all = prev_outputs['pred_boxes'][i].detach().cpu().numpy()
            prev_pred_logits_all = prev_outputs['pred_logits'][i].sigmoid().detach().cpu().numpy()
            prev_indices = target['cur_target']['prev_ind_orig']
            prev_track_ids = target['prev_target']['track_ids'][prev_indices[1]].cpu().numpy()

            prev_pred_boxes_all = viz.bbox_cxcy_to_xyxy(prev_pred_boxes_all)

            prev_keep = prev_pred_logits_all[:,0] > cls_threshold
            prev_pred_boxes = prev_pred_boxes_all[prev_keep]
            prev_pred_logits = prev_pred_logits_all[prev_keep]

            prev_keep_ind = prev_keep.nonzero()[0]                

            if use_masks:
                prev_pred_masks_all = prev_outputs['pred_masks'][i].sigmoid().detach().cpu().numpy()
                prev_pred_masks = prev_pred_masks_all[prev_keep]

                if prev_keep.sum() > 0:
                    prev_pred_masks = viz.filter_pred_masks(prev_pred_masks)

            for idx,prev_pred_bbox in enumerate(prev_pred_boxes):
                # Check if prev_prev frame was used; 
                if prev_pred_boxes_all.shape[0] > args.num_queries:
                    if prev_keep_ind[idx] < prev_pred_boxes_all.shape[0] - args.num_queries:
                        color = (255,0,0)
                        pred_logit = f'{prev_pred_logits[idx,0]:.2f}' if args.display_all else None
                        previmg_pred_raw = viz.draw_bbox(previmg_pred_raw,prev_pred_bbox[:4],color,pred_logit)

                        if use_masks:
                            previmg_pred_raw = viz.draw_mask(previmg_pred_raw,prev_pred_masks[idx,0],color)

                        if prev_pred_logits[idx,1] > cls_threshold:
                            color = (0,0,255)
                            pred_logit = f'{prev_pred_logits[idx,1]:.2f}' if args.display_all else None
                            previmg_pred_raw = viz.draw_bbox(previmg_pred_raw,prev_pred_bbox[4:],color,pred_logit)

                            if use_masks:
                                previmg_pred_raw = viz.draw_mask(previmg_pred_raw,prev_pred_masks[idx,1],color)

                else: # Prev frame was the first frame; only object detection performed
                    if idx in prev_indices[0]:
                        if len(prev_track_ids) == 1:
                            color = colors[prev_track_ids[0] - min_track_id]
                        elif len(prev_indices[0]) == 1:
                            color = colors[prev_track_ids - min_track_id]
                        else:
                            color = colors[prev_track_ids[prev_indices[0] == idx][0] - min_track_id]
                    else:
                        color = colors[-idx]
                    pred_logit = f'{prev_pred_logits[idx,0]:.2f}' if args.display_all else None
                    previmg_pred_raw = viz.draw_bbox(previmg_pred_raw,prev_pred_bbox[:4],color,pred_logit)

                    if use_masks:
                        previmg_pred_raw = viz.draw_mask(previmg_pred_raw,prev_pred_masks[idx,0],color)

            if len(prev_indices[0]) > 0:
                prev_pred_match_bboxes = prev_pred_boxes_all[prev_indices[0]]
                prev_pred_match_bboxes = prev_pred_match_bboxes[None] if prev_pred_match_bboxes.ndim == 1 else prev_pred_match_bboxes

                if use_masks:
                    prev_pred_match_masks = prev_pred_masks_all[prev_indices[0]]
                    prev_pred_match_masks = prev_pred_match_masks[None] if prev_pred_match_masks.ndim == 1 else prev_pred_match_masks

                for prev_ind_out in prev_indices[0]:
                    if (prev_indices[0] == prev_ind_out).sum() == 2:
                        ind = np.where(prev_indices[0] == prev_ind_out)[0][1]
                        prev_pred_match_bboxes[ind,:4] = prev_pred_match_bboxes[ind,4:]

                        if use_masks:
                            prev_pred_match_masks[ind,:1] = prev_pred_match_masks[ind,1:]

                if use_masks:
                    prev_pred_match_masks = viz.filter_pred_masks(prev_pred_match_masks)

                for idx,prev_pred_match_bbox in enumerate(prev_pred_match_bboxes):
                    color = colors[prev_track_ids[idx] - min_track_id]
                    previmg_pred_match = viz.draw_bbox(previmg_pred_match,prev_pred_match_bbox[:4],color)
                    
                    if use_masks:
                        previmg_pred_match = viz.draw_mask(previmg_pred_match,prev_pred_match_masks[idx,0],color)                     

            if args.display_all:
                pred = np.concatenate((previmg_pred_raw,previmg_pred_match),axis=1)
            else:
                pred = previmg_pred_match

            if 'track_query_boxes' in target['cur_target']:
                prev_indices_FN = target['cur_target']['prev_ind']
                track_query_boxes = target['cur_target']['track_query_boxes'].cpu().numpy()
                track_query_boxes = viz.bbox_cxcy_to_xyxy(track_query_boxes)

                previmg_keep_queries = prev_img.copy()

                track_query_ids = target['prev_target']['track_ids'][prev_indices_FN[1]].cpu().numpy()

                if len(track_query_ids) > 0:
                    track_query_ids = np.concatenate((track_query_ids,np.array([max_track_id+idx for idx in range(track_query_boxes.shape[0]-len(track_query_ids))]).astype(int)))
                else:
                    track_query_ids = np.array([max_track_id+idx for idx in range(track_query_boxes.shape[0]-len(track_query_ids))])

                for idx,track_query_box in enumerate(track_query_boxes):

                    color = colors[track_query_ids[idx] - min_track_id] if idx < len(track_query_ids) else colors[-idx]
                    previmg_keep_queries = viz.draw_bbox(previmg_keep_queries,track_query_box,color)

                pred = np.concatenate((pred,previmg_keep_queries),axis=1)

        if prev_outputs is not None and 'enc_outputs' in prev_outputs:

            previmg_gt = prev_img.copy()
            boxes_gt = target['prev_target']['boxes'][prev_indices[1]].cpu().numpy()
            boxes_gt = viz.bbox_cxcy_to_xyxy(boxes_gt)
            track_ids_gt = target['prev_target']['track_ids'][prev_indices[1]].cpu().numpy()

            for idx,box_gt in enumerate(boxes_gt):
                color = colors[track_ids_gt[idx] - min_track_id]
                previmg_gt = viz.draw_bbox(previmg_gt,box_gt,color)

            enc_outputs = prev_outputs['enc_outputs']
            enc_pred_logits = enc_outputs['pred_logits'][i,:,0].sigmoid().cpu().numpy()
            enc_pred_boxes = enc_outputs['pred_boxes'][i,:,:4].cpu().numpy()

            enc_pred_boxes = viz.bbox_cxcy_to_xyxy(enc_pred_boxes)

            boxes_list, ind_list = viz.stack_enc_preds(enc_pred_boxes,enc_pred_logits,return_indices=True)

            if use_masks:
                enc_pred_masks = enc_outputs['pred_masks'][i,:,0].detach().cpu().sigmoid().numpy()
                enc_pred_masks = ((enc_pred_masks > mask_threshold)*255).astype(np.uint8)

                masks_list = viz.stack_enc_preds(enc_pred_masks,enc_pred_logits)

            if 'mask_enc_boxes' in enc_outputs:
                pred_mask_enc_boxes = enc_outputs['mask_enc_boxes'][i].cpu().numpy()
                pred_mask_enc_boxes = viz.bbox_cxcy_to_xyxy(mask_enc_boxes)

                mask_enc_boxes_list = viz.stack_enc_preds(pred_mask_enc_boxes,enc_pred_logits)

                mask_enc_boxes_frames = []

            if not args.display_all:
                boxes_list = boxes_list[-1:]
                ind_list = ind_list[-1:]

                if use_masks:
                    masks_list = masks_list[-1:]

                if 'mask_enc_boxes' in enc_outputs:
                    mask_enc_boxes_list = mask_enc_boxes_list[-1]

            enc_frames = []
            for b,boxes in enumerate(boxes_list):
                enc_frame = prev_img.copy()
                for idx,box in enumerate(boxes):

                    if ind_list[b][idx] < len(boxes_gt) and ind_list[b][idx] in prev_indices[0] and prev_outputs['pred_logits'].shape[1] == args.num_queries:
                        if len(track_ids_gt) == 1:
                            color = colors[track_ids_gt[0] - min_track_id]
                        elif len(prev_indices[0]) == 1:
                            color = colors[track_ids_gt - min_track_id]
                        else:
                            color = colors[track_ids_gt[prev_indices[0] == ind_list[b][idx]][0] - min_track_id]
                        thickness = 2
                    else:
                        color = colors[-idx]
                        thickness = 1

                    enc_frame = viz.draw_bbox(enc_frame,box,color,thickness=thickness)

                    if use_masks:
                        mask = masks_list[b][idx]
                        enc_frame = viz.draw_mask(enc_frame,mask,color)

                enc_frames.append(enc_frame)

                if 'mask_enc_boxes' in enc_outputs:
                    previmg_mask_enc_boxes = prev_img.copy()
                    previmg_mask_enc_boxes = viz.draw_mask(previmg_mask_enc_boxes,mask_enc_boxes_list[b][idx,0],color)
                    mask_enc_boxes_frames.append(previmg_mask_enc_boxes)

            enc_frames.append(np.zeros((enc_frame.shape[0],3,enc_frame.shape[2])))
            enc_frames.append(previmg_gt)
            enc_frames.append(np.zeros((enc_frame.shape[0],10,enc_frame.shape[2])))
            enc_frames_prev = np.concatenate((enc_frames),axis=1)

            if 'mask_enc_boxes' in enc_outputs:
                mask_enc_boxes_frames.append(np.zeros((enc_frame.shape[0],3,enc_frame.shape[2])))
                mask_enc_boxes_frames.append(previmg_gt)
                mask_enc_boxes_frames.append(np.zeros((enc_frame.shape[0],10,enc_frame.shape[2])))
                mask_enc_boxes_frames = np.concatenate((mask_enc_boxes_frames),axis=1)
                enc_frames_prev = np.concatenate((enc_frames_prev,mask_enc_boxes_frames),axis=0)

        pred_boxes_all = outputs['pred_boxes'][i].detach().cpu().numpy()
        pred_logits_all = outputs['pred_logits'][i].sigmoid().detach().cpu().numpy()

        pred_boxes_all = viz.bbox_cxcy_to_xyxy(pred_boxes_all)

        keep = pred_logits_all[:,0] > cls_threshold
        pred_boxes = pred_boxes_all[keep]
        pred_logits = pred_logits_all[keep]

        if use_masks:
            pred_masks_all = outputs['pred_masks'][i].detach().cpu().sigmoid().numpy()
            pred_masks = pred_masks_all[keep]

            if sum(keep) > 0:
                pred_masks = viz.filter_pred_masks(pred_masks)

        if prev_outputs is not None and 'track_query_boxes' in target['cur_target']:
            colors_track = [colors[prev_track_ids[idx] - min_track_id] for idx in range(track_query_boxes.shape[0] - target['cur_target']['num_FPs']) if keep[idx]] 
        else:
            indices = target['cur_target']['indices']
            track_ids = target['cur_target']['track_ids'][indices[1]]
            colors_track = []

            for k in np.where(keep)[0]:
                if k in indices[0]:
                    track_id = track_ids[indices[0] == k]
                    colors_track.append(colors[track_id-min_track_id])
                else:
                    colors_track.append(colors[-k])

        for idx, pred_box in enumerate(pred_boxes):

            color = colors_track[idx] if idx < len(colors_track) else (0,0,0)

            if use_masks:
                img_masks = viz.draw_mask(img_masks,pred_masks[idx,0],color)

            pred_logit = f'{pred_logits[idx,0]:.2f}' if args.display_all else None
            img_pred = viz.draw_bbox(img_pred,pred_box[:4],color,pred_logit)

            if pred_logits[idx,1] > cls_threshold:

                if use_masks:
                    img_masks = viz.draw_mask(img_masks,pred_masks[idx,1],color)

                pred_logit = f'{pred_logits[idx,1]:.2f}' if args.display_all else None
                img_pred = viz.draw_bbox(img_pred,pred_box[4:],color,pred_logit)
                    
        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            enc_pred_logits_topk = enc_outputs['pred_logits'][i,:,0].detach().cpu().sigmoid().numpy()
            enc_pred_boxes_topk = enc_outputs['pred_boxes'][i,:,:4].detach().cpu().numpy()

            assert enc_pred_logits_topk.shape[0] == args.num_queries and enc_pred_boxes_topk.shape[0] == args.num_queries, 'All enc preds were saved instead of just the topk'

            cells_in_frame = target['enc_outputs']['boxes'].shape[1] > 0

            if cells_in_frame:
                enc_indices = enc_outputs['indices'][i]
                track_ids = target['enc_outputs']['track_ids'][enc_indices[1]]

            enc_pred_boxes_topk = viz.bbox_cxcy_to_xyxy(enc_pred_boxes_topk)

            boxes_list, ind_list = viz.stack_enc_preds(enc_pred_boxes_topk,enc_pred_logits_topk, return_indices=True)

            if use_masks:
                enc_pred_masks_topk = enc_outputs['pred_masks'][i,:,0].detach().cpu().sigmoid().numpy()
                            
                # We don't filter masks here; we show all pred masks; they can overlap
                enc_pred_masks_topk = ((enc_pred_masks_topk > mask_threshold)*255).astype(np.uint8)

                masks_list = viz.stack_enc_preds(enc_pred_masks_topk,enc_pred_logits_topk)

            enc_frames = []

            if 'mask_enc_boxes' in enc_outputs:
                mask_enc_boxes = enc_outputs['mask_enc_boxes'][i].detach().cpu().numpy()
                mask_enc_boxes = viz.bbox_cxcy_to_xyxy(mask_enc_boxes)

                mask_enc_boxes_list = viz.stack_enc_preds(mask_enc_boxes, enc_pred_logits_topk)

                mask_enc_boxes_frames = [] 

            if not args.display_all:
                boxes_list = boxes_list[-1:]
                ind_list = ind_list[-1:]

                if use_masks:
                    masks_list = masks_list[-1:]

                if 'mask_enc_boxes' in enc_outputs:
                        mask_enc_boxes_list = mask_enc_boxes_list[-1:]

            for b,boxes in enumerate(boxes_list):
                enc_frame = np.array(cur_img).copy()

                for idx,bounding_box in enumerate(boxes):

                    if cells_in_frame and ind_list[b][idx] in enc_indices[0]:
                        thickness = 2
                        track_id = track_ids[enc_indices[0] == ind_list[b][idx]]
                    else:
                        thickness = 1
                        track_id = None

                    color=colors[-idx] if track_id is None else colors[track_id - min_track_id]

                    enc_frame = viz.draw_bbox(enc_frame,bounding_box[:4],color,thickness=thickness)

                    if use_masks:

                        mask = masks_list[b][idx]
                        enc_frame = viz.draw_mask(enc_frame,mask,color)

                enc_frames.append(enc_frame)

                if 'mask_enc_boxes' in enc_outputs:
                    mask_enc_boxes = mask_enc_boxes_list[b]
                    img_mask_enc_boxes = np.array(cur_img).copy()

                    for idx,bounding_box in enumerate(mask_enc_boxes):

                        if cells_in_frame and ind_list[b][idx] in enc_indices[0]:
                            thickness = 2
                            track_id = track_ids[enc_indices[0] == ind_list[b][idx]]
                        else:
                            thickness = 1
                            track_id = None

                        img_mask_enc_boxes = viz.draw_bbox(img_mask_enc_boxes,bounding_box[:4],color,thickness=thickness)            

                    mask_enc_boxes_frames.append(img_mask_enc_boxes)

            if cells_in_frame:
                img_gt = cur_img.copy()
                boxes_gt = target['enc_outputs']['boxes'].cpu().numpy()
                track_ids = target['enc_outputs']['track_ids'].cpu().numpy()
                boxes_gt = viz.bbox_cxcy_to_xyxy(boxes_gt)

                for idx,bounding_box in enumerate(boxes_gt):
                    color=colors[track_ids[idx]-min_track_id]
                    img_gt = viz.draw_bbox(img_gt,bounding_box[:4],color)

                enc_frames.append(np.zeros((enc_frame.shape[0],3,enc_frame.shape[2])))
                enc_frames.append(img_gt)

                if 'mask_enc_boxes' in enc_outputs:
                    mask_enc_boxes_frames.append(np.zeros((enc_frame.shape[0],3,enc_frame.shape[2])))
                    mask_enc_boxes_frames.append(img_gt)

            if i == 0:
                enc_frames.append(np.zeros((enc_frame.shape[0],10,enc_frame.shape[2])))
                if 'mask_enc_boxes' in enc_outputs:
                    mask_enc_boxes_frames.append(np.zeros((enc_frame.shape[0],10,enc_frame.shape[2])))

            enc_frames_cur = np.concatenate((enc_frames),axis=1)
            if 'mask_enc_boxes' in enc_outputs:
                mask_enc_boxes_frames = np.concatenate((mask_enc_boxes_frames),axis=1)
                enc_frames_cur = np.concatenate((enc_frames_cur,mask_enc_boxes_frames),axis=0)

            if prev_outputs is not None:
                enc_frames = np.concatenate((enc_frames_prev,enc_frames_cur),axis=1)
            else:
                enc_frames = enc_frames_cur

            batch_enc_frames.append(enc_frames)

        if prev_outputs is not None: 
            if args.display_all:
                pred = np.concatenate((pred,img_pred,img_masks),axis=1)
            else:
                pred = np.concatenate((pred,img_pred),axis=1)
        else:
            if args.display_all:
                pred = np.concatenate((img_pred,img_masks),axis=1)
            else:
                pred = img_pred

        if 'CoMOT' in outputs['aux_outputs'][-1] and outputs['aux_outputs'][-1]['CoMOT']:
            pred_logits_all = outputs['aux_outputs'][-1]['pred_logits'].detach().cpu().sigmoid().numpy()[i,-args.num_queries:]
            pred_boxes_all = outputs['aux_outputs'][-1]['pred_boxes'].detach().cpu().numpy()[i,-args.num_queries:]
            CoMOT_indices = outputs['aux_outputs'][-1]['CoMOT_indices'][i]
            CoMOT_track_ids = target['CoMOT']['track_ids'][CoMOT_indices[1]].cpu().numpy()

            keep = pred_logits_all[:,0] > cls_threshold
            keep_ind = np.where(keep)[0]

            if 'pred_masks' in outputs['aux_outputs'][-1]:
                pred_masks_all = outputs['aux_outputs'][-1]['pred_masks'].detach().cpu().numpy()[i,-args.num_queries:]
                pred_masks = pred_masks_all[keep]

                if keep.sum() > 0:
                    pred_masks = viz.filter_pred_masks(pred_masks)

                pred_masks_all[keep] = pred_masks

            pred_boxes_all = viz.bbox_cxcy_to_xyxy(pred_boxes_all)

            img_CoMOT_raw = cur_img.copy()
            img_CoMOT_match = cur_img.copy()

            for idx, pred_box in enumerate(pred_boxes_all):

                if idx in CoMOT_indices[0]:
                    if len(CoMOT_track_ids) == 1:
                        color = colors[CoMOT_track_ids[0] - min_track_id]
                    elif len(CoMOT_indices[0]) == 1:
                        color = colors[CoMOT_track_ids - min_track_id]
                    else:
                        color = colors[CoMOT_track_ids[CoMOT_indices[0] == idx][0] - min_track_id]
                else:
                    color = colors[-idx]

                pred_logit = f'{pred_logits_all[idx,0]:.2f}' if args.display_all else f'{pred_logits_all[idx,0]:.1f}'
                
                if idx in keep_ind:
                    img_CoMOT_raw = viz.draw_bbox(img_CoMOT_raw,pred_box[:4],color,pred_logit)

                    if use_masks:
                        img_CoMOT_raw = viz.draw_mask(img_CoMOT_raw,pred_masks_all[idx,0],color)

                if idx in CoMOT_indices[0]:
                    img_CoMOT_match = viz.draw_bbox(img_CoMOT_match,pred_box[:4],color,pred_logit)

                    if use_masks:
                        img_CoMOT_match = viz.draw_mask(img_CoMOT_match,pred_masks_all[idx,0],color)        
                
            batch_CoMOT_frames.append(img_CoMOT_match)
            
            if args.display_all:
                batch_CoMOT_frames.append(img_CoMOT_raw)


        if print_gt:
    
            gt_dict = {}

            for t in ['prev_prev','prev','cur','fut','prev_flex','cur_flex']:

                if '_flex' in t:
                    target_holder = target
                    target_name = t[:-5] + '_target'
                    image_name = t[:-5] + '_image'                                
                else:
                    target_holder = target['target_og']
                    target_name = t + '_target'
                    image_name = t + '_image'

                man_track = target_holder['man_track']

                frame_gt = viz.preprocess_img(target,image_name)

                if not target[target_name]['empty']:

                    # If just object detection, then just current frame is used so two frames back gives us no valuable information
                    boxes = target_holder[target_name]['boxes'].cpu().numpy()
                    boxes = viz.bbox_cxcy_to_xyxy(boxes)

                    track_ids = target_holder[target_name]['track_ids'].cpu().numpy()

                    if use_masks:
                        masks = target_holder[target_name]['masks'].cpu().numpy()

                    for idx, box in enumerate(boxes):

                        color = colors[track_ids[idx]-min_track_id]
                        frame_gt = viz.draw_bbox(frame_gt,box[:4],color)

                        if use_masks:
                            frame_gt = viz.draw_mask(frame_gt,masks[idx,0],color)

                        if box[-1] > 0:
                            frame_gt = viz.draw_bbox(frame_gt,box[4:],color)

                            if use_masks:
                                frame_gt = viz.draw_mask(frame_gt,masks[idx,1],color)

                        track_id_ind = man_track[:,0] == track_ids[idx]

                        if man_track[track_id_ind,1] == target[target_name]['framenb']:
                            mother_id = man_track[track_id_ind,-1]
                            color = colors[mother_id - min_track_id]
                            centroid = (int(np.clip(box[0] + box[2] / 2,0,width)), int(np.clip(box[1] + box[3] / 2,0,height)))
                            frame_gt = cv2.circle(frame_gt, centroid, radius=2, color=color, thickness=-1)

                gt_dict[t] = frame_gt

            if args.display_all:
                if prev_outputs is not None:
                    gts = np.concatenate((gt_dict['prev_flex'],gt_dict['cur_flex'],np.zeros((cur_img.shape[0],10,3)),gt_dict['prev_prev'],gt_dict['prev'],gt_dict['cur'],gt_dict['fut']),axis=1)
                else:
                    gts = np.concatenate((gt_dict['cur_flex'],np.zeros((cur_img.shape[0],10,3)),gt_dict['prev'],gt_dict['cur'],gt_dict['fut']),axis=1)
            else:
                gts = gt_dict['cur_flex']

        if args.display_all:
            raw_images = np.concatenate((prev_prev_img,prev_img,cur_img,fut_img),axis=1)
        else:
            raw_images = cur_img
        
        if i == 0:
            res = np.concatenate((pred,np.zeros((cur_img.shape[0],10,3)),gts,np.zeros((cur_img.shape[0],10,3)),raw_images),axis=1)
        else:
            res = np.concatenate((res,np.concatenate((pred,np.zeros((cur_img.shape[0],10,3)),gts,np.zeros((cur_img.shape[0],10,3)),raw_images),axis=1)),axis=0)

    cv2.imwrite(str(savepath / folder / 'standard' / filename),res)

    if 'enc_outputs' in outputs:
        cv2.imwrite(str(savepath / folder / 'enc_outputs' / filename),np.concatenate((batch_enc_frames),axis=1))

    if 'CoMOT' in outputs['aux_outputs'][-1]:
        cv2.imwrite(str(savepath / folder / 'CoMOT' / filename),np.concatenate((batch_CoMOT_frames),axis=1))

def plot_tracking_results(img,bbs,masks,colors,div_track=None,new_cells=None):

    img = np.copy(np.array(img))
    height = img.shape[0]
    width = img.shape[1]
    mask_threshold = 0.5
    alpha = 0.4
    box_converter = box_cxcy_to_xyxy(height,width)

    if bbs is not None:
        bbs = bbs.detach().cpu().numpy()

        if masks is not None:
            masks = masks.detach().cpu().numpy()

            masks_filt = np.zeros((masks.shape))
            argmax = np.argmax(masks,axis=0)
            for m in range(masks.shape[0]):
                masks_filt[m,argmax==m] = masks[m,argmax==m]
            masks_filt = masks_filt > mask_threshold
            masks = (masks_filt*255).astype(np.uint8)

        if bbs is not None:
            bbs = box_converter.convert(bbs)

        for idx,bounding_box in enumerate(bbs):

            thickness = 2 if new_cells is not None and new_cells[idx] == True else 1
            img = cv2.rectangle(
                img,
                (int(np.clip(bounding_box[0],0,width)), int(np.clip(bounding_box[1],0,height))),
                (int(np.clip(bounding_box[0] + bounding_box[2],0,width)), int(np.clip(bounding_box[1] + bounding_box[3],0,height))),
                color=(colors[idx]),
                thickness = thickness)

            if masks is not None:
                mask = cv2.resize(masks[idx],(width,height)) 
                mask = np.repeat(mask[:,:,None],3,axis=-1)
                mask[mask[...,0]>0] = colors[idx]
                img[mask>0] = img[mask>0]*(1-alpha) + mask[mask>0]*(alpha)

        # if track ; currently want to see how much object uses divisions
        if div_track is not None and (div_track != -1).sum() > 0:
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
