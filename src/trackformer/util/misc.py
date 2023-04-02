# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import datetime
import os
import pickle
import subprocess
import time
from argparse import Namespace
from collections import defaultdict, deque
from tkinter import W
from typing import List, Optional
import re
import cv2 
import numpy as np
import math
import torch
import torch.distributed as dist
import torch.nn.functional as F
# needed due to empty tensor bug in pytorch and torchvision 0.5
import torchvision
from torch import Tensor, nn
from pathlib import Path
# from torchmetrics.classification import BinaryAveragePrecision

from . import box_ops

if int(re.findall('\d+',(torchvision.__version__[:4]))[-1]) < 7:
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size

class box_cxcy_to_xyxy():

    def __init__(self,height,width):
        self.height = height
        self.width = width

    def convert(self,boxes):

        boxes[:,1::2] = boxes[:,1::2] * self.height
        boxes[:,::2] = boxes[:,::2] * self.width

        boxes[:,0] = boxes[:,0] - boxes[:,2] // 2
        boxes[:,1] = boxes[:,1] - boxes[:,3] // 2

        if boxes.shape[1] > 4:
            boxes[:,4] = boxes[:,4] - boxes[:,6] // 2
            boxes[:,5] = boxes[:,5] - boxes[:,7] // 2

        return boxes

def plot_results(outputs,prev_outputs,targets,samples,savepath,filename,folder,args,meta_data=None):

    targets_og = [target['target_og'] for target in targets]
    cur_targets = [target['cur_target'] for target in targets]
    print_gt = True
    filename = Path(filename)
    height = samples.shape[-2]
    width = samples.shape[-1]
    bs = samples.shape[0]
    spacer = 10
    threshold = 0.5 
    mask_threshold = 0.5
    fontscale = 0.4
    alpha = 0.3
    use_masks = 'pred_masks' in outputs
    box_converter = box_cxcy_to_xyxy(height,width)
    batch_enc_frames = []
    colors = [tuple((255*np.random.random(3))) for _ in range(600)]
    
    if meta_data is not None:
        meta_data_keys = list(meta_data.keys())

        for meta_data_key in meta_data_keys:

            outputs_TM = meta_data[meta_data_key]['outputs']
            targets_TM = [target[meta_data_key] for target in targets]

            if meta_data_key == 'dn_track':
                # dn_track = np.zeros((height,(width*7 + spacer)*bs,3))
                for i in range(bs):

                    img = samples[i].permute(1,2,0)                
                    
                    boxes = targets_TM[i]['track_query_boxes_gt'].detach().cpu().numpy()
                    noised_boxoes = targets_TM[i]['track_query_boxes'].detach().cpu().numpy()
                
                    boxes = box_converter.convert(boxes)
                    noised_boxes = box_converter.convert(noised_boxoes)

                    img = img.detach().cpu().numpy().copy()
                    img = np.repeat(np.mean(img,axis=-1)[:,:,np.newaxis],3,axis=-1)
                    img = (255*(img - np.min(img)) / np.ptp(img)).astype(np.uint8)
                    img_gt = img.copy()
                    img_noised_gt = img.copy()
                    img_pred_track = img.copy()
                    img_pred_object = img.copy()
                    img_all_pred = img.copy()

                    TP_mask = targets_TM[i]['track_queries_TP_mask']

                    previmg = targets[i]['prev_image'].permute(1,2,0).detach().cpu().numpy()
                    previmg = np.repeat(np.mean(previmg,axis=-1)[:,:,np.newaxis],3,axis=-1)
                    previmg = (255*(previmg - np.min(previmg)) / np.ptp(previmg)).astype(np.uint8)
                    previmg_noised = previmg.copy()

                    for idx,bbox in enumerate(boxes):
                        bounding_box = bbox[:4]
                        img_gt = cv2.rectangle(
                            img_gt,
                            (int(np.clip(bounding_box[0],0,width)), int(np.clip(bounding_box[1],0,height))),
                            (int(np.clip(bounding_box[0] + bounding_box[2],0,width)), int(np.clip(bounding_box[1] + bounding_box[3],0,height))),
                            color=colors[idx] if TP_mask[idx] else (0,0,0),
                            thickness = 1)

                        previmg = cv2.rectangle(
                            previmg,
                            (int(np.clip(bounding_box[0],0,width)), int(np.clip(bounding_box[1],0,height))),
                            (int(np.clip(bounding_box[0] + bounding_box[2],0,width)), int(np.clip(bounding_box[1] + bounding_box[3],0,height))),
                            color=colors[idx] if TP_mask[idx] else (0,0,0),
                            thickness = 1)

                    for idx,bbox in enumerate(noised_boxes):
                        bounding_box = bbox[:4]
                        img_noised_gt = cv2.rectangle(
                            img_noised_gt,
                            (int(np.clip(bounding_box[0],0,width)), int(np.clip(bounding_box[1],0,height))),
                            (int(np.clip(bounding_box[0] + bounding_box[2],0,width)), int(np.clip(bounding_box[1] + bounding_box[3],0,height))),
                            color=colors[idx] if TP_mask[idx] else (0,0,0),
                            thickness = 1)

                        previmg_noised = cv2.rectangle(
                            previmg_noised,
                            (int(np.clip(bounding_box[0],0,width)), int(np.clip(bounding_box[1],0,height))),
                            (int(np.clip(bounding_box[0] + bounding_box[2],0,width)), int(np.clip(bounding_box[1] + bounding_box[3],0,height))),
                            color=colors[idx] if TP_mask[idx] else (0,0,0),
                            thickness = 1)


                    pred_boxes = outputs_TM['pred_boxes'][i].detach().cpu().numpy()
                    pred_logits = outputs_TM['pred_logits'][i].sigmoid().detach().cpu().numpy()

                    pred_boxes = box_converter.convert(pred_boxes)

                    for idx,bbox in enumerate(pred_boxes):
                        if idx < pred_logits.shape[0] - (~targets_TM[i]['track_queries_mask']).sum() and pred_logits[idx,0] < threshold:
                            bounding_box = bbox[:4]
                            img_all_pred = cv2.rectangle(
                                img_all_pred,
                                (int(np.clip(bounding_box[0],0,width)), int(np.clip(bounding_box[1],0,height))),
                                (int(np.clip(bounding_box[0] + bounding_box[2],0,width)), int(np.clip(bounding_box[1] + bounding_box[3],0,height))),
                                color=colors[idx] if TP_mask[idx] else (0,0,0),
                                thickness = 1)  

                            img_all_pred = cv2.putText(
                                img_all_pred,
                                text = f'{pred_logits[idx,0]:.2f}', 
                                org=(int(bounding_box[0]) - 5, int(bounding_box[1] + bounding_box[3] // 4 + int(fontscale*30))), 
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                                fontScale = fontscale,
                                color = colors[idx] if TP_mask[idx] else (128,128,128),
                                thickness=1,
                            )

                    keep = pred_logits[:,0] > threshold
                    where = np.where(pred_logits[:,0] > threshold)[0]
                    pred_logits = pred_logits[keep]
                    pred_boxes = pred_boxes[keep]

                    if use_masks:
                        pred_masks = outputs_TM['pred_masks'][i].sigmoid().detach().cpu().numpy()[keep]

                        if sum(keep) > 0:
                            masks_filt = np.zeros((pred_masks.shape))
                            argmax = np.argmax(pred_masks,axis=0)
                            for m in range(pred_masks.shape[0]):
                                masks_filt[m,argmax==m] = pred_masks[m,argmax==m]
                            masks_filt = masks_filt > mask_threshold
                            pred_masks = (masks_filt*255).astype(np.uint8)

                    for idx,bbox in enumerate(pred_boxes): 
                        if where[idx] < outputs_TM['pred_logits'][i].shape[0] - (~targets_TM[i]['track_queries_mask']).sum():
                            bounding_box = bbox[:4]
                            img_pred_track = cv2.rectangle(
                                img_pred_track,
                                (int(np.clip(bounding_box[0],0,width)), int(np.clip(bounding_box[1],0,height))),
                                (int(np.clip(bounding_box[0] + bounding_box[2],0,width)), int(np.clip(bounding_box[1] + bounding_box[3],0,height))),
                                color=colors[where[idx]] if TP_mask[where[idx]] else (0,0,0),
                                thickness = 1)            

                            if args.display_all:
                                img_pred_track = cv2.putText(
                                    img_pred_track,
                                    text = f'{pred_logits[idx,0]:.2f}', 
                                    org=(int(bounding_box[0]) - 5, int(bounding_box[1] + bounding_box[3] // 4 + int(fontscale*30))), 
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                                    fontScale = fontscale,
                                    color = colors[where[idx]] if TP_mask[where[idx]] else (128,128,128),
                                    thickness=1,
                                )

                            if use_masks:
                                mask = cv2.resize(pred_masks[idx,0],(width,height))
                                mask = np.repeat(mask[...,None],3,axis=-1)
                                mask[mask[...,0]>0] = colors[where[idx]] if idx < len(colors) else (128,128,128)
                                img_pred_track[mask>0] = img_pred_track[mask>0]*(1-alpha) + mask[mask>0]*(alpha)

                            if pred_logits[idx,1] > threshold:
                                bounding_box = bbox[4:]
                                img_pred_track = cv2.rectangle(
                                    img_pred_track,
                                    (int(np.clip(bounding_box[0],0,width)), int(np.clip(bounding_box[1],0,height))),
                                    (int(np.clip(bounding_box[0] + bounding_box[2],0,width)), int(np.clip(bounding_box[1] + bounding_box[3],0,height))),
                                    color=colors[where[idx]] if TP_mask[where[idx]] else (0,0,0),
                                    thickness = 1)  

                                if args.display_all:
                                    img_pred_track = cv2.putText(
                                        img_pred_track,
                                        text = f'{pred_logits[idx,1]:.2f}', 
                                        org=(int(bounding_box[0]) - 5, int(bounding_box[1] + bounding_box[3] // 4 + int(fontscale*30))), 
                                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                                        fontScale = fontscale,
                                        color = colors[where[idx]] if TP_mask[where[idx]] else (128,128,128),
                                        thickness=1,
                                    )

                                if use_masks:
                                    mask = cv2.resize(pred_masks[idx,1],(width,height))
                                    mask = np.repeat(mask[...,None],3,axis=-1)
                                    mask[mask[...,0]>0] = colors[where[idx]] if idx < len(colors) else (128,128,128)
                                    img_pred_track[mask>0] = img_pred_track[mask>0]*(1-alpha) + mask[mask>0]*(alpha)
                        else:
                            bounding_box = bbox[:4]
                            img_pred_object = cv2.rectangle(
                                img_pred_object,
                                (int(np.clip(bounding_box[0],0,width)), int(np.clip(bounding_box[1],0,height))),
                                (int(np.clip(bounding_box[0] + bounding_box[2],0,width)), int(np.clip(bounding_box[1] + bounding_box[3],0,height))),
                                color=colors[where[idx]] if TP_mask[where[idx]] else (0,0,0),
                                thickness = 1)            

                            img_pred_object = cv2.putText(
                                img_pred_object,
                                text = f'{pred_logits[idx,0]:.2f}', 
                                org=(int(bounding_box[0]) - 5, int(bounding_box[1] + bounding_box[3] // 4 + int(fontscale*30))), 
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                                fontScale = fontscale,
                                color = colors[where[idx]] if TP_mask[where[idx]] else (128,128,128),
                                thickness=1,
                            )

                            if use_masks:
                                mask = cv2.resize(pred_masks[idx,0],(width,height))
                                mask = np.repeat(mask[...,None],3,axis=-1)
                                mask[mask[...,0]>0] = colors[where[idx]] if idx < len(colors) else (128,128,128)
                                img_pred_object[mask>0] = img_pred_object[mask>0]*(1-alpha) + mask[mask>0]*(alpha)
                            
                    if args.display_all:
                        if outputs_TM['pred_logits'][i].shape[0] == targets_TM[i]['track_queries_mask'].sum():
                            img_pred = np.concatenate((img_pred_track,img_all_pred),axis=1)
                        else:
                            img_pred = np.concatenate((img_pred_track,img_pred_object,img_all_pred),axis=1)
                    else:
                        img_pred = np.concatenate((img_pred_track,img),axis=1)

                    if i == 0:
                        if args.display_all:
                            dn_track = np.concatenate((previmg,previmg_noised,img_gt,img_noised_gt,img_pred,np.zeros((previmg.shape[0],20,3),dtype=previmg.dtype)),axis=1)
                        else:
                            dn_track = np.concatenate((previmg_noised,img_pred,np.zeros((previmg.shape[0],20,3),dtype=previmg.dtype)),axis=1)
                    else:
                        if args.display_all:
                            dn_track = np.concatenate((dn_track,previmg,previmg_noised,img_gt,img_noised_gt,img_pred),axis=1)
                        else:
                            dn_track = np.concatenate((dn_track,previmg,previmg_noised,img_pred),axis=1)

                cv2.imwrite(str(savepath / folder / 'dn_track' / filename),dn_track)

            elif meta_data_key == 'dn_enc':
                dn_encs = []
                for i in range(bs):

                    img = samples[i].permute(1,2,0)                
                    
                    enc_boxes_noised = targets_TM[i]['enc_boxes_noised'].detach().cpu().numpy()
                    enc_boxes = targets_TM[i]['enc_boxes'].detach().cpu().numpy()
                    boxes = targets_TM[i]['boxes'].detach().cpu().numpy()

                    enc_boxes_noised = box_converter.convert(enc_boxes_noised)
                    enc_boxes = box_converter.convert(enc_boxes)
                    boxes = box_converter.convert(boxes)

                    img = img.detach().cpu().numpy().copy()
                    img = np.repeat(np.mean(img,axis=-1)[:,:,np.newaxis],3,axis=-1)
                    img = (255*(img - np.min(img)) / np.ptp(img)).astype(np.uint8)
                    img_gt = img.copy()
                    img_enc_noised = img.copy()
                    img_enc = img.copy()
                    img_enc_mask = img.copy()
                    img_enc_thresh = img.copy()
                    img_pred = img.copy()
                    img_pred_all = img.copy()


                    for idx,bbox in enumerate(boxes):
                        bounding_box = bbox[:4]
                        img_gt = cv2.rectangle(
                            img_gt,
                            (int(np.clip(bounding_box[0],0,width)), int(np.clip(bounding_box[1],0,height))),
                            (int(np.clip(bounding_box[0] + bounding_box[2],0,width)), int(np.clip(bounding_box[1] + bounding_box[3],0,height))),
                            color=(0,0,0),
                            thickness = 1)


                    for idx,bbox in enumerate(enc_boxes_noised):
                        bounding_box = bbox[:4]
                        img_enc_noised = cv2.rectangle(
                            img_enc_noised,
                            (int(np.clip(bounding_box[0],0,width)), int(np.clip(bounding_box[1],0,height))),
                            (int(np.clip(bounding_box[0] + bounding_box[2],0,width)), int(np.clip(bounding_box[1] + bounding_box[3],0,height))),
                            color=colors[idx],
                            thickness = 1)

                    for idx,bbox in enumerate(enc_boxes):
                        bounding_box = bbox[:4]
                        img_enc = cv2.rectangle(
                            img_enc,
                            (int(np.clip(bounding_box[0],0,width)), int(np.clip(bounding_box[1],0,height))),
                            (int(np.clip(bounding_box[0] + bounding_box[2],0,width)), int(np.clip(bounding_box[1] + bounding_box[3],0,height))),
                            color=colors[idx],
                            thickness = 1)

                    for idx,bbox in enumerate(enc_boxes):
                        if targets_TM[i]['enc_logits'][idx] > 0.2:
                            bounding_box = bbox[:4]
                            img_enc_thresh = cv2.rectangle(
                                img_enc_thresh,
                                (int(np.clip(bounding_box[0],0,width)), int(np.clip(bounding_box[1],0,height))),
                                (int(np.clip(bounding_box[0] + bounding_box[2],0,width)), int(np.clip(bounding_box[1] + bounding_box[3],0,height))),
                                color=colors[idx],
                                thickness = 1)

                            img_enc_thresh = cv2.putText(
                                    img_enc_thresh,
                                    text = f'{targets_TM[i]["enc_logits"][idx]:.2f}', 
                                    org=(int(bounding_box[0]) - 5, int(bounding_box[1] + bounding_box[3] // 4 + int(fontscale*30))), 
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                                    fontScale = fontscale,
                                    color = colors[idx],
                                    thickness=1,
                                )

                    pred_boxes = outputs_TM['pred_boxes'][i].detach().cpu().numpy()
                    pred_logits = outputs_TM['pred_logits'][i].sigmoid().detach().cpu().numpy()

                    pred_boxes = box_converter.convert(pred_boxes)

                    for idx,bbox in enumerate(pred_boxes):
                        bounding_box = bbox[:4]
                        img_pred_all = cv2.rectangle(
                            img_pred_all,
                            (int(np.clip(bounding_box[0],0,width)), int(np.clip(bounding_box[1],0,height))),
                            (int(np.clip(bounding_box[0] + bounding_box[2],0,width)), int(np.clip(bounding_box[1] + bounding_box[3],0,height))),
                            color=colors[idx],
                            thickness = 1)  
                        
                        img_pred_all = cv2.putText(
                            img_pred_all,
                            text = f'{pred_logits[idx,0]:.2f}', 
                            org=(int(bounding_box[0]) - 5, int(bounding_box[1] + bounding_box[3] // 4 + int(fontscale*30))), 
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                            fontScale = fontscale,
                            color = colors[idx],
                            thickness=1,
                        )

                    keep = pred_logits[:,0] > threshold
                    where = np.where(pred_logits[:,0] > threshold)[0]
                    pred_boxes = pred_boxes[keep]
                    pred_logits = pred_logits[keep]

                    if use_masks:
                        pred_masks = outputs_TM['pred_masks'][i].sigmoid().detach().cpu().numpy()
                        pred_masks = pred_masks[keep]

                        if sum(keep) > 0:
                            masks_filt = np.zeros((pred_masks.shape))
                            argmax = np.argmax(pred_masks,axis=0)
                            for m in range(pred_masks.shape[0]):
                                masks_filt[m,argmax==m] = pred_masks[m,argmax==m]
                            masks_filt = masks_filt > mask_threshold
                            pred_masks = (masks_filt*255).astype(np.uint8)

                    for idx,bbox in enumerate(pred_boxes): 
                        bounding_box = bbox[:4]
                        img_pred = cv2.rectangle(
                            img_pred,
                            (int(np.clip(bounding_box[0],0,width)), int(np.clip(bounding_box[1],0,height))),
                            (int(np.clip(bounding_box[0] + bounding_box[2],0,width)), int(np.clip(bounding_box[1] + bounding_box[3],0,height))),
                            color=colors[where[idx]],
                            thickness = 1)            

                        img_pred = cv2.putText(
                            img_pred,
                            text = f'{pred_logits[idx,0]:.2f}', 
                            org=(int(bounding_box[0]) - 5, int(bounding_box[1] + bounding_box[3] // 4 + int(fontscale*30))), 
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                            fontScale = fontscale,
                            color = colors[where[idx]],
                            thickness=1,
                        )

                        if use_masks:
                            mask = cv2.resize(pred_masks[idx,0],(width,height))
                            mask = np.repeat(mask[...,None],3,axis=-1)
                            mask[mask[...,0]>0] = colors[idx] if idx < len(colors) else (128,128,128)
                            img_pred[mask>0] = img_pred[mask>0]*(1-alpha) + mask[mask>0]*(alpha)

                    dn_encs.append(np.concatenate((img,img_enc_mask,img_enc_thresh,img_enc,img_enc_noised,img_pred,img_pred_all,img_gt),axis=1))

                if len(dn_encs) == 1:
                    dn_enc = dn_encs[0]
                else:
                    dn_enc = np.concatenate(dn_encs,1)

                cv2.imwrite(str(savepath / folder / 'dn_enc' / filename),dn_enc)

            elif meta_data_key == 'dn_object':
                dn_object = np.zeros((height,(width*5 + spacer)*bs,3))
                for i in range(bs):

                    img = samples[i].permute(1,2,0)                
                    
                    boxes_noised = targets_TM[i]['noised_boxes'].detach().cpu().numpy()
                    boxes = targets_TM[i]['noised_boxes_gt'].detach().cpu().numpy()
                    TP_mask = ~targets_TM[i]['track_queries_fal_pos_mask'].cpu().numpy()
                
                    boxes = box_converter.convert(boxes)
                    boxes_noised = box_converter.convert(boxes_noised)

                    img = img.detach().cpu().numpy().copy()
                    img = np.repeat(np.mean(img,axis=-1)[:,:,np.newaxis],3,axis=-1)
                    img = (255*(img - np.min(img)) / np.ptp(img)).astype(np.uint8)
                    img_gt = img.copy()
                    img_noised_gt = img.copy()
                    img_pred_all = img.copy()
                    img_pred = img.copy()


                    for idx,bbox in enumerate(boxes):
                        bounding_box = bbox[:4]
                        img_gt = cv2.rectangle(
                            img_gt,
                            (int(np.clip(bounding_box[0],0,width)), int(np.clip(bounding_box[1],0,height))),
                            (int(np.clip(bounding_box[0] + bounding_box[2],0,width)), int(np.clip(bounding_box[1] + bounding_box[3],0,height))),
                            color=colors[idx] if TP_mask[idx] else (0,0,0),
                            thickness = 1)


                    for idx,bbox in enumerate(boxes_noised):
                        bounding_box = bbox[:4]
                        img_noised = cv2.rectangle(
                            img_noised_gt,
                            (int(np.clip(bounding_box[0],0,width)), int(np.clip(bounding_box[1],0,height))),
                            (int(np.clip(bounding_box[0] + bounding_box[2],0,width)), int(np.clip(bounding_box[1] + bounding_box[3],0,height))),
                            color=colors[idx] if TP_mask[idx] else (0,0,0),
                            thickness = 1)

                    pred_boxes = outputs_TM['pred_boxes'][i].detach().cpu().numpy()
                    pred_logits = outputs_TM['pred_logits'][i].sigmoid().detach().cpu().numpy()

                    pred_boxes = box_converter.convert(pred_boxes)

                    for idx,bbox in enumerate(pred_boxes):
                        bounding_box = bbox[:4]
                        img_pred_all = cv2.rectangle(
                            img_pred_all,
                            (int(np.clip(bounding_box[0],0,width)), int(np.clip(bounding_box[1],0,height))),
                            (int(np.clip(bounding_box[0] + bounding_box[2],0,width)), int(np.clip(bounding_box[1] + bounding_box[3],0,height))),
                            color=colors[idx] if TP_mask[idx] else (0,0,0),
                            thickness = 1)  
                        
                        if args.display_all:
                            img_pred_all = cv2.putText(
                                img_pred_all,
                                text = f'{pred_logits[idx,0]:.2f}', 
                                org=(int(bounding_box[0]) - 5, int(bounding_box[1] + bounding_box[3] // 4 + int(fontscale*30))), 
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                                fontScale = fontscale,
                                color = colors[idx] if TP_mask[idx] else (128,128,128),
                                thickness=1,
                            )

                    keep = pred_logits[:,0] > threshold
                    where = np.where(pred_logits[:,0] > threshold)[0]
                    pred_boxes = pred_boxes[keep]
                    pred_logits = pred_logits[keep]

                    for idx,bbox in enumerate(pred_boxes): 
                        bounding_box = bbox[:4]
                        img_pred = cv2.rectangle(
                            img_pred,
                            (int(np.clip(bounding_box[0],0,width)), int(np.clip(bounding_box[1],0,height))),
                            (int(np.clip(bounding_box[0] + bounding_box[2],0,width)), int(np.clip(bounding_box[1] + bounding_box[3],0,height))),
                            color=colors[where[idx]] if TP_mask[where[idx]] else (0,0,0),
                            thickness = 1)            

                        if args.display_all:
                            img_pred = cv2.putText(
                                img_pred,
                                text = f'{pred_logits[idx,0]:.2f}', 
                                org=(int(bounding_box[0]) - 5, int(bounding_box[1] + bounding_box[3] // 4 + int(fontscale*30))), 
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                                fontScale = fontscale,
                                color = colors[where[idx]] if TP_mask[where[idx]] else (128,128,128),
                                thickness=1,
                            )

                    dn_object[:,(width*5+spacer)*i + width*0:(width*5+spacer)*i + width*1] = img
                    dn_object[:,(width*5+spacer)*i + width*1:(width*5+spacer)*i + width*2] = img_gt
                    dn_object[:,(width*5+spacer)*i + width*2:(width*5+spacer)*i + width*3] = img_noised
                    dn_object[:,(width*5+spacer)*i + width*3:(width*5+spacer)*i + width*4] = img_pred
                    dn_object[:,(width*5+spacer)*i + width*4:(width*5+spacer)*i + width*5] = img_pred_all

                cv2.imwrite(str(savepath / folder / 'dn_object' / filename),dn_object)


    for i in range(bs):

        min_track_id = torch.min(targets[i]['prev_prev_target']['track_ids'])            
        max_track_id = torch.max(targets[i]['fut_target']['track_ids']) + 1            

        prev_prev_img = targets[i]['prev_prev_image'].permute(1,2,0).cpu().numpy()
        prev_prev_img = np.repeat(np.mean(prev_prev_img,axis=-1)[:,:,np.newaxis],3,axis=-1)
        prev_prev_img = (255*(prev_prev_img - np.min(prev_prev_img)) / np.ptp(prev_prev_img)).astype(np.uint8)
        prev_prev_img_gt = prev_prev_img.copy()

        previmg = targets[i]['prev_image'].permute(1,2,0)
        previmg = previmg.detach().cpu().numpy().copy()
        previmg = np.repeat(np.mean(previmg,axis=-1)[:,:,np.newaxis],3,axis=-1)
        previmg = (255*(previmg - np.min(previmg)) / np.ptp(previmg)).astype(np.uint8)
        previmg_track_only = previmg.copy()
        previmg_object_only = previmg.copy()

        img = samples[i].permute(1,2,0)
        img = img.detach().cpu().numpy().copy()
        img = np.repeat(np.mean(img,axis=-1)[:,:,np.newaxis],3,axis=-1)
        img = (255*(img - np.min(img)) / np.ptp(img)).astype(np.uint8)
        img_pred = img.copy()
        img_masks = img.copy()

        fut_img = targets[i]['fut_image'].permute(1,2,0).cpu().numpy()
        fut_img = np.repeat(np.mean(fut_img,axis=-1)[:,:,np.newaxis],3,axis=-1)
        fut_img = (255*(fut_img - np.min(fut_img)) / np.ptp(fut_img)).astype(np.uint8)
        
        if prev_outputs is not None:
            prev_bbs = prev_outputs['pred_boxes'][i].detach().cpu().numpy()
            prev_classes = prev_outputs['pred_logits'][i].sigmoid().detach().cpu().numpy()

            prev_keep = prev_classes[:,0] > threshold
            prev_bbs = prev_bbs[prev_keep]
            prev_classes = prev_classes[prev_keep]

            keep_ind = prev_keep.nonzero()[0]

            if sum(prev_keep) > 0:
                prev_bbs = box_converter.convert(prev_bbs)

            for idx,bbox in enumerate(prev_bbs):
                if keep_ind[idx] < prev_outputs['pred_logits'][i].shape[0] - (~cur_targets[i]['track_queries_mask']).sum():
                    bounding_box = bbox[:4]
                    previmg_track_only = cv2.rectangle(
                        previmg_track_only,
                        (int(np.clip(bounding_box[0],0,width)), int(np.clip(bounding_box[1],0,height))),
                        (int(np.clip(bounding_box[0] + bounding_box[2],0,width)), int(np.clip(bounding_box[1] + bounding_box[3],0,height))),
                        color=(255,0,0),
                        thickness = 1)

                    if args.display_all:
                        previmg_track_only = cv2.putText(
                            previmg_track_only,
                            text = f'{prev_classes[idx,0]:.2f}', 
                            org=(int(bounding_box[0]) - 5, int(bounding_box[1] + bounding_box[3] // 4 + int(fontscale*30))), 
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                            fontScale = 0.4,
                            color = (128,128,128),
                            thickness=1,
                        )

                    if prev_classes[idx,1] > 0.5:
                        bounding_box = bbox[4:]
                        previmg_track_only = cv2.rectangle(
                            previmg_track_only,
                            (int(np.clip(bounding_box[0],0,width)), int(np.clip(bounding_box[1],0,height))),
                            (int(np.clip(bounding_box[0] + bounding_box[2],0,width)), int(np.clip(bounding_box[1] + bounding_box[3],0,height))),
                            color = (0,0,255),
                            thickness = 1)

                        if args.display_all:
                            previmg_track_only = cv2.putText(
                                previmg_track_only,
                                text = f'{prev_classes[idx,1]:.2f}', 
                                org=(int(bounding_box[0]) - 5, int(bounding_box[1] + bounding_box[3] // 4 + int(fontscale*30))), 
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                                fontScale = fontscale,
                                color = (128,128,128),
                                thickness=1,
                            )
                else:
                    bounding_box = bbox[:4]
                    previmg_object_only = cv2.rectangle(
                        previmg_object_only,
                        (int(np.clip(bounding_box[0],0,width)), int(np.clip(bounding_box[1],0,height))),
                        (int(np.clip(bounding_box[0] + bounding_box[2],0,width)), int(np.clip(bounding_box[1] + bounding_box[3],0,height))),
                        color=(0,0,0),
                        thickness = 1)

                    if args.display_all:
                        previmg_object_only = cv2.putText(
                            previmg_object_only,
                            text = f'{prev_classes[idx,0]:.2f}', 
                            org=(int(bounding_box[0]) - 5, int(bounding_box[1] + bounding_box[3] // 4 + int(fontscale*30))), 
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                            fontScale = 0.4,
                            color = (128,128,128),
                            thickness=1,
                        )

                    if prev_classes[idx,1] > 0.5:
                        bounding_box = bbox[4:]
                        previmg_object_only = cv2.rectangle(
                            previmg_object_only,
                            (int(np.clip(bounding_box[0],0,width)), int(np.clip(bounding_box[1],0,height))),
                            (int(np.clip(bounding_box[0] + bounding_box[2],0,width)), int(np.clip(bounding_box[1] + bounding_box[3],0,height))),
                            color = (128,128,128),
                            thickness = 1)

                        if args.display_all:
                            previmg_object_only = cv2.putText(
                                previmg_object_only,
                                text = f'{prev_classes[idx,1]:.2f}', 
                                org=(int(bounding_box[0]) - 5, int(bounding_box[1] + bounding_box[3] // 4 + int(fontscale*30))), 
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                                fontScale = fontscale,
                                color = (0,0,0),
                                thickness=1,
                            )

            track_query_boxes = cur_targets[i]['track_query_boxes'].detach().cpu()

            if track_query_boxes.shape[0] > 0:
        
                track_query_boxes[:,1::2] = track_query_boxes[:,1::2] * height
                track_query_boxes[:,::2] = track_query_boxes[:,::2] * width
                track_query_boxes[:,0] = track_query_boxes[:,0] - torch.div(track_query_boxes[:,2],2,rounding_mode='floor')
                track_query_boxes[:,1] = track_query_boxes[:,1] - torch.div(track_query_boxes[:,3],2,rounding_mode='floor')

            previmg_keep_queries = previmg.copy()

            track_ids = targets[i]['prev_target']['track_ids'][cur_targets[i]['prev_ind'][1]]
            track_ids = torch.cat((track_ids,torch.tensor([max_track_id+idx for idx in range(track_query_boxes.shape[0]-len(track_ids))],dtype=track_ids.dtype).to(args.device)))

            if args.display_all:
                for idx,bounding_box in enumerate(prev_bbs):
                    previmg_keep_queries = cv2.rectangle(
                        previmg_keep_queries,
                        (int(np.clip(bounding_box[0],0,width)), int(np.clip(bounding_box[1],0,height))),
                        (int(np.clip(bounding_box[0] + bounding_box[2],0,width)), int(np.clip(bounding_box[1] + bounding_box[3],0,height))),
                        color=(0,0,0),
                        thickness = 1 )

            for idx,bounding_box in enumerate(track_query_boxes):
                previmg_keep_queries = cv2.rectangle(
                    previmg_keep_queries,
                    (int(np.clip(bounding_box[0],0,width)), int(np.clip(bounding_box[1],0,height))),
                    (int(np.clip(bounding_box[0] + bounding_box[2],0,width)), int(np.clip(bounding_box[1] + bounding_box[3],0,height))),
                    color=colors[track_ids[idx] - min_track_id] if idx < len(track_ids) else colors[-idx],
                    thickness = 1 if cur_targets[i]['track_queries_TP_mask'][cur_targets[i]['track_queries_mask']][idx] else 1)


            if args.display_all:
                if prev_outputs['pred_logits'][i].shape[0] > (~cur_targets[i]['track_queries_mask']).sum():
                    pred = np.concatenate((previmg_track_only,previmg_object_only,previmg_keep_queries),axis=1)
                else:
                    pred = np.concatenate((previmg_object_only,previmg_keep_queries),axis=1)
            else:
                pred = previmg_keep_queries
                    

        if prev_outputs is not None and 'enc_outputs' in prev_outputs:
            enc_outputs = prev_outputs['enc_outputs']
            enc_pred_logits = enc_outputs['pred_logits'].cpu()
            enc_pred_boxes = enc_outputs['pred_boxes'].cpu()

            logits_topk, ind_topk = torch.topk(enc_pred_logits[i,:,0].sigmoid(),args.num_queries)
            boxes_topk = enc_pred_boxes[i,ind_topk]

            boxes_topk = box_converter.convert(boxes_topk.numpy())

            t0,t1,t2,t3 = 0.1,0.3,0.5,0.8
            boxes_list = []
            boxes_list.append(boxes_topk[logits_topk < t0])
            boxes_list.append(boxes_topk[(logits_topk > t0) * (logits_topk < t1)])
            boxes_list.append(boxes_topk[(logits_topk > t1) * (logits_topk < t2)])
            boxes_list.append(boxes_topk[(logits_topk > t2) * (logits_topk < t3)])
            boxes_list.append(boxes_topk[logits_topk > t3])

            enc_frames = []
            for boxes in boxes_list:
                enc_frame = np.array(previmg).copy()
                for idx,bounding_box in enumerate(boxes):
                    enc_frame = cv2.rectangle(
                    enc_frame,
                    (int(np.clip(bounding_box[0],0,width)), int(np.clip(bounding_box[1],0,height))),
                    (int(np.clip(bounding_box[0] + bounding_box[2],0,width)), int(np.clip(bounding_box[1] + bounding_box[3],0,height))),
                    color=colors[idx],
                    thickness = 1)
                enc_frames.append(enc_frame)

            enc_frames_prev = np.concatenate((enc_frames),axis=1)

        
        bbs = outputs['pred_boxes'][i].detach().cpu().numpy()
        classes = outputs['pred_logits'][i].sigmoid().detach().cpu().numpy()

        keep = classes[:,0] > threshold
        bbs = bbs[keep]
        classes = classes[keep]

        if use_masks:
            masks = outputs['pred_masks'][i].detach().cpu().sigmoid().numpy()[keep]

            if sum(keep) > 0:
                masks_filt = np.zeros((masks.shape))
                argmax = np.argmax(masks,axis=0)
                for m in range(masks.shape[0]):
                    masks_filt[m,argmax==m] = masks[m,argmax==m]
                masks_filt = masks_filt > mask_threshold
                masks = (masks_filt*255).astype(np.uint8)

        if prev_outputs is not None:
            track_keep = keep[:track_query_boxes.shape[0]]
            colors_select = [colors[track_ids[idx] - min_track_id] for idx in range(len(track_keep)) if track_keep[idx]] 
        else:
            track_ids = cur_targets[i]['track_ids'][cur_targets[i]['indices'][1]]
            colors_select = [colors[track_id-min_track_id] for track_id in track_ids]

        if sum(keep) > 0:
            bbs = box_converter.convert(bbs)

        for idx, bbox in enumerate(bbs):
            bounding_box = bbox[:4]
            img_pred = cv2.rectangle(
                img_pred,
                (int(np.clip(bounding_box[0],0,width)), int(np.clip(bounding_box[1],0,height))),
                (int(np.clip(bounding_box[0] + bounding_box[2],0,width)), int(np.clip(bounding_box[1] + bounding_box[3],0,height))),
                color=colors_select[idx] if idx < len(colors_select) else (0,0,0),
                thickness = 1)

            if args.display_all:
                img_pred = cv2.putText(
                    img_pred,
                    text = f'{classes[idx,0]:.2f}', 
                    org=(int(bounding_box[0]) - 5, int(bounding_box[1] + bounding_box[3] // 4 + int(fontscale*30))), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale = fontscale,
                    color = colors_select[idx] if idx < len(colors_select) else (128,128,128),
                    thickness=1,
                )

            if use_masks:
                mask = cv2.resize(masks[idx,0],(width,height)) 
                mask = np.repeat(mask[...,None],3,axis=-1)
                mask[mask[...,0]>0] = colors_select[idx] if idx < len(colors_select) else (128,128,128)
                img_masks[mask>0] = img_masks[mask>0]*(1-alpha) + mask[mask>0]*(alpha)


            if classes[idx,1] > threshold:
                bounding_box = bbox[4:]
                img_pred = cv2.rectangle(
                    img_pred,
                    (int(np.clip(bounding_box[0],0,width)), int(np.clip(bounding_box[1],0,height))),
                    (int(np.clip(bounding_box[0] + bounding_box[2],0,width)), int(np.clip(bounding_box[1] + bounding_box[3],0,height))),
                    color=colors_select[idx] if idx < len(colors_select) else (0,0,0),
                    thickness = 1)

                if args.display_all:
                    img_pred = cv2.putText(
                        img_pred,
                        text = f'{classes[idx,1]:.2f}', 
                        org=(int(bounding_box[0]) - 5, int(bounding_box[1] + bounding_box[3] // 4 + int(fontscale*30))), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale = fontscale,
                        color = colors_select[idx] if idx < len(colors_select) else (128,128,128),
                        thickness=1,
                    )

                if use_masks:
                    mask = cv2.resize(masks[idx,1],(width,height))
                    mask = np.repeat(mask[...,None],3,axis=-1)
                    mask[mask[...,0]>0] = colors_select[idx] if idx < len(colors_select) else (128,128,128)
                    img_masks[mask>0] = img_masks[mask>0]*(1-alpha) + mask[mask>0]*(alpha)
                    
        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            enc_pred_logits = enc_outputs['pred_logits'].detach().cpu()
            enc_pred_boxes = enc_outputs['pred_boxes'].detach().cpu()

            logits_topk, ind_topk = torch.topk(enc_pred_logits[i,:,0].sigmoid(),args.num_queries)
            boxes_topk = enc_pred_boxes[i,ind_topk]

            boxes_topk = box_converter.convert(boxes_topk.numpy())

            t0,t1,t2,t3 = 0.1,0.3,0.5,0.8
            boxes_list = []
            boxes_list.append(boxes_topk[logits_topk < t0])
            boxes_list.append(boxes_topk[(logits_topk > t0) * (logits_topk < t1)])
            boxes_list.append(boxes_topk[(logits_topk > t1) * (logits_topk < t2)])
            boxes_list.append(boxes_topk[(logits_topk > t2) * (logits_topk < t3)])
            boxes_list.append(boxes_topk[logits_topk > t3])

            if 'pred_masks' in enc_outputs:
                enc_pred_masks = enc_outputs['pred_masks'].detach().cpu().sigmoid()
                masks_topk = enc_pred_masks[i,ind_topk].numpy()

                masks_filt = np.zeros((masks_topk.shape))
                argmax = np.argmax(masks_topk,axis=0)
                for m in range(masks_topk.shape[0]):
                    masks_filt[m,argmax==m] = masks_topk[m,argmax==m]
                masks_filt = masks_filt > mask_threshold
                masks_topk = (masks_filt*255).astype(np.uint8)

                masks_list = []
                masks_list.append(masks_topk[logits_topk < t0])
                masks_list.append(masks_topk[(logits_topk > t0) * (logits_topk < t1)])
                masks_list.append(masks_topk[(logits_topk > t1) * (logits_topk < t2)])
                masks_list.append(masks_topk[(logits_topk > t2) * (logits_topk < t3)])
                masks_list.append(masks_topk[logits_topk > t3])

            enc_frames = []
            for b,boxes in enumerate(boxes_list):
                enc_frame = np.array(img).copy()
                for idx,bounding_box in enumerate(boxes):
                    enc_frame = cv2.rectangle(
                    enc_frame,
                    (int(np.clip(bounding_box[0],0,width)), int(np.clip(bounding_box[1],0,height))),
                    (int(np.clip(bounding_box[0] + bounding_box[2],0,width)), int(np.clip(bounding_box[1] + bounding_box[3],0,height))),
                    color=colors[idx],
                    thickness = 1)

                    if 'pred_masks' in enc_outputs and b > 0:
                        mask = masks_list[b][idx,0]
                        mask = cv2.resize(mask,(width,height))
                        mask = np.repeat(mask[...,None],3,axis=-1)
                        mask[mask[...,0]>0] = colors[idx] if idx < len(colors) else (128,128,128)
                        enc_frame[mask>0] = enc_frame[mask>0]*(1-alpha) + mask[mask>0]*(alpha)

                enc_frames.append(enc_frame)

            enc_frames_cur = np.concatenate((enc_frames),axis=1)

            if prev_outputs is not None:
                enc_frames = np.concatenate((enc_frames_prev,enc_frames_cur),axis=1)
            else:
                enc_frames = enc_frames_cur

            batch_enc_frames.append(enc_frames)

        if prev_outputs is not None: 
            pred = np.concatenate((pred,img_pred,img_masks),axis=1)
        else:
            pred = np.concatenate((img_pred,img_masks),axis=1)

        if print_gt:
    
            target = targets[i]
            cur_target = cur_targets[i]
            target_og = targets_og[i]
            man_track = target['man_track']
            man_track_og = target_og['man_track']

            # If just object detection, then just current frame is used so two frames back gives us no valuable information
            bbs = target_og['prev_prev_target']['boxes'].cpu().numpy()
            bbs = box_converter.convert(bbs)

            track_ids = target_og['prev_prev_target']['track_ids'].cpu().numpy()

            if use_masks:
                masks = target_og['prev_prev_target']['masks'].cpu().numpy()

            for idx, bbox in enumerate(bbs):
                bounding_box = bbox[:4]
                prev_prev_img_gt = cv2.rectangle(
                    prev_prev_img_gt,
                    (int(np.clip(bounding_box[0],0,width)), int(np.clip(bounding_box[1],0,height))),
                    (int(np.clip(bounding_box[0] + bounding_box[2],0,width)), int(np.clip(bounding_box[1] + bounding_box[3],0,height))),
                    color=colors[track_ids[idx]-min_track_id],
                    thickness = 1)

                if use_masks:
                    mask = masks[idx,0]
                    mask = np.repeat(mask[...,None],3,axis=-1)
                    mask[mask[...,0]>0] = colors[track_ids[idx]-min_track_id] if idx < len(colors) else (128,128,128)
                    prev_prev_img_gt[mask>0] = prev_prev_img_gt[mask>0]*(1-alpha) + mask[mask>0]*(alpha)

                if bbs[idx,-1] > 0:
                    bounding_box = bbox[4:]
                    prev_prev_img_gt = cv2.rectangle(
                        prev_prev_img_gt,
                        (int(np.clip(bounding_box[0],0,width)), int(np.clip(bounding_box[1],0,height))),
                        (int(np.clip(bounding_box[0] + bounding_box[2],0,width)), int(np.clip(bounding_box[1] + bounding_box[3],0,height))),
                        color=colors[track_ids[idx]-min_track_id],
                        thickness = 1)

                    if use_masks:
                        mask = masks[idx,1]
                        mask = np.repeat(mask[...,None],3,axis=-1)
                        mask[mask[...,0]>0] = colors[track_ids[idx]-min_track_id] if idx < len(colors) else (128,128,128)
                        prev_prev_img_gt[mask>0] = prev_prev_img_gt[mask>0]*(1-alpha) + mask[mask>0]*(alpha)

            previmg_gt = previmg.copy()

            prev_target = target['prev_target']
            bbs = prev_target['boxes'].detach().cpu().numpy()
            bbs = box_converter.convert(bbs)

            track_ids = prev_target['track_ids'].cpu().numpy()

            if use_masks:
                masks = prev_target['masks'].detach().cpu().numpy()

            for idx, bbox in enumerate(bbs):
                bounding_box = bbox[:4]
                previmg_gt = cv2.rectangle(
                    previmg_gt,
                    (int(np.clip(bounding_box[0],0,width)), int(np.clip(bounding_box[1],0,height))),
                    (int(np.clip(bounding_box[0] + bounding_box[2],0,width)), int(np.clip(bounding_box[1] + bounding_box[3],0,height))),
                    color=colors[track_ids[idx]-min_track_id],
                    thickness = 1)

                if use_masks:
                    mask = masks[idx,0]
                    mask = np.repeat(mask[...,None],3,axis=-1)
                    mask[mask[...,0]>0] = colors[track_ids[idx]-min_track_id] if idx < len(colors) else (128,128,128)
                    previmg_gt[mask>0] = previmg_gt[mask>0]*(1-alpha) + mask[mask>0]*(alpha)

                if bbs[idx,-1] > 0:
                    bounding_box = bbox[4:]
                    previmg_gt = cv2.rectangle(
                        previmg_gt,
                        (int(np.clip(bounding_box[0],0,width)), int(np.clip(bounding_box[1],0,height))),
                        (int(np.clip(bounding_box[0] + bounding_box[2],0,width)), int(np.clip(bounding_box[1] + bounding_box[3],0,height))),
                        color= colors[track_ids[idx]-min_track_id],
                        thickness = 1)

                    if use_masks:
                        mask = masks[idx,1]
                        mask = np.repeat(mask[...,None],3,axis=-1)
                        mask[mask[...,0]>0] = colors[track_ids[idx]-min_track_id] if idx < len(colors) else (128,128,128)
                        previmg_gt[mask>0] = previmg_gt[mask>0]*(1-alpha) + mask[mask>0]*(alpha)

                if man_track[track_ids[idx]-1,1] == target['prev_target']['framenb']:
                    mother_id = man_track[track_ids[idx]-1,-1]
                    previmg_gt = cv2.circle(previmg_gt, (int(np.clip(bounding_box[0] + bounding_box[2] / 2,0,width)), int(np.clip(bounding_box[1] + bounding_box[3] / 2,0,height))), radius=1, color=colors[mother_id - min_track_id], thickness=-1)

            previmg_gt_og = previmg.copy()

            bbs = target_og['prev_target']['boxes'].detach().cpu().numpy()
            bbs = box_converter.convert(bbs)

            track_ids = target_og['prev_target']['track_ids'].cpu().numpy()

            if use_masks:
                masks = target_og['prev_target']['masks'].detach().cpu().numpy()

            for idx, bbox in enumerate(bbs):
                bounding_box = bbox[:4]
                previmg_gt_og = cv2.rectangle(
                    previmg_gt_og,
                    (int(np.clip(bounding_box[0],0,width)), int(np.clip(bounding_box[1],0,height))),
                    (int(np.clip(bounding_box[0] + bounding_box[2],0,width)), int(np.clip(bounding_box[1] + bounding_box[3],0,height))),
                    color=colors[track_ids[idx]-min_track_id],
                    thickness = 1)

                if 'masks' in target_og['prev_target']:
                    mask = masks[idx,0]
                    mask = np.repeat(mask[...,None],3,axis=-1)
                    mask[mask[...,0]>0] = colors[track_ids[idx]-min_track_id] if idx < len(colors) else (128,128,128)
                    previmg_gt_og[mask>0] = previmg_gt_og[mask>0]*(1-alpha) + mask[mask>0]*(alpha)

                if bbs[idx,-1] > 0:
                    bounding_box = bbox[4:]
                    previmg_gt_og = cv2.rectangle(
                        previmg_gt_og,
                        (int(np.clip(bounding_box[0],0,width)), int(np.clip(bounding_box[1],0,height))),
                        (int(np.clip(bounding_box[0] + bounding_box[2],0,width)), int(np.clip(bounding_box[1] + bounding_box[3],0,height))),
                        color= colors[track_ids[idx]-min_track_id],
                        thickness = 1)

                    if use_masks:
                        mask = masks[idx,1]
                        mask = np.repeat(mask[...,None],3,axis=-1)
                        mask[mask[...,0]>0] = colors[track_ids[idx]-min_track_id] if idx < len(colors) else (128,128,128)
                        previmg_gt_og[mask>0] = previmg_gt_og[mask>0]*(1-alpha) + mask[mask>0]*(alpha)

                if man_track_og[track_ids[idx]-1,1] == target['prev_target']['framenb']:
                    mother_id = man_track_og[track_ids[idx]-1,-1]
                    previmg_gt_og = cv2.circle(previmg_gt_og, (int(np.clip(bounding_box[0] + bounding_box[2] / 2,0,width)), int(np.clip(bounding_box[1] + bounding_box[3] / 2,0,height))), radius=2, color=colors[mother_id - min_track_id], thickness=-1)


            img_gt = img.copy()

            bbs = cur_target['boxes'].detach().cpu().numpy()
            bbs = box_converter.convert(bbs)

            track_ids = cur_target['track_ids'].cpu().numpy()

            if use_masks:
                masks = cur_target['masks'].detach().cpu().numpy()

            for idx, bbox in enumerate(bbs):
                bounding_box = bbox[:4]
                img_gt = cv2.rectangle(
                    img_gt,
                    (int(np.clip(bounding_box[0],0,width)), int(np.clip(bounding_box[1],0,height))),
                    (int(np.clip(bounding_box[0] + bounding_box[2],0,width)), int(np.clip(bounding_box[1] + bounding_box[3],0,height))),
                    color=colors[track_ids[idx]-min_track_id],
                    thickness = 1)

                if use_masks:
                    mask = masks[idx,0]
                    mask = np.repeat(mask[...,None],3,axis=-1)
                    mask[mask[...,0]>0] = colors[track_ids[idx]-min_track_id] if idx < len(colors) else (128,128,128)
                    img_gt[mask>0] = img_gt[mask>0]*(1-alpha) + mask[mask>0]*(alpha)

                if bbs[idx,-1] > 0:
                    bounding_box = bbox[4:]
                    img_gt = cv2.rectangle(
                        img_gt,
                        (int(np.clip(bounding_box[0],0,width)), int(np.clip(bounding_box[1],0,height))),
                        (int(np.clip(bounding_box[0] + bounding_box[2],0,width)), int(np.clip(bounding_box[1] + bounding_box[3],0,height))),
                        color=colors[track_ids[idx]-min_track_id],
                        thickness = 1)

                    if use_masks:
                        mask = masks[idx,1]
                        mask = np.repeat(mask[...,None],3,axis=-1)
                        mask[mask[...,0]>0] = colors[track_ids[idx]-min_track_id] if idx < len(colors) else (128,128,128)
                        img_gt[mask>0] = img_gt[mask>0]*(1-alpha) + mask[mask>0]*(alpha)

                if man_track[track_ids[idx]-1,1] == target['cur_target']['framenb']:
                    mother_id = man_track[track_ids[idx]-1,-1]
                    img_gt = cv2.circle(img_gt, (int(np.clip(bounding_box[0] + bounding_box[2] / 2,0,width)), int(np.clip(bounding_box[1] + bounding_box[3] / 2,0,height))), radius=2, color=colors[mother_id - min_track_id], thickness=-1)


            img_gt_og = img.copy()
            bbs = target_og['cur_target']['boxes'].detach().cpu().numpy()
            bbs = box_converter.convert(bbs)

            track_ids = target_og['cur_target']['track_ids'].cpu().numpy()

            if use_masks:
                masks = target_og['cur_target']['masks'].detach().cpu().numpy()

            for idx, bbox in enumerate(bbs):
                bounding_box = bbox[:4]
                img_gt_og = cv2.rectangle(
                    img_gt_og,
                    (int(np.clip(bounding_box[0],0,width)), int(np.clip(bounding_box[1],0,height))),
                    (int(np.clip(bounding_box[0] + bounding_box[2],0,width)), int(np.clip(bounding_box[1] + bounding_box[3],0,height))),
                    color=colors[track_ids[idx]-min_track_id],
                    thickness = 1)

                if use_masks:
                    mask = masks[idx,0]
                    mask = np.repeat(mask[...,None],3,axis=-1)
                    mask[mask[...,0]>0] = colors[track_ids[idx]-min_track_id] if idx < len(colors) else (128,128,128)
                    img_gt_og[mask>0] = img_gt_og[mask>0]*(1-alpha) + mask[mask>0]*(alpha)

                if bbs[idx,-1] > 0:
                    bounding_box = bbox[4:]
                    img_gt_og = cv2.rectangle(
                        img_gt_og,
                        (int(np.clip(bounding_box[0],0,width)), int(np.clip(bounding_box[1],0,height))),
                        (int(np.clip(bounding_box[0] + bounding_box[2],0,width)), int(np.clip(bounding_box[1] + bounding_box[3],0,height))),
                        color=colors[track_ids[idx]-min_track_id],
                        thickness = 1)

                    if use_masks:
                        mask = masks[idx,1]
                        mask = np.repeat(mask[...,None],3,axis=-1)
                        mask[mask[...,0]>0] = colors[track_ids[idx]-min_track_id] if idx < len(colors) else (128,128,128)
                        img_gt_og[mask>0] = img_gt_og[mask>0]*(1-alpha) + mask[mask>0]*(alpha)

                if man_track_og[track_ids[idx]-1,1] == target['cur_target']['framenb']:
                    mother_id = man_track_og[track_ids[idx]-1,-1]
                    img_gt_og = cv2.circle(img_gt_og, (int(np.clip(bounding_box[0] + bounding_box[2] / 2,0,width)), int(np.clip(bounding_box[1] + bounding_box[3] / 2,0,height))), radius=2, color=colors[mother_id - min_track_id], thickness=-1)


            fut_img_gt = fut_img.copy()

            bbs = target_og['fut_target']['boxes'].detach().cpu().numpy()
            bbs = box_converter.convert(bbs)

            track_ids = target_og['fut_target']['track_ids'].cpu().numpy()

            if use_masks:
                masks = target_og['fut_target']['masks'].detach().cpu().numpy()

            for idx, bbox in enumerate(bbs):
                bounding_box = bbox[:4]
                fut_img_gt = cv2.rectangle(
                    fut_img_gt,
                    (int(np.clip(bounding_box[0],0,width)), int(np.clip(bounding_box[1],0,height))),
                    (int(np.clip(bounding_box[0] + bounding_box[2],0,width)), int(np.clip(bounding_box[1] + bounding_box[3],0,height))),
                    color=colors[track_ids[idx]-min_track_id],
                    thickness = 1)

                if use_masks:
                    mask = masks[idx,0]
                    mask = np.repeat(mask[...,None],3,axis=-1)
                    mask[mask[...,0]>0] = colors[track_ids[idx]-min_track_id] if idx < len(colors) else (128,128,128)
                    fut_img_gt[mask>0] = fut_img_gt[mask>0]*(1-alpha) + mask[mask>0]*(alpha)

                if bbs[idx,-1] > 0:
                    bounding_box = bbox[4:]
                    fut_img_gt = cv2.rectangle(
                        fut_img_gt,
                        (int(np.clip(bounding_box[0],0,width)), int(np.clip(bounding_box[1],0,height))),
                        (int(np.clip(bounding_box[0] + bounding_box[2],0,width)), int(np.clip(bounding_box[1] + bounding_box[3],0,height))),
                        color=colors[track_ids[idx]-min_track_id],
                        thickness = 1)

                    if use_masks:
                        mask = masks[idx,1]
                        mask = np.repeat(mask[...,None],3,axis=-1)
                        mask[mask[...,0]>0] = colors[track_ids[idx]-min_track_id] if idx < len(colors) else (128,128,128)
                        fut_img_gt[mask>0] = fut_img_gt[mask>0]*(1-alpha) + mask[mask>0]*(alpha)


                if man_track[track_ids[idx]-1,1] == target['fut_target']['framenb']:
                    mother_id = man_track[track_ids[idx]-1,-1]
                    fut_img_gt = cv2.circle(fut_img_gt, (int(np.clip(bounding_box[0] + bounding_box[2] / 2,0,width)), int(np.clip(bounding_box[1] + bounding_box[3] / 2,0,height))), radius=2, color=colors[mother_id - min_track_id], thickness=-1)

            if args.display_all:
                if prev_outputs is not None:
                    gts = np.concatenate((previmg_gt,img_gt,np.zeros((img.shape[0],10,3)),prev_prev_img_gt,previmg_gt_og,img_gt_og,fut_img_gt),axis=1)
                else:
                    gts = np.concatenate((img_gt,np.zeros((img.shape[0],10,3)),previmg_gt_og,img_gt_og,fut_img_gt),axis=1)
            else:
                gts = img_gt

        if args.display_all:
            raw_images = np.concatenate((prev_prev_img,previmg,img,fut_img),axis=1)
        else:
            raw_images = img
        
        if i == 0:
            res = np.concatenate((pred,np.zeros((img.shape[0],10,3)),gts,np.zeros((img.shape[0],10,3)),raw_images),axis=1)
        else:
            res = np.concatenate((res,np.concatenate((pred,np.zeros((img.shape[0],10,3)),gts,np.zeros((img.shape[0],10,3)),raw_images),axis=1)),axis=0)

    cv2.imwrite(str(savepath / folder / 'standard' / filename),res)

    if 'enc_outputs' in outputs:
        cv2.imwrite(str(savepath / folder / 'enc_outputs' / filename),np.concatenate((batch_enc_frames),axis=1))


def plot_tracking_results(img,bbs,masks,colors,cells=None,div_track=None,new_cells=None,track=True):

    img = np.array(img)
    height = img.shape[0]
    width = img.shape[1]
    mask_threshold = 0.5
    alpha = 0.4
    box_converter = box_cxcy_to_xyxy(height,width)

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
            mask[mask[...,0]>0] = colors[idx] if idx < len(colors) else (128,128,128)
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


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def all_gather(data):

    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, print_freq, delimiter="\t", vis=None, debug=False):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.vis = vis
        self.print_freq = print_freq
        self.debug = debug

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {meter}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, epoch=None, header=None):
        i = 0
        if header is None:
            header = 'Epoch: [{}]'.format(epoch)

        world_len_iterable = get_world_size() * len(iterable)

        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(world_len_iterable))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data_time: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % self.print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i * get_world_size(), world_len_iterable, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i * get_world_size(), world_len_iterable, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))

                if self.vis is not None:
                    y_data = [self.meters[legend_name].median
                              for legend_name in self.vis.viz_opts['legend']
                              if legend_name in self.meters]
                    y_data.append(iter_time.median)

                    self.vis.plot(y_data, i * get_world_size() + (epoch - 1) * world_len_iterable)

                # DEBUG
                # if i != 0 and i % self.print_freq == 0:
                if self.debug and i % self.print_freq == 0:
                    break

            i += 1
            end = time.time()

        # if self.vis is not None:
        #     self.vis.reset()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, _, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False

    elif tensor_list[0].ndim == 4:
        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size 
        b, _, n, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, n, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2], : img.shape[3]].copy_(img)
            m[:, : img.shape[2], :img.shape[3]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor] = None):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

    def unmasked_tensor(self, index: int):
        tensor = self.tensors[index]

        if not self.mask[index].any():
            return tensor

        h_index = self.mask[index, 0, :].nonzero(as_tuple=True)[0]
        if len(h_index):
            tensor = tensor[:, :, :h_index[0]]

        w_index = self.mask[index, :, 0].nonzero(as_tuple=True)[0]
        if len(w_index):
            tensor = tensor[:, :w_index[0], :]

        return tensor


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)

        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

    if not is_master:
        def line(*args, **kwargs):
            pass
        def images(*args, **kwargs):
            pass
        Visdom.line = line
        Visdom.images = images


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ and 'SLURM_PTY_PORT' not in os.environ:
        # slurm process but not interactive
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print(f'| distributed init (rank {args.rank}): {args.dist_url}', flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)
    # torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    if float(torchvision.__version__[:3]) < 0.7:
        if input.numel() > 0:
            return torch.nn.functional.interpolate(
                input, size, scale_factor, mode, align_corners
            )

        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)


class DistributedWeightedSampler(torch.utils.data.DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, replacement=True):
        super(DistributedWeightedSampler, self).__init__(dataset, num_replicas, rank, shuffle)

        assert replacement

        self.replacement = replacement

    def __iter__(self):
        iter_indices = super(DistributedWeightedSampler, self).__iter__()
        if hasattr(self.dataset, 'sample_weight'):
            indices = list(iter_indices)

            weights = torch.tensor([self.dataset.sample_weight(idx) for idx in indices])

            g = torch.Generator()
            g.manual_seed(self.epoch)

            weight_indices = torch.multinomial(
                weights, self.num_samples, self.replacement, generator=g)
            indices = torch.tensor(indices)[weight_indices]

            iter_indices = iter(indices.tolist())
        return iter_indices

    def __len__(self):
        return self.num_samples


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_loss(inputs, targets, num_boxes, weights, alpha: float = 0.25, gamma: float = 2, query_mask=None, reduction=True):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
            
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none",weight=weights)
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if not reduction:
        return loss

    if query_mask is not None:
        loss = torch.stack([l[m].mean(0) for l, m in zip(loss, query_mask)])
        return loss.sum() / num_boxes
    return loss.mean(1).sum() / num_boxes


def nested_dict_to_namespace(dictionary):
    namespace = dictionary
    if isinstance(dictionary, dict):
        namespace = Namespace(**dictionary)
        for key, value in dictionary.items():
            setattr(namespace, key, nested_dict_to_namespace(value))
    return namespace

def nested_dict_to_device(dictionary, device):
    output = {}
    if isinstance(dictionary, dict):
        for key, value in dictionary.items():
            output[key] = nested_dict_to_device(value, device)
        return output
    return dictionary.to(device)

def threshold_indices(indices,targets,output_target,max_ind):
    '''
    indices: output from Hungarian matcher
    max_ind: the number of predictions in current batch
    return_swap_indices: if True, return the indices that indicate the cls/boxes need to be swapped

    This functions thresholds the indices outputted from the matcher.
    The swap_indices tells which boxes/class predictions need to be flipped.
    '''

    for (ind_out,ind_tgt),target in zip(indices,targets):

        for i in range(len(ind_out)):
          
            if ind_out[i] >= max_ind:
                ind_out[i] -= max_ind

                assert target[output_target]['boxes'][ind_tgt[i]][-1] > 0, 'Currently, this only swaps boxes where divisions have occurred. Object detection should only occur in the first box, not the second.'

                target[output_target]['boxes'][ind_tgt[i]] = torch.cat((target[output_target]['boxes'][ind_tgt[i],4:],target[output_target]['boxes'][ind_tgt[i],:4]),axis=-1)
                target[output_target]['labels'][ind_tgt[i]] = torch.cat((target[output_target]['labels'][ind_tgt[i],1:],target[output_target]['labels'][ind_tgt[i],:1]),axis=-1)

                if 'masks' in target[output_target]:
                    target[output_target]['masks'][ind_tgt[i]] = torch.cat((target[output_target]['masks'][ind_tgt[i],1:],target[output_target]['masks'][ind_tgt[i],:1]),axis=0)

    return indices, targets


def update_metrics_dict(metrics_dict:dict,acc_dict:dict,loss_dict:dict,weight_dict:dict,i,lr=None):
    '''
    After every iteration, the metrics dict is updated with the current loss and acc for that sample

    metrics_dict: dict
    Stores data for all epochs (metrics + loss)
    acc_dict: dict
    Stores acc info for current iteration
    loss_dict: dict
    Stores loss info for current iteration
    weight_dict: dict
    Stores weights for each loss
    i: int
    Iteration number
    '''
    
    metrics_keys = ['bbox_det_acc','mask_det_acc','overall_track_acc','divisions_track_acc','post_division_track_acc','new_cells_track_acc']

    if i == 0:
        for metrics_key in metrics_keys:
            if metrics_key in acc_dict.keys(): # add the accuracy info; these are two digits; first is # correct; second is total #
                metrics_dict[metrics_key] = acc_dict[metrics_key]
            else:
                metrics_dict[metrics_key] = np.ones((1,1,2)) * np.nan

        for weight_dict_key in weight_dict.keys(): # add the loss info which is a single number
            metrics_dict[weight_dict_key] = (loss_dict[weight_dict_key].detach().cpu().numpy()[None,None] * weight_dict[weight_dict_key]) if weight_dict_key in loss_dict else np.array(np.nan)[None,None]
        
        if lr is not None:
            metrics_dict['lr'] = lr
        
    else:
        for metrics_key in metrics_keys:
            if metrics_key in acc_dict.keys():
                metrics_dict[metrics_key] = np.concatenate((metrics_dict[metrics_key],acc_dict[metrics_key]),axis=1)
            else:
                metrics_dict[metrics_key] = np.concatenate((metrics_dict[metrics_key],np.ones((1,1,2)) * np.nan),axis=1)

        for weight_dict_key in weight_dict.keys():
            loss_dict_key_loss = (loss_dict[weight_dict_key].detach().cpu().numpy()[None,None] * weight_dict[weight_dict_key]) if weight_dict_key in loss_dict else np.array(np.nan)[None,None]
            metrics_dict[weight_dict_key] = np.concatenate((metrics_dict[weight_dict_key],loss_dict_key_loss),axis=1)

    assert metrics_dict['loss'].shape[0] == 1, 'Only one epoch worth of loss / metric info should be added'

    return metrics_dict

def display_loss(metrics_dict:dict,i,i_total,epoch,dataset):
    '''Print the loss
    
    metrics_dict:dict
    Contains loss / acc info over all epochs
    i: int
    Describes iteration #'''

    display_loss = {}

    for key in metrics_dict.keys():
        if 'loss' in key and not bool(re.search('\d',key)) and key != 'lr':
            display_loss[key] = f'{np.nan if np.isnan(metrics_dict[key][-1]).all() else np.nanmean(metrics_dict[key][-1]):.4f}'

    pad = int(math.log10(i_total))+1
    print(f'{dataset}  Epoch: {epoch} ({i:0{pad}}/{i_total-1})',display_loss)


def save_metrics_pkl(metrics_dict,output_dir,dataset,epoch):

    if not (output_dir / ('metrics_' + dataset + '.pkl')).exists() or epoch == 1:
        with open(output_dir / ('metrics_' + dataset + '.pkl'), 'wb') as f:
            pickle.dump(metrics_dict, f)
    else:
        with open(output_dir / ('metrics_' + dataset + '.pkl'), 'rb') as f:
            loaded_metrics_dict = pickle.load(f)

            assert loaded_metrics_dict['loss'].shape[0] == epoch - 1

            for metrics_dict_key in metrics_dict.keys():
                loaded_metrics_dict[metrics_dict_key] = np.concatenate((loaded_metrics_dict[metrics_dict_key],metrics_dict[metrics_dict_key]),axis=0)

            assert loaded_metrics_dict['loss'].shape[0] == epoch
        
        with open(output_dir / ('metrics_' + dataset + '.pkl'), 'wb') as f:
            pickle.dump(loaded_metrics_dict, f)



def calc_bbox_acc(outputs,targets,args):
    cls_thresh = args.cls_threshold
    iou_thresh = args.iou_threshold
    TP_bbox = TP_mask = 0
    FN = FP = FP_bbox = FP_mask = 0
    acc_dict = {}
    for t,target in enumerate(targets):
        indices = target['indices']
        pred_logits = outputs['pred_logits'].sigmoid().detach()[t]

        if target['empty']: # No objects in image so it should be all zero
            FP += int((pred_logits > cls_thresh).sum())
            continue

        FP += sum([1 for ind in range(pred_logits.shape[0]) if (ind not in indices[0] and pred_logits[ind,0] > cls_thresh)])

        pred_boxes = outputs['pred_boxes'].detach()[t]
        tgt_boxes = target['boxes'].detach()

        assert target['track_queries_mask'].sum() == 0, 'This function calculates detection accuracy only; not tracking'
        assert tgt_boxes[:,4:].sum() == 0, 'All boxes should not contain divisions since only object detection is being done here'

        if 'pred_masks' in outputs:
            pred_masks = outputs['pred_masks'].sigmoid().detach()[t]
            tgt_masks = target['masks'].detach()

        for ind_out, ind_tgt in zip(indices[0],indices[1]):
            if pred_logits[ind_out,0] > cls_thresh:
                iou = box_ops.generalized_box_iou(
                    box_ops.box_cxcywh_to_xyxy(pred_boxes[ind_out:ind_out+1,:4]),
                    box_ops.box_cxcywh_to_xyxy(tgt_boxes[ind_tgt:ind_tgt+1,:4]),
                    return_iou_only=True)

                if iou > iou_thresh:
                    TP_bbox += 1
                else:
                    FP_bbox += 1

                if 'pred_masks' in outputs:
                    pred_mask = pred_masks[ind_out:ind_out+1,:1]
                    pred_mask_scaled = F.interpolate(pred_mask, size=tgt_masks.shape[-2:], mode="bilinear", align_corners=False)
                    mask_iou = box_ops.mask_iou(pred_mask_scaled.flatten(1),tgt_masks[ind_tgt:ind_tgt+1,:1].flatten(1))

                    if mask_iou > iou_thresh:
                        TP_mask += 1
                    else:
                        FP_mask += 1
            else:
                FN += 1

    acc_dict['bbox_det_acc'] = np.array((TP_bbox,TP_bbox + FN + FP + FP_bbox),dtype=np.int32)[None,None]

    if 'pred_masks' in outputs:
        acc_dict['mask_det_acc'] = np.array((TP_mask,TP_mask + FN + FP + FP_mask ),dtype=np.int32)[None,None]

    return acc_dict

def calc_AP(outputs,targets,cls_thresh=0.5,iou_thresholds=[0.5]):

    AP_metric = BinaryAveragePrecision(thresholds=None)
    mAP = torch.zeros((len(targets)))

    for t,target in enumerate(targets):
        pred_all_logits = outputs['pred_logits'][t,:,0].detach().cpu().sigmoid()
        pred_ind = torch.where(pred_all_logits > cls_thresh)[0]

        pred_logits = pred_all_logits[pred_ind]
        pred_boxes = outputs['pred_boxes'][t,pred_ind,:4].detach().cpu()
        tgt_boxes = target['boxes'].detach().cpu()[:,:4]

        argmax = torch.argsort(pred_logits)
        pred_boxes_ord = pred_boxes[argmax]

        iou_thresholds = [0.5]
        AP = torch.zeros((len(iou_thresholds)))

        for i,iou_threshold in enumerate(iou_thresholds):
            labels = torch.zeros((len(pred_logits)))
            iou = torch.zeros((len(pred_logits)))
            
            iou_matrix = box_ops.generalized_box_iou(box_ops.box_cxcywh_to_xyxy(pred_boxes_ord),box_ops.box_cxcywh_to_xyxy(tgt_boxes),return_iou_only=True)

            for p in range(len(pred_boxes_ord)):

                if (iou_matrix[p] > iou_threshold).sum() > 0:
                    ind = torch.argmax(iou_matrix[p])
                    iou[p] = iou_matrix[p,ind]
                    iou_matrix = torch.cat((iou_matrix[:,:ind],iou_matrix[:,ind+1:]),axis=1)
                    labels[p] = 1

            if iou_matrix.shape[1] > 0:
                FP_indices = torch.where(labels == 0)[0]
                for idx in range(iou_matrix.shape[1]):
                    if idx < len(FP_indices):
                        labels[FP_indices[idx]] = 1
                    else:
                        labels = torch.cat((labels,torch.ones((1))))
                        iou = torch.cat((iou,torch.zeros((1))))

            AP[i] = AP_metric(iou,labels)

        mAP[t] = AP.mean()

    return mAP


def calc_track_acc(outputs,targets,args):
    cls_thresh = args.cls_threshold
    iou_thresh = args.iou_threshold
    TP = FP = FN = 0
    div_acc = np.zeros((2),dtype=np.int32)
    new_cells_acc = np.zeros((2),dtype=np.int32)
    track_acc_dict = {}
    for t,target in enumerate(targets):
        indices = target['indices']
        pred_logits = outputs['pred_logits'][t].sigmoid().detach()
        pred_boxes = outputs['pred_boxes'][t].detach()
        tgt_boxes = target['boxes'].detach()

        if target['empty']: # No objects to track
            FP += int((pred_logits[:,0] > cls_thresh).sum())
            continue

        # Coutning False Positives; cells leaving the frame + False Positives added to the frame
        pred_logits_FPs = pred_logits[~target['track_queries_TP_mask'] * target['track_queries_mask'],0]
        FP += int((pred_logits_FPs > cls_thresh).sum())

        # Calculate accuracy for new objects detected; FPs or TPs
        for query_id in range(pred_logits.shape[0]):
            if (~target['track_queries_mask'])[query_id]:
                if query_id in indices[0]:
                    if pred_logits[query_id,0] > cls_thresh:
                        ind_loc = torch.where(indices[0] == query_id)[0]
                        iou = box_ops.generalized_box_iou(
                                box_ops.box_cxcywh_to_xyxy(pred_boxes[query_id,:4][None]),
                                box_ops.box_cxcywh_to_xyxy(tgt_boxes[indices[1][ind_loc],:4]),
                                return_iou_only=True
                                )

                        if iou > iou_thresh:
                            TP += 1
                            new_cells_acc += 1
                        else:
                            FP += 1       
                            new_cells_acc[1] += 1  
                    else:
                        FN += 1
                        new_cells_acc[1] += 1
                else:
                    if pred_logits[query_id,0] > cls_thresh:
                        FP += 1

        pred_track_logits = pred_logits[target['track_queries_TP_mask']]
        pred_track_boxes = pred_boxes[target['track_queries_TP_mask']]
        box_matching = target['track_query_match_ids']

        for p,pred_logit in enumerate(pred_track_logits):
            if pred_logit[0] < cls_thresh:
                FN += 1

            else:
                iou = box_ops.generalized_box_iou(
                        box_ops.box_cxcywh_to_xyxy(pred_track_boxes[p:p+1,:4]),
                        box_ops.box_cxcywh_to_xyxy(tgt_boxes[box_matching[p],:4][None]),
                        return_iou_only=True
                    )
                
                if iou > iou_thresh:
                    TP += 1
                else:
                    FP += 1

            # Need to check for divisions
            if pred_logit[1] > cls_thresh and tgt_boxes[box_matching[p],-1] == 0: # Predicted FP division
                FP += 1  
                div_acc[1] += 1           
            elif pred_logit[1] < cls_thresh and tgt_boxes[box_matching[p],-1] > 0: # Predicted FN division
                FN += 1
                div_acc[1] += 1           
            elif pred_logit[1] > cls_thresh and tgt_boxes[box_matching[p],-1] > 0: # Correctly predictly TP division
                iou = box_ops.generalized_box_iou(
                        box_ops.box_cxcywh_to_xyxy(pred_track_boxes[p:p+1,4:]),
                        box_ops.box_cxcywh_to_xyxy(tgt_boxes[box_matching[p],4:][None]),
                        return_iou_only=True
                    )
                # Divided cells were not accounted above so we add one to correct & total column
                if iou > iou_thresh:
                    TP += 1
                    div_acc += 1
                else:
                    FP += 1
                    div_acc[1] += 1

            elif pred_logit[1] > cls_thresh and tgt_boxes[box_matching[p],-1] == 0: # Predicted TN correctly
                pass

    track_acc_dict['overall_track_acc'] = np.array((TP,TP + FN + FP),dtype=np.int32)[None,None]
    track_acc_dict['divisions_track_acc'] = div_acc[None,None]
    track_acc_dict['new_cells_track_acc'] = new_cells_acc[None,None]

    return track_acc_dict


def combine_div_boxes(box):

    new_box = torch.zeros_like(box)

    min_y = min(box[1] - box[3] / 2, box[5] - box[7] / 2)
    max_y = max(box[1] + box[3] / 2, box[5] + box[7] / 2)
    avg_y = (min_y + max_y) / 2
    new_box[1] = avg_y
    new_box[3] = max_y - min_y

    min_x = min(box[0] - box[2] / 2, box[4] - box[6] / 2)
    max_x = max(box[0] + box[2] / 2, box[4] + box[6] / 2)
    avg_x = (min_x + max_x) / 2
    new_box[0] = avg_x
    new_box[2] = max_x - min_x

    return new_box

def combine_div_masks(mask, prev_mask):

    combined_mask = torch.zeros_like(mask)
    combined_mask[0] += mask[0].clone()
    combined_mask[0] += mask[1].clone()

    div_cells_loc_y, div_cells_loc_x = torch.where(mask[0]+mask[1])

    prev_cell_loc_y, prev_cell_loc_x = torch.where(prev_mask[0])

    diff_loc_y, diff_loc_x = div_cells_loc_y.float().mean() - prev_cell_loc_y.float().mean(), div_cells_loc_x.float().mean() - prev_cell_loc_x.float().mean()

    prev_cell_loc = (torch.clamp(prev_cell_loc_y + int(diff_loc_y),0,mask.shape[1] - 1 ), torch.clamp(prev_cell_loc_x + int(diff_loc_x),0,mask.shape[2] - 1))

    combined_mask[0][prev_cell_loc] = 1
    
    return combined_mask

def divide_box(box,fut_box):

    new_box = torch.zeros_like(box)

    new_box[2:4] = fut_box[2:4]
    new_box[6:8] = fut_box[6:8]

    min_y = min(fut_box[1] - fut_box[3] / 2, fut_box[5] - fut_box[7] / 2)
    max_y = max(fut_box[1] + fut_box[3] / 2, fut_box[5] + fut_box[7] / 2)
    avg_y = (min_y + max_y) / 2
    dif_y = box[1] - avg_y

    min_x = min(fut_box[0] - fut_box[2] / 2, fut_box[4] - fut_box[6] / 2)
    max_x = max(fut_box[0] + fut_box[2] / 2, fut_box[4] + fut_box[6] / 2)
    avg_x = (min_x + max_x) / 2
    dif_x = box[0] - avg_x

    new_box[0::4] = fut_box[0::4] + dif_x
    new_box[1::4] = fut_box[1::4] + dif_y

    new_box_y_0 = torch.clamp((new_box[1::4] - new_box[3::4] / 2), box[1] - box[3] / 2, box[1] + box[3] / 2)
    new_box_x_0 = torch.clamp((new_box[0::4] - new_box[2::4] / 2), box[0] - box[2] / 2, box[0] + box[2] / 2)
    new_box_y_1 = torch.clamp((new_box[1::4] + new_box[3::4] / 2), box[1] - box[3] / 2, box[1] + box[3] / 2)
    new_box_x_1 = torch.clamp((new_box[0::4] + new_box[2::4] / 2), box[0] - box[2] / 2, box[0] + box[2] / 2)

    new_box[1::4] = (new_box_y_0 + new_box_y_1) / 2
    new_box[3::4] = new_box_y_1 - new_box_y_0
    new_box[0::4] = (new_box_x_0 + new_box_x_1) / 2
    new_box[2::4] = new_box_x_1 - new_box_x_0

    return new_box

def divide_mask(mask,fut_mask):

    div_mask = torch.zeros_like(mask)

    avg_loc_y, avg_loc_x = torch.where(mask[0])[0].float().mean(), torch.where(mask[0])[1].float().mean()

    fut_avg_loc_y, fut_avg_loc_x = torch.where(fut_mask[0] + fut_mask[1])[0].float().mean(), torch.where(fut_mask[0] + fut_mask[1])[1].float().mean()

    diff_loc_y, diff_loc_x = avg_loc_y - fut_avg_loc_y, avg_loc_x - fut_avg_loc_x

    fut_cell_1_loc = torch.where(fut_mask[0])
    fut_cell_2_loc = torch.where(fut_mask[1])

    fut_cell_1_loc = (torch.clamp(fut_cell_1_loc[0] + int(diff_loc_y),0,mask.shape[1] - 1 ), torch.clamp(fut_cell_1_loc[1] + int(diff_loc_x),0,mask.shape[2] - 1))
    fut_cell_2_loc = (torch.clamp(fut_cell_2_loc[0] + int(diff_loc_y),0,mask.shape[1] - 1), torch.clamp(fut_cell_2_loc[1] + int(diff_loc_x),0,mask.shape[2] - 1))

    div_mask[0][fut_cell_1_loc] = 1
    div_mask[1][fut_cell_2_loc] = 1

    return div_mask

def calc_iou(box_1,box_2):

    if (box_1[-1] == 0 and box_2[-1] == 0) or (box_1.shape[0] == 4 and box_2.shape[0] == 4):
        iou = box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(box_1[:4]),
            box_ops.box_cxcywh_to_xyxy(box_2[:4]),
            return_iou_only=True
        )

    elif box_1[-1] > 0 and box_2[-1] > 0:
        iou_1 = box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(box_1[:4]),
            box_ops.box_cxcywh_to_xyxy(box_2[:4]),
            return_iou_only=True
        )

        iou_2 = box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(box_1[4:]),
            box_ops.box_cxcywh_to_xyxy(box_2[4:]),
            return_iou_only=True
        )

        iou = (iou_1 + iou_2) / 2

        iou_1_flip = box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(box_1[:4]),
            box_ops.box_cxcywh_to_xyxy(box_2[4:]),
            return_iou_only=True
        )

        iou_2_flip = box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(box_1[4:]),
            box_ops.box_cxcywh_to_xyxy(box_2[:4]),
            return_iou_only=True
        )

        iou_flip = (iou_1_flip + iou_2_flip) / 2

        iou = max(iou,iou_flip)

    else:
        iou = 0

    return iou


def add_noise_to_boxes(boxes,l_1,l_2,clamp=True):
    noise = torch.rand_like(boxes) * 2 - 1
    boxes[..., :2] += boxes[..., 2:] * noise[..., :2] * l_1
    boxes[..., 2:] *= 1 + l_2 * noise[..., 2:]
    if clamp:
        boxes = torch.clamp(boxes,0,1)
    return boxes



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

def point_sample(input, point_coords, **kwargs):
    'Adapted from Detectron2'
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.
    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.
    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output



def update_early_or_late_track_divisions(
    outputs,
    targets,
    prev_target_name,
    cur_target_name,
    fut_target_name,
    ):

    device = outputs['pred_logits'].device
    use_masks = 'masks' in targets[0][cur_target_name]

    # height,width = targets[0][prev_target_name]['orig_size']

    # check for early / late cell division and adjust ground truths as necessary
    for t, target in enumerate(targets):

        man_track = target['man_track']

        prev_target = target[prev_target_name]
        cur_target = target[cur_target_name]
        fut_target = target[fut_target_name]

        if 'track_query_match_ids' in cur_target:

            # Get all prdictions for TP track queries
            pred_boxes_track = outputs['pred_boxes'][t][cur_target['track_queries_TP_mask']].detach()
            pred_logits_track = outputs['pred_logits'][t][cur_target['track_queries_TP_mask']].sigmoid().detach()
            # Check to see if there were any divisions in the future frame; if not, we skip to check for early division

            targets = man_track_ids(targets,prev_target_name,cur_target_name)

            boxes = cur_target['boxes'].clone()
            track_ids = cur_target['track_ids'].clone()

            for p, pred_box in enumerate(pred_boxes_track):
                box = boxes[cur_target['track_query_match_ids'][p]].clone()
                track_id = track_ids[cur_target['track_query_match_ids'][p]].clone()

                assert track_id in prev_target['track_ids']

                if prev_target['flexible_divisions'][prev_target['track_ids'] == track_id]:
                    continue

                # First check if the model predicted a single cell instead of a division
                if box[-1] > 0 and pred_logits_track[p,0] > 0.5 and pred_logits_track[p,-1] < 0.5: #division
                    
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

                        assert cur_target['labels_orig'][div_bool_1,1] == 1
                        cur_target['labels_orig'] = cur_target['labels_orig'][~div_bool_2]

                        if use_masks:
                            mask = cur_target['masks'][cur_target['track_query_match_ids'][p]]
                            prev_mask = prev_target['masks'][prev_target['track_ids'] == track_id][0]
                            combined_mask = combine_div_masks(mask,prev_mask)
                            cur_target['masks'][cur_target['track_query_match_ids'][p]] = combined_mask

                            cur_target['masks_orig'][div_bool_1] = combined_mask
                            cur_target['masks_orig'] = cur_target['masks_orig'][~div_bool_2]    

                        man_track[track_id-1,2] += 1                    
                        man_track[div_track_id_1-1,1] += 1                    
                        man_track[div_track_id_2-1,1] += 1                    

                        # Check to see if one of the daughters cells leave the FOV the frame after they are born
                        # If so, the mother cell will replace the other daugher cell since this is just tracking and division occured                        
                        if man_track[div_track_id_1-1,1] > man_track[div_track_id_1-1,2] or man_track[div_track_id_2-1,1] > man_track[div_track_id_2-1,2]:
                            man_track[track_id-1,2] = torch.max(man_track[div_track_id_1-1,2],man_track[div_track_id_2-1,2])
                            man_track[div_track_id_1-1,1:] = -1
                            man_track[div_track_id_2-1,1:] = -1

                            # Since cell division does not exist in future frames, we need to update the fut_track_id to the mother_id
                            if div_track_id_1 in fut_target['track_ids'] and div_track_id_2 in fut_target['track_ids']: 
                                raise NotImplementedError
                            elif div_track_id_1 in fut_target['track_ids']:
                                fut_target['track_ids'][fut_target['track_ids'] == div_track_id_1] = track_id.long().to(device)
                                if fut_target_name != 'fut_target' and div_track_id_1 in target['fut_target']['track_ids']:
                                    target['fut_target']['track_ids'][target['fut_target']['track_ids'] == div_track_id_1] = track_id.long().to(device)
                            elif div_track_id_2 in fut_target['track_ids']:
                                fut_target['track_ids'][fut_target['track_ids'] == div_track_id_2] = track_id.long().to(device)
                                if fut_target_name != 'fut_target' and div_track_id_2 in target['fut_target']['track_ids']:
                                    target['fut_target']['track_ids'][target['fut_target']['track_ids'] == div_track_id_2] = track_id.long().to(device)

                            fut_target['track_ids_orig'] = fut_target['track_ids'].clone()
                            if fut_target_name != 'fut_target':
                                target['fut_target']['track_ids_orig'] = target['fut_target']['track_ids'].clone()

                            if div_track_id_1 in man_track[:,-1] and div_track_id_2 in man_track[:,-1]:
                                # error in dataset here. Cell divides two frames in a row
                                fut_div_track_id_1, fut_div_track_id_2 = man_track[(man_track[:,-1] == div_track_id_1),0]
                                man_track[fut_div_track_id_1-1,-1] = 0
                                man_track[fut_div_track_id_2-1,-1] = 0  
                                fut_div_track_id_1, fut_div_track_id_2 = man_track[(man_track[:,-1] == div_track_id_2),0]
                                man_track[fut_div_track_id_1-1,-1] = 0
                                man_track[fut_div_track_id_2-1,-1] = 0                          
                            elif div_track_id_1 in man_track[:,-1]:
                                fut_div_track_id_1, fut_div_track_id_2 = man_track[(man_track[:,-1] == div_track_id_1),0]
                                man_track[fut_div_track_id_1-1,-1] = track_id
                                man_track[fut_div_track_id_2-1,-1] = track_id
                            elif div_track_id_2 in man_track[:,-1]:
                                fut_div_track_id_1, fut_div_track_id_2 = man_track[(man_track[:,-1] == div_track_id_2),0]
                                man_track[fut_div_track_id_1-1,-1] = track_id
                                man_track[fut_div_track_id_2-1,-1] = track_id

                            assert (torch.arange(1,target['man_track'].shape[0]+1,dtype=target['man_track'].dtype).to(target['man_track'].device) == target['man_track'][:,0]).all()

                # assert cur_target['track_query_match_ids'].max() < cur_target['boxes'].shape[0]
                 

            targets = man_track_ids(targets,cur_target_name,fut_target_name)
            assert (torch.arange(1,target['man_track'].shape[0]+1,dtype=target['man_track'].dtype).to(target['man_track'].device) == target['man_track'][:,0]).all()

            for p, pred_box in enumerate(pred_boxes_track):
                box = boxes[cur_target['track_query_match_ids'][p]].clone()

                if box[-1] == 0 and (pred_logits_track[p] > 0.5).all(): 
                    # if model predcitions division, check future frame and see if there is a division

                    track_id = track_ids[cur_target['track_query_match_ids'][p]].clone()

                    if man_track[track_id-1,2] == prev_target['framenb']:
                        track_id = man_track[track_id-1,-1]
                        
                    if track_id not in fut_target['track_ids']:
                        continue  # Cell leaves chamber in future frame

                    fut_box_ind = (fut_target['track_ids'] == track_id).nonzero()[0][0]
                    fut_box = fut_target['boxes'][fut_box_ind]
                    assert (torch.arange(1,target['man_track'].shape[0]+1,dtype=target['man_track'].dtype).to(target['man_track'].device) == target['man_track'][:,0]).all()


                    if fut_box[-1] > 0: # If cell divides next frame, we check to see if the model is predicting an early division
                        div_box = divide_box(box,fut_box)

                        iou_div = calc_iou(div_box,pred_box)
                        iou = calc_iou(box[:4],pred_box[:4])

                        if iou_div - iou > 0 and iou_div > 0.5:
                            cur_target['boxes'][cur_target['track_query_match_ids'][p]] = div_box
                            cur_target['labels'][cur_target['track_query_match_ids'][p]] = torch.tensor([0,0]).to(device)

                            fut_track_id_1, fut_track_id_2 = man_track[man_track[:,-1] == track_id,0]
                            fut_box_1 = fut_target['boxes_orig'][fut_target['track_ids_orig'] == fut_track_id_1][0]
                            fut_box_2 = fut_target['boxes_orig'][fut_target['track_ids_orig'] == fut_track_id_2][0]

                            if (div_box[:2] - fut_box_1[:2]).square().sum() +(div_box[4:6] - fut_box_2[:2]).square().sum() > (div_box[:2] - fut_box_2[:2]).square().sum() +(div_box[4:6] - fut_box_1[:2]).square().sum():
                                fut_track_id_1, fut_track_id_2 = fut_track_id_2, fut_track_id_1
                            assert (torch.arange(1,target['man_track'].shape[0]+1,dtype=target['man_track'].dtype).to(target['man_track'].device) == target['man_track'][:,0]).all()

                            ind_tgt_orig = torch.where(cur_target['boxes_orig'].eq(box).all(-1))[0][0]
                            cur_target['boxes_orig'][ind_tgt_orig,:4] = div_box[:4]
                            cur_target['boxes_orig'] = torch.cat((cur_target['boxes_orig'], torch.cat((div_box[4:],torch.zeros_like(div_box[4:])))[None]),axis=0)

                            cur_target['track_ids_orig'][ind_tgt_orig] = fut_track_id_1 
                            cur_target['track_ids_orig'] = torch.cat((cur_target['track_ids_orig'], torch.tensor([fut_track_id_2]).to(device)))

                            cur_target['labels_orig'] = torch.cat((cur_target['labels_orig'], cur_target['labels_orig'][:1]),axis=0)                   
                            cur_target['flexible_divisions_orig'] = torch.cat((cur_target['flexible_divisions_orig'], torch.tensor([False]).to(device)))                   

                            if use_masks:
                                mask = cur_target['masks'][cur_target['track_query_match_ids'][p]]
                                fut_mask = fut_target['masks'][fut_box_ind]
                                div_mask = divide_mask(mask,fut_mask)
                                cur_target['masks'][cur_target['track_query_match_ids'][p]] = div_mask
                                cur_target['masks_orig'][ind_tgt_orig,:1] = div_mask[:1]
                                cur_target['masks_orig'] = torch.cat((cur_target['masks_orig'], torch.cat((div_mask[1:],torch.zeros_like(div_mask[1:])))[None]),axis=0)
                            assert (torch.arange(1,target['man_track'].shape[0]+1,dtype=target['man_track'].dtype).to(target['man_track'].device) == target['man_track'][:,0]).all()

                            man_track[fut_track_id_1-1,1] -= 1
                            man_track[fut_track_id_2-1,1] -= 1
                            man_track[track_id-1,2] -= 1
                            assert (torch.arange(1,target['man_track'].shape[0]+1,dtype=target['man_track'].dtype).to(target['man_track'].device) == target['man_track'][:,0]).all()

                            if man_track[track_id-1,1] > man_track[track_id-1,2]:
                                man_track[track_id-1,1:] = -1
                                man_track[fut_track_id_1-1,-1] = 0
                                man_track[fut_track_id_2-1,-1] = 0
                            
                            assert (torch.arange(1,target['man_track'].shape[0]+1,dtype=target['man_track'].dtype).to(target['man_track'].device) == target['man_track'][:,0]).all()

    if 'track_query_match_ids' in cur_target: # This needs to be updated for aux outputs and enc outputs because the matcher is rerun then

        targets = man_track_ids(targets,prev_target_name,cur_target_name)

        prev_track_ids = prev_target['track_ids'][cur_target['prev_ind'][1]]

        # match track ids between frames
        target_ind_match_matrix = prev_track_ids.unsqueeze(dim=1).eq(cur_target['track_ids'])
        cur_target['target_ind_matching'] = target_ind_match_matrix.any(dim=1)
        cur_target['track_query_match_ids'] = target_ind_match_matrix.nonzero()[:, 1]

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

    assert (torch.arange(1,target['man_track'].shape[0]+1,dtype=target['man_track'].dtype).to(target['man_track'].device) == target['man_track'][:,0]).all()


    return targets

def update_object_detection(
    outputs,
    targets,
    indices,
    num_queries,
    prev_target_name,
    cur_target_name,
    fut_target_name,):

    N = outputs['pred_logits'].shape[1]
    use_masks = 'masks' in targets[0][cur_target_name]
    device = outputs['pred_logits'].device

    # Indicies are saved in targets for calcualting object detction / tracking accuracy
    for t,(target,(ind_out,ind_tgt)) in enumerate(zip(targets,indices)):

        prev_target = target[prev_target_name]
        cur_target = target[cur_target_name]
        fut_target = target[fut_target_name]

        man_track = target['man_track']
        framenb = cur_target['framenb']

        skip =[] # If a GT cell is split into two cells, we want to skip the second cell
        ind_keep = torch.tensor([True for _ in range(len(ind_tgt))]).bool()

        for ind_out_i, ind_tgt_i in zip(ind_out,ind_tgt):
            # Confirm prediction is an object query, not a track query
            if ind_out_i >= (N - num_queries) and ind_tgt_i not in skip:
                assert not cur_target['track_queries_mask'][ind_out_i] 
                track_id = cur_target['track_ids'][ind_tgt_i].clone()
                assert (man_track[:,-1] == track_id).sum() == 2 or (man_track[:,-1] == track_id).sum() == 0
                
                # Check if cell has just divided --> the two daugheter cells will be labeled cell 1 and 2
                if man_track[track_id-1,1] == framenb and man_track[track_id-1,-1] > 0:
                    
                    # Get id of mother cell by using the man_track
                    mother_id = man_track[track_id-1,-1].clone().long()
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
                    iou_sep = calc_iou(pred_boxes_1_2,boxes_1_2)

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
                    iou_combined = box_ops.generalized_box_iou(box_ops.box_cxcywh_to_xyxy(potential_pred_boxes[:,:4]),box_ops.box_cxcywh_to_xyxy(combined_box[None,:4]),return_iou_only=True)

                    # Get ind of which pred box that matched best to the combined box
                    max_ind = torch.argmax(iou_combined)
                    assert iou_combined[max_ind] <= 1 and iou_combined[max_ind] >= 0 and iou_sep <= 1 and iou_sep >= 0, 'Calc_iou not working; producings numbers outside of 0 and 1'

                    # We check to see if the separate pred boxes 1 & 2 have a higher iou than the pred box combined
                    if iou_combined[max_ind] - iou_sep > 0 and iou_combined[max_ind] > 0.5:
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

                        # if 'track_query_match_ids' in cur_target:
                        #     unmatch_id = cur_target['track_query_match_ids'][ind_tgt_2]

                        assert mother_id not in cur_target['track_ids_orig']
                        ind_tgt_orig_1 = cur_target['track_ids_orig'] == track_id_1
                        cur_target['track_ids_orig'][ind_tgt_orig_1] = mother_id
                        cur_target['boxes_orig'][ind_tgt_orig_1] = combined_box
                        cur_target['flexible_divisions_orig'][ind_tgt_orig_1] = True

                        ind_orig_keep = cur_target['track_ids_orig'] != track_id_2
                        
                        cur_target['track_ids_orig'] = cur_target['track_ids_orig'][ind_orig_keep]
                        cur_target['boxes_orig'] = cur_target['boxes_orig'][ind_orig_keep]
                        cur_target['labels_orig'] = cur_target['labels_orig'][ind_orig_keep]
                        cur_target['flexible_divisions_orig'] = cur_target['flexible_divisions_orig'][ind_orig_keep]

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
                            
                        man_track[mother_id-1,2] += 1
                        man_track[track_id_1-1,1] += 1
                        man_track[track_id_2-1,1] += 1

                        # Check to see if a division occurs in the futre frame. A cell could have left the FOV
                        if man_track[track_id_1-1,2] < man_track[track_id_1-1,1] or man_track[track_id_2-1,2] < man_track[track_id_2-1,1]:
                            man_track[mother_id-1,2] = torch.max(man_track[track_id_1-1,2],man_track[track_id_2-1,2])
                            man_track[track_id_1-1,1:] = -1
                            man_track[track_id_2-1,1:] = -1

                            # Since cell division does not exist in future frames, we need to update the fut_track_id to the mother_id
                            if track_id_1 in fut_target['track_ids_orig']:
                                fut_target['track_ids_orig'][fut_target['track_ids_orig'] == track_id_1] = mother_id
                            elif track_id_2 in fut_target['track_ids_orig']:
                                fut_target['track_ids_orig'][fut_target['track_ids_orig'] == track_id_2] = mother_id

                            if track_id_1 in man_track[:,-1] and track_id_2 in man_track[:,-1]:
                                # error in dataset here. Cell divides two frames in a row
                                div_track_id_1, div_track_id_2 = man_track[(man_track[:,-1] == track_id_1),0]
                                man_track[div_track_id_1-1,-1] = 0
                                man_track[div_track_id_2-1,-1] = 0
                                div_track_id_1, div_track_id_2 = man_track[(man_track[:,-1] == track_id_2),0]
                                man_track[div_track_id_1-1,-1] = 0
                                man_track[div_track_id_2-1,-1] = 0
                            elif track_id_1 in man_track[:,-1]:
                                div_track_id_1, div_track_id_2 = man_track[(man_track[:,-1] == track_id_1),0]
                                man_track[div_track_id_1-1,-1] = mother_id
                                man_track[div_track_id_2-1,-1] = mother_id
                            elif track_id_2 in man_track[:,-1]:
                                div_track_id_1, div_track_id_2 = man_track[(man_track[:,-1] == track_id_2),0]
                                man_track[div_track_id_1-1,-1] = mother_id
                                man_track[div_track_id_2-1,-1] = mother_id
  
                # Check if cell is about to divide
                elif man_track[track_id-1,2] == framenb and (man_track[:,-1] == track_id).sum() == 2:                              

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
                        potential_object_query_indices = [ind_out_id for ind_out_id in torch.arange(N-num_queries,N) if ((ind_out_id not in ind_out or ind_out_id == ind_out_i) and outputs['pred_logits'][t,ind_out_id,0].sigmoid().detach() > 0.5)]
            
                        if len(potential_object_query_indices) > 1:

                            # Get potential pred div boxes
                            potential_pred_boxes = outputs['pred_boxes'][t,potential_object_query_indices].detach()

                            # Calculate iou for all combinations of pred div cells and the simulated div cell
                            iou_div_all = box_ops.generalized_box_iou(box_ops.box_cxcywh_to_xyxy(potential_pred_boxes[:,:4]),box_ops.box_cxcywh_to_xyxy(torch.cat((div_box[None,:4],div_box[None,4:]),axis=0)),return_iou_only=True)

                            # Find best matching div cells
                            match_ind = torch.argmax(iou_div_all,axis=0).to('cpu')

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

                                    # if 'track_query_match_ids' in cur_target:
                                    #     cur_target['track_query_match_ids'] = torch.cat((cur_target['track_query_match_ids'],torch.tensor([len(cur_target['boxes'])-1]).to(device)))

                                    ind_keep = torch.cat((ind_keep,torch.tensor([True])))

                                    ind_tgt_orig_i = torch.where(cur_target['boxes_orig'].eq(box).all(-1))[0][0]

                                    cur_target['boxes_orig'][ind_tgt_orig_i] = torch.cat((div_box[:4],torch.zeros_like(div_box[:4])))
                                    cur_target['boxes_orig'] = torch.cat((cur_target['boxes_orig'],torch.cat((div_box[4:],torch.zeros_like(div_box[:4])))[None]))

                                    cur_target['labels_orig'] = torch.cat((cur_target['labels_orig'],torch.tensor([0,1])[None,].to(device)))
                                    cur_target['track_ids_orig'][ind_tgt_orig_i] = fut_track_id_1
                                    cur_target['track_ids_orig'] = torch.cat((cur_target['track_ids_orig'],torch.tensor([fut_track_id_2]).to(device)))
                                    cur_target['flexible_divisions_orig'][ind_tgt_orig_i] = True
                                    cur_target['flexible_divisions_orig'] = torch.cat((cur_target['flexible_divisions_orig'],torch.tensor([True]).to(device)))

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

                                    ind_out[ind_out == ind_out_i] = torch.tensor([potential_object_query_indices[match_ind[0]]])
                                    ind_out = torch.cat((ind_out,torch.tensor([potential_object_query_indices[match_ind[1]]])))
                                    ind_tgt = torch.cat((ind_tgt,torch.tensor([cur_target['boxes'].shape[0]-1])))                    

                                    assert len(ind_out) == len(ind_tgt)
                                    assert len(cur_target['boxes']) == len(cur_target['labels'])

                                    man_track[track_id-1,2] -= 1
                                    man_track[fut_track_id_1-1,1] -= 1
                                    man_track[fut_track_id_2-1,1] -= 1

                                    if man_track[track_id-1,1] > man_track[track_id-1,2]:
                                        man_track[track_id-1,1:] = -1
                                        man_track[fut_track_id_1-1,-1] = 0
                                        man_track[fut_track_id_2-1,-1] = 0

        if False:
            assert (torch.arange(1,target['man_track'].shape[0]+1,dtype=target['man_track'].dtype) == target['man_track'][:,0]).all()

        cur_target['boxes'] = cur_target['boxes'][ind_tgt[ind_keep].sort()[0]]
        cur_target['labels'] = cur_target['labels'][ind_tgt[ind_keep].sort()[0]]
        cur_target['track_ids'] = cur_target['track_ids'][ind_tgt[ind_keep].sort()[0]]
        cur_target['flexible_divisions'] = cur_target['flexible_divisions'][ind_tgt[ind_keep].sort()[0]]

        if use_masks:
            cur_target['masks'] = cur_target['masks'][ind_tgt[ind_keep].sort()[0]]

        if 'track_query_match_ids' in cur_target: # This needs to be updated for aux outputs and enc outputs because the matcher is rerun then
            prev_track_ids = prev_target['track_ids'][cur_target['prev_ind'][1]]

            # match track ids between frames
            target_ind_match_matrix = prev_track_ids.unsqueeze(dim=1).eq(cur_target['track_ids'])
            cur_target['target_ind_matching'] = target_ind_match_matrix.any(dim=1)
            cur_target['track_query_match_ids'] = target_ind_match_matrix.nonzero()[:, 1]

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

    return targets, indices

def man_track_ids(targets,input_target_name:str,output_target_name:str = None):

    target_names = ['prev_prev_target','prev_target','cur_target','fut_target']

    features = ['track_ids','boxes','labels','flexible_divisions']
    if 'masks' in targets[0][output_target_name]:
        features += ['masks']

    for target in targets:
        for target_name in target_names:
            target_reset = target[target_name]
            for feature in features:
                target_reset[feature] = target_reset[feature + '_orig'].clone()
       
        input_target = target[input_target_name]
        output_target = target[output_target_name]
        
        framenb = output_target['framenb']
        prev_track_ids = input_target['track_ids']

        if 'prev_ind' in output_target:
            prev_track_ids = prev_track_ids[output_target['prev_ind'][1]]

        if 'target_ind_matching' in output_target:
            if output_target['num_FPs'] == 0:
                target_ind_matching = output_target['target_ind_matching']
            else:
                target_ind_matching = output_target['target_ind_matching'][:-output_target['num_FPs']]

        man_track = target['man_track'].clone()
        man_track = man_track[(man_track[:,1] <= framenb) * (man_track[:,2] >= framenb)]

        cell_divisions = man_track[:,-1]

        for idx,prev_track_id in enumerate(prev_track_ids):
            if 'target_ind_matching' in output_target and not target_ind_matching[idx]:
                continue # This is necesary for when FN are added; the model is forced to detect the cell instead of tracking it
            if prev_track_id not in output_target['track_ids_orig']: # If cell does not track to next frame
                if prev_track_id in cell_divisions: # check if cell divided

                    div_cur_track_ids = man_track[man_track[:,-1] == prev_track_id,0]

                    assert len(div_cur_track_ids) == 2

                    div_ind_1 = output_target['track_ids'] == div_cur_track_ids[0]
                    div_ind_2 = output_target['track_ids'] == div_cur_track_ids[1]

                    output_target['track_ids'][div_ind_1] = prev_track_id
                    remove_ind = output_target['track_ids'] != div_cur_track_ids[1]         

                    for feature in features:
                        if feature not in ['flexible_divisions','track_ids']:
                            feature_len = output_target[feature].shape[1]
                            output_target[feature][div_ind_1,feature_len//2:] = output_target[feature][div_ind_2,:feature_len//2]
                        output_target[feature] = output_target[feature][remove_ind]

    return targets
            
def split_outputs(outputs,indices,new_outputs=None,update_masks=False):
    
    if new_outputs is None:
        new_outputs = outputs
    new_outputs['pred_logits'] = outputs['pred_logits'][:,indices[0]:indices[1]]
    new_outputs['pred_boxes'] = outputs['pred_boxes'][:,indices[0]:indices[1]]

    if 'pred_masks' in outputs:
        new_outputs['pred_masks'] = outputs['pred_masks'][:,indices[0]:indices[1]]

    if 'aux_outputs' in outputs:
        for lid in range(len(outputs['aux_outputs'])):
            new_outputs['aux_outputs'][lid]['pred_logits'] = outputs['aux_outputs'][lid]['pred_logits'][:,indices[0]:indices[1]]
            new_outputs['aux_outputs'][lid]['pred_boxes'] = outputs['aux_outputs'][lid]['pred_boxes'][:,indices[0]:indices[1]]

            if 'pred_masks' in outputs['aux_outputs'][lid]:
                new_outputs['aux_outputs'][lid]['pred_masks'] = outputs['aux_outputs'][lid]['pred_masks'][:,indices[0]:indices[1]]

    return new_outputs
