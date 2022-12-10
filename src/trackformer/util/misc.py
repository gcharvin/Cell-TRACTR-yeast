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
from torch import Tensor
from visdom import Visdom
from pathlib import Path

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

def plot_results(outputs,prev_outputs,targets,samples,savepath,filename,train,meta_data=None):
    
    # show_class_label = True
    filename = Path(filename)
    height = samples.tensors.shape[-2]
    width = samples.tensors.shape[-1]
    bs = samples.tensors.shape[0]
    spacer = 10
    threshold = 0.5
    folder = 'train_outputs' if train else 'eval_outputs'
    num_img = 5
    blank = np.zeros((height, (width*num_img + spacer)*bs,3))
    fontscale = 0.4
    alpha = 0.3
    pred_masks = 'pred_masks' in outputs
    box_converter = box_cxcy_to_xyxy(height,width)

    if meta_data is not None:
        meta_data_keys = list(meta_data.keys())

        for meta_data_key in meta_data_keys:

            outputs_TM = meta_data[meta_data_key]['outputs']
            targets_TM = [target[meta_data_key] for target in targets]

            if meta_data_key == 'dn_track':
                dn_track = np.zeros((height,(width*6 + spacer)*bs,3))
                for i in range(bs):
                    colors = [tuple((255*np.random.random(3))) for _ in range(50)]

                    img = samples.tensors[i].permute(1,2,0)                
                    
                    boxes = targets_TM[i]['track_query_boxes_gt'].detach().cpu().numpy()
                    noised_boxoes = targets_TM[i]['track_query_boxes'].detach().cpu().numpy()
                
                    boxes = box_converter.convert(boxes)
                    noised_boxes = box_converter.convert(noised_boxoes)

                    img = img.detach().cpu().numpy().copy()
                    img = np.repeat(np.mean(img,axis=-1)[:,:,np.newaxis],3,axis=-1)
                    img = (255*(img - np.min(img)) / np.ptp(img)).astype(np.uint8)
                    img_gt = img.copy()
                    img_noised_gt = img.copy()
                    img_pred = img.copy()
                    img_all_pred = img.copy()

                    TP_mask = ~targets_TM[i]['track_queries_fal_pos_mask']

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

                        if pred_logits[idx,1] > threshold:
                            bounding_box = bbox[4:]
                            img_all_pred = cv2.rectangle(
                                img_all_pred,
                                (int(np.clip(bounding_box[0],0,width)), int(np.clip(bounding_box[1],0,height))),
                                (int(np.clip(bounding_box[0] + bounding_box[2],0,width)), int(np.clip(bounding_box[1] + bounding_box[3],0,height))),
                                color=colors[idx] if TP_mask[idx] else (0,0,0),
                                thickness = 1)  

                            img_all_pred = cv2.putText(
                                img_all_pred,
                                text = f'{pred_logits[idx,1]:.2f}', 
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

                    for idx,bbox in enumerate(pred_boxes): 
                        bounding_box = bbox[:4]
                        img_pred = cv2.rectangle(
                            img_pred,
                            (int(np.clip(bounding_box[0],0,width)), int(np.clip(bounding_box[1],0,height))),
                            (int(np.clip(bounding_box[0] + bounding_box[2],0,width)), int(np.clip(bounding_box[1] + bounding_box[3],0,height))),
                            color=colors[where[idx]] if TP_mask[where[idx]] else (0,0,0),
                            thickness = 1)            

                        img_pred = cv2.putText(
                            img_pred,
                            text = f'{pred_logits[idx,0]:.2f}', 
                            org=(int(bounding_box[0]) - 5, int(bounding_box[1] + bounding_box[3] // 4 + int(fontscale*30))), 
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                            fontScale = fontscale,
                            color = colors[where[idx]] if TP_mask[where[idx]] else (128,128,128),
                            thickness=1,
                        )

                        if pred_logits[idx,1] > threshold:
                            bounding_box = bbox[4:]
                            img_pred = cv2.rectangle(
                                img_pred,
                                (int(np.clip(bounding_box[0],0,width)), int(np.clip(bounding_box[1],0,height))),
                                (int(np.clip(bounding_box[0] + bounding_box[2],0,width)), int(np.clip(bounding_box[1] + bounding_box[3],0,height))),
                                color=colors[where[idx]] if TP_mask[where[idx]] else (0,0,0),
                                thickness = 1)  

                            img_pred = cv2.putText(
                                img_pred,
                                text = f'{pred_logits[idx,1]:.2f}', 
                                org=(int(bounding_box[0]) - 5, int(bounding_box[1] + bounding_box[3] // 4 + int(fontscale*30))), 
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                                fontScale = fontscale,
                                color = colors[where[idx]] if TP_mask[where[idx]] else (128,128,128),
                                thickness=1,
                            )

                            
                    dn_track[:,(width*6+spacer)*i + width*0:(width*6+spacer)*i + width*1] = previmg
                    dn_track[:,(width*6+spacer)*i + width*1:(width*6+spacer)*i + width*2] = previmg_noised
                    dn_track[:,(width*6+spacer)*i + width*2:(width*6+spacer)*i + width*3] = img_gt
                    dn_track[:,(width*6+spacer)*i + width*3:(width*6+spacer)*i + width*4] = img_noised_gt
                    dn_track[:,(width*6+spacer)*i + width*4:(width*6+spacer)*i + width*5] = img_pred
                    dn_track[:,(width*6+spacer)*i + width*5:(width*6+spacer)*i + width*6] = img_all_pred

                cv2.imwrite(str(savepath / folder / (filename.stem + '_dn_track' + filename.suffix)),dn_track)

            if meta_data_key == 'dn_object':
                dn_object = np.zeros((height,(width*5 + spacer)*bs,3))
                for i in range(bs):
                    colors = [tuple((255*np.random.random(3))) for _ in range(50)]

                    img = samples.tensors[i].permute(1,2,0)                
                    
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

                cv2.imwrite(str(savepath / folder / (filename.stem + '_dn_object' + filename.suffix)),dn_object)


    for i in range(bs):

        colors = [tuple((255*np.random.random(3))) for _ in range(50)]

        previmg = targets[i]['prev_image'].permute(1,2,0)
        prev_bbs = prev_outputs['pred_boxes'][i].detach().cpu().numpy()
        prev_classes = prev_outputs['pred_logits'][i].sigmoid().detach().cpu().numpy()

        prev_keep = prev_classes[:,0] > threshold
        prev_bbs = prev_bbs[prev_keep,]
        prev_classes = prev_classes[prev_keep]

        if sum(prev_keep) > 0:
    
            prev_bbs[:,1::2] = prev_bbs[:,1::2] * height
            prev_bbs[:,::2] = prev_bbs[:,::2] * width
            prev_bbs[:,0] = prev_bbs[:,0] - prev_bbs[:,2] // 2
            prev_bbs[:,1] = prev_bbs[:,1] - prev_bbs[:,3] // 2
            prev_bbs[:,4] = prev_bbs[:,4] - prev_bbs[:,6] // 2
            prev_bbs[:,5] = prev_bbs[:,5] - prev_bbs[:,7] // 2

        previmg = previmg.detach().cpu().numpy().copy()
        previmg = np.repeat(np.mean(previmg,axis=-1)[:,:,np.newaxis],3,axis=-1)
        previmg = (255*(previmg - np.min(previmg)) / np.ptp(previmg)).astype(np.uint8)
        previmg_copy = previmg.copy()

        for idx,bbox in enumerate(prev_bbs):
            bounding_box = bbox[:4]
            previmg = cv2.rectangle(
                previmg,
                (int(np.clip(bounding_box[0],0,width)), int(np.clip(bounding_box[1],0,height))),
                (int(np.clip(bounding_box[0] + bounding_box[2],0,width)), int(np.clip(bounding_box[1] + bounding_box[3],0,height))),
                color=(0,0,0),#colors[idx],
                thickness = 1)

            previmg = cv2.putText(
                previmg,
                text = f'{prev_classes[idx,0]:.2f}', 
                org=(int(bounding_box[0]) - 5, int(bounding_box[1] + bounding_box[3] // 4 + int(fontscale*30))), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale = 0.4,
                color = (128,128,128),#colors[idx],
                thickness=1,
            )

            if prev_classes[idx,1] > 0.5:
                bounding_box = bbox[4:]
                previmg = cv2.rectangle(
                    previmg,
                    (int(np.clip(bounding_box[0],0,width)), int(np.clip(bounding_box[1],0,height))),
                    (int(np.clip(bounding_box[0] + bounding_box[2],0,width)), int(np.clip(bounding_box[1] + bounding_box[3],0,height))),
                    color = (128,128,128),
                    thickness = 1)

                previmg = cv2.putText(
                    previmg,
                    text = f'{prev_classes[idx,1]:.2f}', 
                    org=(int(bounding_box[0]) - 5, int(bounding_box[1] + bounding_box[3] // 4 + int(fontscale*30))), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale = fontscale,
                    color = (0,0,0),
                    thickness=1,
                )

        track_query_boxes = targets[i]['track_query_boxes'].detach().cpu()

        if track_query_boxes.shape[0] > 0:
    
            track_query_boxes[:,1::2] = track_query_boxes[:,1::2] * height
            track_query_boxes[:,::2] = track_query_boxes[:,::2] * width
            track_query_boxes[:,0] = track_query_boxes[:,0] - track_query_boxes[:,2] // 2
            track_query_boxes[:,1] = track_query_boxes[:,1] - track_query_boxes[:,3] // 2

        previmg_track = previmg_copy.copy()

        for idx,bounding_box in enumerate(track_query_boxes):
            previmg_track = cv2.rectangle(
                previmg_track,
                (int(np.clip(bounding_box[0],0,width)), int(np.clip(bounding_box[1],0,height))),
                (int(np.clip(bounding_box[0] + bounding_box[2],0,width)), int(np.clip(bounding_box[1] + bounding_box[3],0,height))),
                color=colors[idx],
                thickness = 1)

        img = samples.tensors[i].permute(1,2,0)

        bbs = outputs['pred_boxes'][i].detach().cpu().numpy()
        classes = outputs['pred_logits'][i].sigmoid().detach().cpu().numpy()

        keep = classes[:,0] > threshold
        bbs = bbs[keep]
        classes = classes[keep]

        if pred_masks:
            masks = outputs['pred_masks'][i].sigmoid().detach().cpu().numpy()[keep]
            if sum(keep) > 0:
                masks = np.concatenate((masks[:,0],masks[:,1]),axis=0)
                masks_filt = np.zeros((masks.shape))
                argmax = np.argmax(masks,axis=0)
                for m in range(masks.shape[0]):
                    masks_filt[m,argmax==m] = masks[m,argmax==m]
                masks_filt = np.stack((masks_filt[:masks.shape[0]//2],masks_filt[masks.shape[0]//2:]),axis=1)
                masks_filt[masks_filt > threshold] = 255
                masks = masks_filt.astype(np.uint8)

        track_keep = keep[:track_query_boxes.shape[0]]
        colors = [colors[idx] for idx in range(len(track_keep)) if track_keep[idx]] 

        if sum(keep) > 0:

            bbs[:,1::2] = bbs[:,1::2] * height
            bbs[:,::2] = bbs[:,::2] * width
            bbs[:,0] = bbs[:,0] - bbs[:,2] // 2
            bbs[:,1] = bbs[:,1] - bbs[:,3] // 2
            bbs[:,4] = bbs[:,4] - bbs[:,6] // 2
            bbs[:,5] = bbs[:,5] - bbs[:,7] // 2

        img = img.detach().cpu().numpy().copy()
        img = np.repeat(np.mean(img,axis=-1)[:,:,np.newaxis],3,axis=-1)
        img = (255*(img - np.min(img)) / np.ptp(img)).astype(np.uint8)
        img_copy = img.copy()

        for idx, bbox in enumerate(bbs):
            bounding_box = bbox[:4]
            img = cv2.rectangle(
                img,
                (int(np.clip(bounding_box[0],0,width)), int(np.clip(bounding_box[1],0,height))),
                (int(np.clip(bounding_box[0] + bounding_box[2],0,width)), int(np.clip(bounding_box[1] + bounding_box[3],0,height))),
                color=colors[idx] if idx < len(colors) else (0,0,0),
                thickness = 1)

            img = cv2.putText(
                img,
                text = f'{classes[idx,0]:.2f}', 
                org=(int(bounding_box[0]) - 5, int(bounding_box[1] + bounding_box[3] // 4 + int(fontscale*30))), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale = fontscale,
                color = colors[idx] if idx < len(colors) else (128,128,128),
                thickness=1,
            )

            if pred_masks:
                mask = cv2.resize(masks[idx,0],(width,height)) 
                mask = np.repeat(mask[...,None],3,axis=-1)
                mask[mask[...,0]>0] = colors[idx] if idx < len(colors) else (128,128,128)
                img[mask>0] = img[mask>0]*(1-alpha) + mask[mask>0]*(alpha)


            if classes[idx,1] > threshold:
                bounding_box = bbox[4:]
                img = cv2.rectangle(
                    img,
                    (int(np.clip(bounding_box[0],0,width)), int(np.clip(bounding_box[1],0,height))),
                    (int(np.clip(bounding_box[0] + bounding_box[2],0,width)), int(np.clip(bounding_box[1] + bounding_box[3],0,height))),
                    color=colors[idx] if idx < len(colors) else (0,0,0),
                    thickness = 1)

                img = cv2.putText(
                    img,
                    text = f'{classes[idx,1]:.2f}', 
                    org=(int(bounding_box[0]) - 5, int(bounding_box[1] + bounding_box[3] // 4 + int(fontscale*30))), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale = fontscale,
                    color = colors[idx] if idx < len(colors) else (128,128,128),
                    thickness=1,
                )

                if pred_masks:
                    mask = cv2.resize(masks[idx,1],(width,height))
                    mask = np.repeat(mask[...,None],3,axis=-1)
                    mask[mask[...,0]>0] = colors[idx] if idx < len(colors) else (128,128,128)
                    img[mask>0] = img[mask>0]*(1-alpha) + mask[mask>0]*(alpha)
                    
        blank[:,(width*num_img+spacer)*i + width*0:(width*num_img+spacer)*i + width*1] = previmg_copy
        blank[:,(width*num_img+spacer)*i + width*1:(width*num_img+spacer)*i + width*2] = previmg
        blank[:,(width*num_img+spacer)*i + width*2:(width*num_img+spacer)*i + width*3] = previmg_track
        blank[:,(width*num_img+spacer)*i + width*3:(width*num_img+spacer)*i + width*4] = img
        blank[:,(width*num_img+spacer)*i + width*4:(width*num_img+spacer)*i + width*5] = img_copy

    cv2.imwrite(str(savepath / folder / filename),blank)


def plot_tracking_results(img,bbs,masks,colors,cells,div_track=None,new_cells=None,track=True):

    img = np.array(img)
    img_copy = img.copy()
    height = img.shape[0]
    width = img.shape[1]
    blank = np.zeros((height, 2*width,3))
    fontscale = 0.4
    mask_threshold = 0.5
    alpha = 0.4

    bbs = bbs.detach().cpu().numpy()

    if masks is not None:
        masks = masks.detach().cpu().numpy()

        masks_filt = np.zeros((masks.shape))
        argmax = np.argmax(masks,axis=0)
        for m in range(masks.shape[0]):
            masks_filt[m,argmax==m] = masks[m,argmax==m]
        masks_filt = masks_filt > mask_threshold
        masks = (masks_filt*255).astype(np.uint8)

    write_cellnb = False

    if bbs is not None:
        bbs[:,1::2] = bbs[:,1::2] * height
        bbs[:,::2] = bbs[:,::2] * width
        bbs[:,0] = bbs[:,0] - bbs[:,2] // 2
        bbs[:,1] = bbs[:,1] - bbs[:,3] // 2

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

        if write_cellnb:
            img = cv2.putText(
                img,
                text = str(cells[idx]), 
                org=(int(bounding_box[0]) - 5, int(bounding_box[1] + bounding_box[3] // 4 + int(fontscale*30))), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale = fontscale,
                color = colors[idx],
                thickness=1,
            )

    if track and div_track is not None and sum(div_track) > 0:
        div_track_nbs = np.unique(div_track)
        div_track_nbs = [d for d in div_track_nbs if d != 0] 
        for div_track_nb in div_track_nbs:
            div_loc = np.where(div_track == div_track_nb)[0]
            cell_1 = bbs[div_loc[0]]
            cell_2 = bbs[div_loc[1]]

            img = cv2.arrowedLine(
                img,
                (int(cell_1[0] + cell_1[2] // 2), int(cell_1[1] + cell_1[3] // 2)),
                (int(cell_2[0] + cell_2[2] // 2), int(cell_2[1] + cell_2[3] // 2)),
                color=(1, 1, 1),
                thickness=1,
            )


    blank[:,:width] = img_copy
    blank[:,width:width*2] = img

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

def threshold_indices(indices,max_ind):
    '''
    indices: output from Hungarian matcher
    max_ind: the number of predictions in current batch
    return_swap_indices: if True, return the indices that indicate the cls/boxes need to be swapped

    This functions thresholds the indices outputted from the matcher.
    The swap_indices tells which boxes/class predictions need to be flipped.
    '''

    swap_indices = [[],[]]

    for idx,(ind_out,ind_tgt) in enumerate(indices):

        for i in range(len(ind_out)):
          
            if ind_out[i] >= max_ind:
                ind_out[i] -= max_ind
                swap_indices[idx].append(ind_out[i])

        swap_indices[idx] = torch.tensor(swap_indices[idx])

    return indices, swap_indices


def update_metrics_dict(metrics_dict:dict,acc_dict:dict,loss_dict:dict,weight_dict:dict,i):
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

    if i == 0:
        for metrics_key in acc_dict.keys(): # add the accuracy info; these are two digits; first is # correct; second is total #
            metrics_dict[metrics_key] = acc_dict[metrics_key]

        for loss_dict_key in loss_dict.keys(): # add the loss info which is a single number
            metrics_dict[loss_dict_key] = (loss_dict[loss_dict_key].detach().cpu().numpy()[None,None] * weight_dict[loss_dict_key]) if loss_dict_key in loss_dict else np.array(np.nan)[None,None]

        for weight_dict_key in weight_dict.keys(): # If there are empty chambers in both samples, we need to manually add nan to standarize it
            if weight_dict_key not in loss_dict and (('mask' not in weight_dict_key) or ('mask' in weight_dict_key and not bool(re.search('\d',weight_dict_key)))):
                metrics_dict[weight_dict_key] = np.array(np.nan)[None,None]
        
    else:
        for metrics_key in acc_dict.keys():
            metrics_dict[metrics_key] = np.concatenate((metrics_dict[metrics_key],acc_dict[metrics_key]),axis=1)

        for loss_dict_key in loss_dict.keys():
            loss_dict_key_loss = (loss_dict[loss_dict_key].detach().cpu().numpy()[None,None] * weight_dict[loss_dict_key]) if loss_dict_key in loss_dict else np.array(np.nan)[None,None]
            metrics_dict[loss_dict_key] = np.concatenate((metrics_dict[loss_dict_key],loss_dict_key_loss),axis=1)

        for weight_dict_key in weight_dict.keys(): # If there are empty chambers in both samples, we need to manually add nan to standarize it
            if weight_dict_key not in loss_dict and (('mask' not in weight_dict_key) or ('mask' in weight_dict_key and not bool(re.search('\d',weight_dict_key)))):
                metrics_dict[weight_dict_key] = np.concatenate((metrics_dict[weight_dict_key],np.array(np.nan)[None,None]),axis=1)

    return metrics_dict

def display_loss(metrics_dict:dict,i,i_total,epoch,dataset):
    '''Print the loss
    
    metrics_dict:dict
    Contains loss / acc info over all epochs
    i: int
    Describes iteration #'''

    display_loss = {}

    for key in metrics_dict.keys():
        if 'loss' in key and not bool(re.search('\d',key)):
            display_loss[key] = f'{np.nanmean(metrics_dict[key][-1]):.4f}'

    pad = int(math.log10(i_total))+1
    print(f'{dataset}  Epoch: {epoch} ({i:0{pad}}/{i_total})',display_loss)


def save_metrics_pkl(metrics_dict,output_dir,dataset):

    if not (output_dir / ('metrics_' + dataset + '.pkl')).exists():
        with open(output_dir / ('metrics_' + dataset + '.pkl'), 'wb') as f:
            pickle.dump(metrics_dict, f)
    else:
        with open(output_dir / ('metrics_' + dataset + '.pkl'), 'rb') as f:
            loaded_metrics_dict = pickle.load(f)

            for metrics_dict_key in metrics_dict.keys():
                loaded_metrics_dict[metrics_dict_key] = np.concatenate((loaded_metrics_dict[metrics_dict_key],metrics_dict[metrics_dict_key]),axis=0)
        
        with open(output_dir / ('metrics_' + dataset + '.pkl'), 'wb') as f:
            pickle.dump(loaded_metrics_dict, f)



def calc_bbox_acc(outputs,targets, indices, cls_thresh = 0.5, iou_thresh = 0.75):
    acc = torch.zeros((2),dtype=torch.int32)
    for t,target in enumerate(targets):
        bboxes = target['boxes'].detach().cpu()
        pred_logits = outputs['pred_logits'].sigmoid().detach().cpu()[t]
        pred_boxes = outputs['pred_boxes'].detach().cpu()[t]

        object_detect_only = True if sum(target['track_queries_mask']) == 0 else False

        # We are only interestd in calcualting object detection accuracy; not tracking accuracy
        ind_out,ind_tgt = indices[t]
        num_track = target['track_queries_mask'].sum() # count the total number of track queries
        keep = ind_out > (num_track-1) 
            
        bboxes = bboxes[ind_tgt[keep]]
        pred_logits = pred_logits[ind_out[keep]]
        pred_boxes = pred_boxes[ind_out[keep]]

        if target['empty']: # No objects in image so it should be all zero
            acc[1] += (pred_logits > cls_thresh).sum()
            continue
        div_keep = bboxes[:,-1] > 0
        bboxes = torch.cat((bboxes[:,:4],bboxes[div_keep,4:]))
        num_bboxes = bboxes.shape[0]
        acc[1] += num_bboxes

        for p in range(pred_logits.shape[0]):
            if pred_logits[p,0] > cls_thresh:
                match = False
                for b,bbox in enumerate(bboxes): # For each prediction in pred_logits, check if it matches with a target bbox
                    iou = box_ops.generalized_box_iou(
                        box_ops.box_cxcywh_to_xyxy(pred_boxes[p:p+1,:4]),
                        box_ops.box_cxcywh_to_xyxy(bbox[None]),
                        return_iou_only=True
                    )
                    if iou > iou_thresh:
                        acc[0] += 1
                        bboxes = torch.cat((bboxes[:b],bboxes[b+1:]))
                        match = True
                        break
                # bboxes are removed at they are matched; is pred_logit > cls_thresh but does not match to a bbox then it is incorrect
                if not match:
                    acc[1] += 1 # Add 1 incorrect for every time it predicts something wrong

            if pred_logits[p,1] > cls_thresh:
                match = False
                for b,bbox in enumerate(bboxes):
                    iou = box_ops.generalized_box_iou(
                        box_ops.box_cxcywh_to_xyxy(pred_boxes[p:p+1,4:]),
                        box_ops.box_cxcywh_to_xyxy(bbox[None]),
                        return_iou_only=True
                    )
                    if iou > iou_thresh:
                        acc[0] += 1
                        bboxes = torch.cat((bboxes[:b],bboxes[b+1:]))
                        match = True
                        break

                if not match:
                    acc[1] += 1 # Add 1 incorrect for every time it predicts something wrong

    acc = acc[None,None]

    if object_detect_only:
        return acc, torch.zeros_like(acc)
    else:
        return torch.zeros_like(acc), acc

def calc_track_acc(outputs,targets,indices,cls_thresh=0.5,iou_thresh=0.75):
    acc = torch.zeros((2),dtype=torch.int32)
    div_acc = torch.zeros((2),dtype=torch.int32)
    track_div_cell_acc = torch.zeros((2),dtype=torch.int32)
    for t,target in enumerate(targets):
        pred_logits = outputs['pred_logits'][t][target['track_queries_TP_mask']].sigmoid().detach().cpu()
        pred_boxes = outputs['pred_boxes'][t][target['track_queries_TP_mask']].detach().cpu()
        track_div_cell = target['track_div_mask'][target['track_queries_TP_mask']]
        if target['empty']: # No objects in image so it should be all zero
            acc[1] += (pred_logits > cls_thresh).sum()
            continue

        box_matching = target['track_query_match_ids']
        bboxes = target['boxes'].detach().cpu()

        acc[1] += target['track_queries_TP_mask'].sum()

        for p,pred_logit in enumerate(pred_logits):
            if pred_logit[0] < cls_thresh:
                acc[1] += 1
                if track_div_cell[p]:
                    track_div_cell_acc[1] += 1

            else:
                iou = box_ops.generalized_box_iou(
                        box_ops.box_cxcywh_to_xyxy(pred_boxes[p:p+1,:4]),
                        box_ops.box_cxcywh_to_xyxy(bboxes[box_matching[p],:4][None]),
                        return_iou_only=True
                    )
                
                if iou > iou_thresh:
                    acc[0] += 1
                    if track_div_cell[p]:
                        track_div_cell_acc += 1
                else:
                    if track_div_cell[p]:
                        track_div_cell_acc[1] += 1

            # Need to check for divisions
            if pred_logit[1] > cls_thresh and bboxes[box_matching[p],-1] == 0: # Predicted FP division
                acc[1] += 1
                if track_div_cell[p]:
                    track_div_cell_acc[1] += 1   
                div_acc[1] += 1           
            elif pred_logit[1] < cls_thresh and bboxes[box_matching[p],-1] > 0: # Predicted FN division
                acc[1] += 1
                if track_div_cell[p]:
                    track_div_cell_acc[1] += 1
                div_acc[1] += 1           
            elif pred_logit[1] > cls_thresh and bboxes[box_matching[p],-1] > 0: # Correctly predictly TP division
                iou = box_ops.generalized_box_iou(
                        box_ops.box_cxcywh_to_xyxy(pred_boxes[p:p+1,4:]),
                        box_ops.box_cxcywh_to_xyxy(bboxes[box_matching[p],4:][None]),
                        return_iou_only=True
                    )
                # Divided cells were not accounted above so we add one to correct & total column
                if iou > iou_thresh:
                    acc += 1
                    if track_div_cell[p]:
                        track_div_cell_acc += 1
                    div_acc += 1
                else:
                    if track_div_cell[p]:
                        track_div_cell_acc[1] += 1
                    div_acc[1] += 1

            elif pred_logit[1] > cls_thresh and bboxes[box_matching[p],-1] == 0: # Predicted TN correctly
                if track_div_cell[p]:
                    track_div_cell_acc += 1

        # Random FPs get added to track queries which will always have no target to track to
        FPs = outputs['pred_logits'][t][~target['track_queries_TP_mask'] * target['track_queries_mask']].sigmoid().detach().cpu()
        acc[1] += (FPs > cls_thresh).sum()

    return acc[None,None], div_acc[None,None], track_div_cell_acc[None,None]