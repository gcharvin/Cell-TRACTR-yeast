# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MOT dataset with tracking training augmentations.
"""
import bisect
import csv
import os
import random
from pathlib import Path
import re
import numpy as np
import torch
import time

from torch.utils.data import Sampler

from . import transforms as T
from .coco import CocoDetection, make_coco_transforms_cells
from ..util import misc

class MOT(CocoDetection):

    def __init__(self, *args, **kwargs):
        super(MOT, self).__init__(*args, **kwargs,)

        man_track_paths = list((self.root.parents[1] / 'man_track').glob('*.txt'))

        img_fps = list((self.root).glob('*.tif'))
        dataset_ctcs = list(set([int(re.findall('\d+',img_fp.stem)[0]) for img_fp in img_fps]))

        man_track_paths = [man_track_path for man_track_path in man_track_paths if int(man_track_path.stem) in dataset_ctcs]

        self.current_epoch = 0 # For inference on the training set, we just set this to zero

    @property
    def sequences(self):
        return self.coco.dataset['sequences']

    @property
    def frame_range(self):
        if 'frame_range' in self.coco.dataset:
            return self.coco.dataset['frame_range']
        else:
            return {'start': 0, 'end': 1.0}

    def seq_length(self, idx):
        return self.coco.imgs[idx]['seq_length']

    def sample_weight(self, idx):
        return 1.0 / self.seq_length(idx)

    def set_epoch(self, epoch):
        """Set the current epoch number."""
        self.current_epoch = epoch

    def set_dataset_type(self,dataset):
        self.dataset_type = dataset

    def set_tracking(self,tracking, shift):
        self.tracking = tracking
        self.shift = tracking and shift

    def set_target_size(self,target_size):
        self.target_size = torch.tensor(list(map(int,re.findall('\d+',target_size))))
        self.RandomCrop = T.RandomCrop(self.target_size)

    def __getitem__(self, idx):

        # Setting a different seed for each sample
        # Important to note that data augmentations will get applied randomly to sequential frames
        # If flipping samples horizontally / vertically are added, will need to make sure randomness is same for all
        random.seed(idx + self.current_epoch)
        torch.manual_seed(idx + self.current_epoch)
        torch.cuda.manual_seed(idx + self.current_epoch)
        np.random.seed(idx + self.current_epoch)

        target = {'main': {}}
        target['main']['training_method'] = 'main'
        main_target = target['main']

        fn = self.coco.imgs[idx]['file_name']
        dataset_nb = re.findall('\d+',fn)[:-1]

        if self.args.masks and self.args.freeze_detr:
            pass
            # skip samples that contain no cells; loss can't back prop with nothing to calculate loss on
        if self.tracking:
            idx = self.get_idx(idx,dataset_nb)

            man_track_id = self.coco.imgs[idx]['man_track_id']
            man_track = self.get_man_track(man_track_id)
            main_target['man_track'] = torch.from_numpy(man_track).long()

        target['dataset_nb'] = torch.tensor(int(dataset_nb[0]))
        fn = self.coco.imgs[idx]['file_name']
        framenb = int(re.findall('\d+',fn)[-1])

        self.RandomCrop.region = None
    
        if self.tracking:
            target, main_target = self.get_prev_and_fut_targets(target, main_target, idx, framenb)

        img, cur_target = self._getitem_from_id(idx, framenb)
        target['cur_image'] = img
        main_target['cur_target'] = cur_target

        if self.tracking and self.crop:
            main_target = misc.update_cropped_man_track(main_target)

        # This saves the original annotations. Used for display purposes only if args.display_all is True. It gets too cluttered when there are too many cells. Therefore, I'd only use this for moma
        if self.dataset_type == 'moma' and self.tracking:
            target = self.save_copy_of_target(target,main_target)

        return img, target
    
    def get_idx(self,idx,dataset_nb):
            
        if idx < 2:
            idx = 2

        prev_prev_fn = self.coco.imgs[idx-2]['file_name']
        prev_prev_dataset_nb = re.findall('\d+',prev_prev_fn)[:-1]
        
        while prev_prev_dataset_nb != dataset_nb:
            idx += 1
            prev_prev_fn = self.coco.imgs[idx-2]['file_name']
            prev_prev_dataset_nb = re.findall('\d+',prev_prev_fn)[:-1]
            
        if idx == len(self.coco.imgs) -1:
            idx -= 1

        # If flexible divisions isn't being used, we don't need the future frame
        if self.flex_div:
            fut_fn = self.coco.imgs[idx+1]['file_name']
            fut_dataset_nb = re.findall('\d+',fut_fn)[:-1]

            if fut_dataset_nb != dataset_nb:
                idx -= 1

        return idx

    def get_man_track(self,man_track_id):

        man_track = np.loadtxt(self.root.parents[1] / 'man_track' / self.root.parts[-2] / (str(man_track_id) + '.txt'),dtype=np.int16)

        # We remove cells that disappear and reappear
        divisions = np.unique(man_track[:,-1])
        divisions = divisions[divisions != 0]
        for div in divisions:
            if (man_track[:,-1] == div).sum() != 2:
                man_track[man_track[:,-1] == div,-1] = 0

        return man_track
    
    def get_prev_and_fut_targets(self,target, main_target, idx, framenb):

        prev_prev_img, prev_prev_target = self._getitem_from_id(idx-2, framenb-2)
        target['prev_prev_image'] = prev_prev_img
        main_target['prev_prev_target'] = prev_prev_target

        if self.RandomCrop.region is not None and self.shift:
            rand_num = random.random()
            self.shift_value /= 4
            if rand_num < 0.5:        
                self.ShiftCrop(rand_num)
            self.shift_value *= 4

        prev_img, prev_target = self._getitem_from_id(idx-1, framenb-1)
        target['prev_image'] = prev_img
        main_target['prev_target'] = prev_target

        if self.RandomCrop.region is not None and self.shift:
            rand_num = random.random()
            if self.num_cells < 10:
                rand_num /= 2
                self.ShiftCrop(rand_num)
            elif rand_num < 0.5:              
                self.ShiftCrop(rand_num)

        if self.flex_div:
            fut_img, fut_target = self._getitem_from_id(idx+1, framenb+1)
            target['fut_image'] = fut_img
            main_target['fut_target'] = fut_target


        return target, main_target
    
    def save_copy_of_target(self, target, main_target):
        target['target_og'] = {}
        target['target_og']['man_track'] = main_target['man_track'].clone()
            
        for target_name in ['prev_prev_target','prev_target','cur_target','fut_target']:
            if not self.flex_div and target_name == 'fut_target':
                continue
            target['target_og'][target_name] = {}
            target['target_og'][target_name]['boxes'] = main_target[target_name]['boxes'].clone()
            target['target_og'][target_name]['track_ids'] = main_target[target_name]['track_ids'].clone()
            target['target_og'][target_name]['is_touching_edge'] = main_target[target_name]['is_touching_edge'].clone()
            target['target_og'][target_name]['framenb'] = main_target[target_name]['framenb'].clone()
            
            if 'masks' in main_target[target_name]:
                target['target_og'][target_name]['masks'] = main_target[target_name]['masks'].clone()

        return target

    def ShiftCrop(self, rand_num):

        h,w = self.RandomCrop.get_h_w()             

        if self.img_w_raw > w:
            shift_x = int(random.random() * self.shift_value)
        else:
            shift_x = 0

        if self.img_h_raw > h:
            shift_y = int(random.random() * self.shift_value)
        else:
            shift_y = 0

        if shift_x != 0 or shift_y != 0:
                
            if rand_num < 0.25:
                if self.RandomCrop.region[0] + self.RandomCrop.region[2] + shift_y < self.img_h:
                    self.RandomCrop.region[0] += shift_y
                else:
                    self.RandomCrop.region[0] = self.img_h - self.target_size[0].item()
            else:
                if self.RandomCrop.region[0] - shift_y > 0:
                    self.RandomCrop.region[0] -= shift_y
                else:
                    self.RandomCrop.region[0] = 0

            if rand_num > 0.125 and rand_num < 0.375:
                if self.RandomCrop.region[1] + self.RandomCrop.region[3] + shift_x < self.img_w:
                    self.RandomCrop.region[1] += shift_x
                else:
                    self.RandomCrop.region[1] = self.img_w - self.target_size[1].item()
            else:
                if self.RandomCrop.region[1] - shift_x > 0:
                    self.RandomCrop.region[1] -= shift_x
                else:
                    self.RandomCrop.region[1] = 0

        assert self.RandomCrop.region[0] >= 0 and self.RandomCrop.region[1] >= 0
        assert self.RandomCrop.region[0] + self.RandomCrop.region[2] <= self.img_h  and self.RandomCrop.region[1] + self.RandomCrop.region[3] <= self.img_w

def build_cells(image_set,args):

    root = Path(args.data_dir) / args.dataset / 'COCO'

    assert root.exists()

    transforms, norm_transforms = make_coco_transforms_cells(image_set)

    shift = args.shift

    if args.dataset == 'val':
        shift = False

    dataset = MOT(
        root / image_set / 'img',
        root / 'annotations'/ image_set / ('anno.json'),
        transforms,
        norm_transforms,
        return_masks=args.masks,
        overflow_boxes=args.overflow_boxes,
        remove_no_obj_imgs=False,
        crop=args.crop,
        batch_size=args.batch_size,
        args=args,
        )  
    
    dataset.set_dataset_type(args.dataset)
    dataset.set_target_size(args.target_size)
    dataset.set_tracking(args.tracking, args.shift)
    dataset.flex_div = args.flex_div
    
    return dataset

class SubsetSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)