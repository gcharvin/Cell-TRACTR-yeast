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

from . import transforms as T
from .coco import CocoDetection, make_coco_transforms_cells

class MOT(CocoDetection):

    def __init__(self, *args, prev_frame_range=1, **kwargs):
        super(MOT, self).__init__(*args, **kwargs,)

        self._prev_frame_range = prev_frame_range

        self.man_tracks = {}

        self.load_all_txt_files = False

        if self.load_all_txt_files:
            man_track_paths = (self.root.parents[1] / 'man_track').glob('*.txt')
            for man_track_path in man_track_paths:
                man_track = np.loadtxt(man_track_path,dtype=np.int16)
                self.man_tracks[man_track_path.stem] = man_track

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

    def set_target_size(self,target_size):
        self.target_size = torch.tensor(list(map(int,re.findall('\d+',target_size))))
        self.RandomCrop = T.RandomCrop(self.target_size)

    def __getitem__(self, idx):
        # random.seed(random.randint(0,100))

        # curr_random_state = {
        #     'random': random.getstate(),
        #     'torch': torch.random.get_rng_state(),
        #     'numpy': np.random.get_state()}

        # Setting a different seed for each sample
        # Important to note that data augmentations will get applied randomly to sequential frames
        # If flipping samples horizontally / vertically are added, will need to make sure randomness is same for all
        random.seed(idx + self.current_epoch)
        torch.manual_seed(idx + self.current_epoch)
        torch.cuda.manual_seed(idx + self.current_epoch)
        np.random.seed(idx + self.current_epoch)

        fn = self.coco.imgs[idx]['file_name']
        dataset_nb = re.findall('\d+',fn)[:-1]

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

        fut_fn = self.coco.imgs[idx+1]['file_name']
        fut_dataset_nb = re.findall('\d+',fut_fn)[:-1]

        if fut_dataset_nb != dataset_nb:
            idx -= 1

        fn = self.coco.imgs[idx]['file_name']
        framenb = int(re.findall('\d+',fn)[-1])

        if self.load_all_txt_files:
            man_track = self.man_tracks[dataset_nb].copy()
        else:
            if len(dataset_nb) > 1:
                man_track = np.loadtxt(self.root.parents[1] / 'man_track' / (f'{dataset_nb[0]}_{dataset_nb[1]}.txt'),dtype=np.int16)
                if man_track.ndim == 1:
                    man_track = man_track[None]
            else:
                man_track = np.loadtxt(self.root.parents[1] / 'man_track' / (dataset_nb[0] + '.txt'),dtype=np.int16)

        # We remove cells that disappear and reappear
        divisions = np.unique(man_track[:,-1])
        divisions = divisions[divisions != 0]
        for div in divisions:
            if (man_track[:,-1] == div).sum() == 1:
                man_track[man_track[:,-1] == div,-1] = 0

        target = {'man_track': torch.from_numpy(man_track).long()}

        if len(dataset_nb) > 1:
            target['dataset_nb'] = torch.tensor(int(dataset_nb[0]))
            target['split_nb'] = torch.tensor(int(dataset_nb[1]))
        else:
            target['dataset_nb'] = torch.tensor(int(dataset_nb[0]))

        self.RandomCrop.region = None

        img, cur_target = self._getitem_from_id(idx, framenb)
        target['cur_image'] = img
        target['cur_target'] = cur_target
        
        prev_img, prev_target = self._getitem_from_id(idx-1, framenb-1)
        target['prev_image'] = prev_img
        target['prev_target'] = prev_target
    
        prev_prev_img, prev_prev_target = self._getitem_from_id(idx-2, framenb-2)
        target['prev_prev_image'] = prev_prev_img
        target['prev_prev_target'] = prev_prev_target

        fut_img, fut_target = self._getitem_from_id(idx+1, framenb+1)
        target['fut_image'] = fut_img
        target['fut_target'] = fut_target

        # random.setstate(curr_random_state['random'])
        # torch.random.set_rng_state(curr_random_state['torch'])
        # np.random.set_state(curr_random_state['numpy'])

        target['target_og'] = {'man_track': torch.from_numpy(man_track)}
            
        for target_name in ['prev_prev_target','prev_target','cur_target','fut_target']:
            target['target_og'][target_name] = {}
            target['target_og'][target_name]['boxes'] = target[target_name]['boxes'].clone()
            target['target_og'][target_name]['track_ids'] = target[target_name]['track_ids'].clone()
            
            if 'masks' in target[target_name]:
                target['target_og'][target_name]['masks'] = target[target_name]['masks'].clone()

        return img, target

    def write_result_files(self, results, output_dir):
        """Write the detections in the format for the MOT17Det sumbission

        Each file contains these lines:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>

        """

        files = {}
        for image_id, res in results.items():
            img = self.coco.loadImgs(image_id)[0]
            file_name_without_ext = os.path.splitext(img['file_name'])[0]
            seq_name, frame = file_name_without_ext.split('_')
            frame = int(frame)

            outfile = os.path.join(output_dir, f"{seq_name}.txt")

            # check if out in keys and create empty list if not
            if outfile not in files.keys():
                files[outfile] = []

            for box, score in zip(res['boxes'], res['scores']):
                if score <= 0.7:
                    continue
                x1 = box[0].item()
                y1 = box[1].item()
                x2 = box[2].item()
                y2 = box[3].item()
                files[outfile].append(
                    [frame, -1, x1, y1, x2 - x1, y2 - y1, score.item(), -1, -1, -1])

        for k, v in files.items():
            with open(k, "w") as of:
                writer = csv.writer(of, delimiter=',')
                for d in v:
                    writer.writerow(d)


class WeightedConcatDataset(torch.utils.data.ConcatDataset):

    def sample_weight(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        if hasattr(self.datasets[dataset_idx], 'sample_weight'):
            return self.datasets[dataset_idx].sample_weight(sample_idx)
        else:
            return 1 / len(self.datasets[dataset_idx])

def build_cells(image_set,args):

    root = Path(args.data_dir) / args.dataset

    assert root.exists()

    transforms, norm_transforms = make_coco_transforms_cells(image_set)

    if args.no_data_aug:
        transforms = None

    dataset = MOT(
        root / image_set / 'img',
        root / 'annotations'/ image_set / ('anno.json'),
        transforms,
        norm_transforms,
        return_masks=args.masks,
        overflow_boxes=args.overflow_boxes,
        remove_no_obj_imgs=False,
        )  
    
    dataset.set_dataset_type(args.dataset)
    dataset.set_target_size(args.target_size)
    
    return dataset
