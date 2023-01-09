# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MOT dataset with tracking training augmentations.
"""
import bisect
import copy
import csv
from email.mime import image
import os
import random
from pathlib import Path
import re
import math
import cv2 
import numpy as np
import torch

from . import transforms as T
from .coco import CocoDetection, make_coco_transforms_cells
# from .coco import build as build_coco
from .crowdhuman import build_crowdhuman


class MOT(CocoDetection):

    def __init__(self, *args, prev_frame_range=1, **kwargs):
        super(MOT, self).__init__(*args, **kwargs)

        self._prev_frame_range = prev_frame_range

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

    def __getitem__(self, idx):
        random.seed(random.randint(0,100))
        random_state = {
            'random': random.getstate(),
            'torch': torch.random.get_rng_state()}

        fn = self.coco.imgs[idx]['file_name']
        # idx = idx if re.findall('\D+',fn)[-1] == '_cur.png' else idx + 1

        img, target = self._getitem_from_id(idx, random_state=random_state)
        target['image'] = img
        target['image_id'] = torch.tensor(idx)
        # target['fn'] = fn
        
        # if self._prev_frame:
        #     prev_img, prev_target = self._getitem_from_id(idx-1, random_state=random_state)

        #     prev_target['filenb'] = torch.tensor(idx-1)
        #     prev_target['image'] = prev_img

        #     framenb = list(map(int,re.findall('\d+',fn)))[-1]
        #     prev_target['framenb'] = torch.tensor(framenb)

        #     target[f'prev_image'] = prev_img
        #     target[f'prev_target'] = prev_target
    

        #     if self._prev_prev_frame:
        #         framenb = list(map(int,re.findall('\d+',fn)))[-1]
        #         pad = re.findall('\d+',fn)[-1].count('0') + int(math.log10(framenb))+1
        #         fn = fn.replace(re.findall('\d+',fn)[-1],f'{framenb-1:0{str(pad)}d}')

        #         if os.path.exists(self.img_folder / fn):
        #             prev_prev_img , prev_prev_target = self._getitem_from_id(idx-3, random_state=random_state)
        #             target[f'prev_prev_image'] = prev_prev_img
        #             target[f'prev_prev_target'] = prev_prev_target

        #             prev_cur_img , prev_cur_target = self._getitem_from_id(idx-2, random_state)

        #             prev_cur_target['filenb'] = torch.tensor(idx-2)
        #             prev_cur_target['framenb'] = torch.tensor(framenb)

        #             # target[f'prev_cur_image'] = prev_cur_img
        #             target[f'prev_cur_target'] = prev_cur_target

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

    root = Path(args.output_dir.parents[1] / 'data' / 'cells' / 'new_dataset')

    assert root.exists()

    split = image_set

    transforms, norm_transforms = make_coco_transforms_cells(image_set)

    if args.evaluate_dataset_with_no_data_aug:
        transforms = None

    datasets = []
    img_folders = ['prev_prev_img', 'prev_cur_img', 'prev_img', 'cur_img', 'fut_prev_img', 'fut_img']
    json_files = ['prev_prev', 'prev_cur', 'prev', 'cur', 'fut_prev', 'fut']

    for img_folder,json_file in zip(img_folders,json_files):
         datasets.append(MOT
                (
                root / split / img_folder,
                root / 'annotations'/ split / (json_file+'.json'),
                transforms,
                norm_transforms,
                return_masks=args.masks,
                overflow_boxes=args.overflow_boxes,
                remove_no_obj_imgs=False,
                )
            )
    
    
    return datasets