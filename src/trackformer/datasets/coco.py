# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
import copy
import random
from pathlib import Path
from collections import Counter

import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

from . import transforms as T


class CocoDetection(torchvision.datasets.CocoDetection):

    fields = ["labels", "area", "iscrowd", "boxes", "track_ids", "masks"]

    def __init__(self,  img_folder, ann_file, transforms, norm_transforms,
                 return_masks=False, overflow_boxes=False, remove_no_obj_imgs=True,
                 min_num_objects=0):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self._norm_transforms = norm_transforms
        self.prepare = ConvertCocoPolysToMask(return_masks, overflow_boxes)
        self.img_folder = img_folder
        self.ann_file = ann_file

        annos_image_ids = [
            ann['image_id'] for ann in self.coco.loadAnns(self.coco.getAnnIds())]
            
        if remove_no_obj_imgs:
            self.ids = sorted(list(set(annos_image_ids)))

        if min_num_objects:
            counter = Counter(annos_image_ids)
            self.ids = [i for i in self.ids if counter[i] >= min_num_objects]

    def _getitem_from_id(self, image_id, man_track, framenb, random_state=None):
        # if random state is given we do the data augmentation with the state
        # and then apply the random jitter. this ensures that (simulated) adjacent
        # frames have independent jitter.
        if random_state is not None:
            curr_random_state = {
                'random': random.getstate(),
                'torch': torch.random.get_rng_state()}
            random.setstate(random_state['random'])
            torch.random.set_rng_state(random_state['torch'])

        img, annotations = super(CocoDetection, self).__getitem__(image_id)
        target = {
            'image_id': torch.tensor(self.ids[image_id]),
            'annotations': annotations,
            'framenb': torch.tensor(framenb),
            }

        img, target = self.prepare(img, target)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        # ignore
        ignore = target.pop("ignore").bool()
        for field in self.fields:
            if field in target:
                target[f"{field}_ignore"] = target[field][ignore]
                target[field] = target[field][~ignore]

        if random_state is not None:
            random.setstate(curr_random_state['random'])
            torch.random.set_rng_state(curr_random_state['torch'])

        img, target = self._norm_transforms(img, target)

        return img, target

    def write_result_files(self, *args):
        pass


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = torch.zeros((len(segmentations), 2, height, width), dtype=torch.uint8)
    for pidx, polygons in enumerate(segmentations):
        if isinstance(polygons, dict):
            rles = {'size': polygons['size'],
                    'counts': polygons['counts'].encode(encoding='UTF-8')}
        else:
            for polygon in polygons:
                if len(polygon) == 0:
                    raise Exception('Data error; every segmentation should have one polygon')
                rles = coco_mask.frPyObjects(polygon, height, width)

        masks[pidx,0] = torch.from_numpy(coco_mask.decode(rles)[:,:,0])
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False, overflow_boxes=False):
        self.return_masks = return_masks
        self.overflow_boxes = overflow_boxes

    def __call__(self, image, target):
        w, h = image.size

        anno = target.pop("annotations")
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        empty = anno[0]['empty']

        boxes = torch.as_tensor([obj["bbox"] for obj in anno], dtype=torch.float32)

        # x,y,w,h --> x,y,x,y
        boxes[:, 2:4] += boxes[:, :2]
        boxes = torch.cat((boxes,torch.zeros_like(boxes)),axis=1)

        if not self.overflow_boxes:
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)

        if not empty:
            assert ((boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])).all() == True, 'Boxes are not correct in coco.py script'

        target["boxes"] = boxes
        target["boxes_orig"] = boxes.clone()

        if self.return_masks:
            segmentations = [[obj["segmentation"]] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)
            target["masks"] = masks
            target["masks_orig"] = masks.clone()

        # for conversion to coco api
        track_ids = torch.tensor([obj["track_id"] for obj in anno])
        classes = torch.tensor([obj["category_id"] for obj in anno]) 
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        ignore = torch.tensor([obj["ignore"] if "ignore" in obj else 0 for obj in anno])

        classes = torch.stack((classes,2 * torch.ones_like(classes)),axis=1)

        target['track_ids'] = track_ids
        target['track_ids_orig'] = track_ids.clone()
        target['flexible_divisions'] = torch.zeros_like(track_ids).bool()
        target['flexible_divisions_orig'] = torch.zeros_like(track_ids).bool()
        target["labels"] = classes - 1
        target["labels_orig"] = (classes - 1).clone()
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["ignore"] = ignore

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        target["empty"] = torch.tensor(empty)

        return image, target


def make_coco_transforms_cells(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if image_set == 'train':
        transforms = [
            T.RandomGaussianBlur(),
            T.RandomGaussianNoise(),
            T.RandomIlluminationVoodoo(),
            
        ]
    elif image_set == 'val':
        transforms = [
            T.RandomGaussianBlur(),
            T.RandomGaussianNoise(),
            T.RandomIlluminationVoodoo(),
        ]
    else:
        ValueError(f'unknown {image_set}')

    return T.Compose(transforms), normalize