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

    def _getitem_from_id(self, image_id, random_state=None):
        # if random state is given we do the data augmentation with the state
        # and then apply the random jitter. this ensures that (simulated) adjacent
        # frames have independent jitter.
        if random_state is not None:
            curr_random_state = {
                'random': random.getstate(),
                'torch': torch.random.get_rng_state()}
            random.setstate(random_state['random'])
            torch.random.set_rng_state(random_state['torch'])

        img, target = super(CocoDetection, self).__getitem__(image_id)
        image_id = self.ids[image_id]
        target = {'image_id': image_id,
                  'annotations': target}
        img, target = self.prepare(img, target)

        if 'track_ids' not in target:
            target['track_ids'] = torch.arange(len(target['labels']))

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

    # def __getitem__(self, idx):
    #     random_state = {
    #         'random': random.getstate(),
    #         'torch': torch.random.get_rng_state()}
    #     img, target = self._getitem_from_id(idx, random_state, random_jitter=False)

    #     if self._prev_frame:
    #         # PREV
    #         prev_img, prev_target = self._getitem_from_id(idx, random_state)
    #         target[f'prev_image'] = prev_img
    #         target[f'prev_target'] = prev_target

    #         if self._prev_prev_frame:
    #             # PREV PREV
    #             prev_prev_img, prev_prev_target = self._getitem_from_id(idx, random_state)
    #             target[f'prev_prev_image'] = prev_prev_img
    #             target[f'prev_prev_target'] = prev_prev_target

    #     return img, target

    def write_result_files(self, *args):
        pass


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        if isinstance(polygons, dict):
            rles = {'size': polygons['size'],
                    'counts': polygons['counts'].encode(encoding='UTF-8')}
        else:
            rles_list = []
            for polygon in polygons:
                if len(polygon) > 0:
                    rles = coco_mask.frPyObjects(polygon, height, width)
                    rles_list.append(rles)
                else:
                    rles_list.append([])

        mask = torch.zeros((height,width,2,1),dtype=torch.uint8)
        for r,rles in enumerate(rles_list):
            if len(rles) > 0:
                mask[:,:,r,0] = torch.from_numpy(coco_mask.decode(rles)[:,:,0])
        # mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=-1)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0).permute(0,-1,1,2)
    else:
        masks = torch.zeros((0, 2, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False, overflow_boxes=False):
        self.return_masks = return_masks
        self.overflow_boxes = overflow_boxes

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        empty = anno[0]['empty']

        classes = [[obj["category_id"],obj["category_id_2"]] for obj in anno] 
        classes = torch.tensor(classes, dtype=torch.int64)

        boxes = [obj["bbox_1"] + obj["bbox_2"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 8)
        # x,y,w,h --> x,y,x,y
        boxes[:, 2:4] += boxes[:, :2]
        boxes[:, 6:8] += boxes[:, 4:6]
        if not self.overflow_boxes:
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)

        if self.return_masks:
            segmentations = [[obj["segmentation_1"],obj["segmentation_2"]] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        if not empty:
            # keypoints = None
            # if anno and "keypoints" in anno[0]:
            #     keypoints = [obj["keypoints"] for obj in anno]
            #     keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            #     num_keypoints = keypoints.shape[0]
            #     if num_keypoints:
            #         keypoints = keypoints.view(num_keypoints, -1, 3)

            keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0]) 
            keep_2 = (((boxes[:, 7] > boxes[:, 5]) & (boxes[:, 6] > boxes[:, 4])) + (torch.sum(boxes[:,4:],axis=1) == 0)) > 0

            if False in keep or (False in keep_2 and sum(boxes[:,4:]) > 0):
                print('Boxes are not correct in coco.py script')

            boxes = boxes[keep]
            classes = classes[keep]
            if self.return_masks:
                masks = masks[keep]
            # if keypoints is not None:
            #     keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes - 1

        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        # if keypoints is not None:
        #     target["keypoints"] = keypoints

        if anno and "track_id" in anno[0]:
            track_ids = torch.tensor([obj["track_id"] for obj in anno])
            target["track_ids"] = track_ids[keep]
        elif not len(boxes):
            target["track_ids"] = torch.empty(0)

        # for conversion to coco api
        area = torch.tensor([[obj["area_1"],obj["area_2"]] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        ignore = torch.tensor([obj["ignore"] if "ignore" in obj else 0 for obj in anno])

        if not empty:
            target["area"] = area[keep]
            target["iscrowd"] = iscrowd[keep]
            target["ignore"] = ignore[keep]
        else:
            target["area"] = area
            target["iscrowd"] = iscrowd
            target["ignore"] = ignore

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        target["empty"] = torch.tensor(empty)

        return image, target


def make_coco_transforms_cells(image_set, img_transform=None, overflow_boxes=False):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if image_set == 'train':
        transforms = [
            # T.RandomHorizontalFlip(),
            T.RandomGaussianBlur(),
            T.RandomGaussianNoise(),
            T.RandomIlluminationVoodoo(),
            
        ]
    elif image_set == 'val':
        transforms = [
            T.RandomGaussianBlur(),
            T.RandomGaussianNoise(),
            T.RandomIlluminationVoodoo(),
            # T.RandomHorizontalFlip(),
        ]
    else:
        ValueError(f'unknown {image_set}')

    return T.Compose(transforms), normalize