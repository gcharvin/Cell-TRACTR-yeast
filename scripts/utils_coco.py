import cv2 
import json
import numpy as np
from itertools import groupby
from skimage.measure import label
import random
import re
import tifffile

random.seed(1)

def train_val_split(files,split=0.8):

    random.shuffle(files)
    train_val_split = 0.8
    split = int(train_val_split*len(files))
    train = sorted(files[:split])
    val = sorted(files[split:])

    return train, val

def create_folders(datapath,folders):

    # Create annotations folder
    (datapath / 'annotations').mkdir(exist_ok=True)
    (datapath / 'man_track').mkdir(exist_ok=True)

    # Create train and val folder to store images and ground truths
    for folder in folders:
        (datapath / 'annotations' / folder).mkdir(exist_ok=True)
        (datapath / folder).mkdir(exist_ok=True)
        for img_type in ['img','gt']:
            (datapath / folder / img_type).mkdir(exist_ok=True)

    # Remove all data (images + json file) from folders
    for folder in folders:
        for img_type in ['img','gt']:
            delete_fps = (datapath / folder / img_type).glob('*.tif')
            for delete_fp in delete_fps:
                delete_fp.unlink()

        json_paths = (datapath / 'annotations' / folder).glob('*.json')
        for json_path in json_paths:
            json_path.unlink()

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle

def polygonFromMask(seg):
    # adapted from https://github.com/hazirbas/coco-json-converter/blob/master/generate_coco_json.py
    contours, _ = cv2.findContours(seg.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    valid_poly = 0
    for contour in contours:
    # Valid polygons have >= 6 coordinates (3 points)
        if contour.size >= 6:
            segmentation.append(contour.astype(float).flatten().tolist())
            valid_poly += 1
    if valid_poly == 0:
        raise ValueError
    return segmentation


class reader():
    def __init__(self,mothermachine,target_size,min_area):

        self.target_size = target_size
        self.min_area = min_area
        self.dtype = {'uint8': 255,
                      'uint16': 65535}

        if mothermachine:
            self.crop = False
        else:
            self.crop = True



    def get_slices(self,seg,shift):
    
        y,x = np.where(seg)
        y,x = int(np.mean(y)) + shift[0] , int(np.mean(x)) + shift[1]

        if y < 0:
            y = 0
        elif y > seg.shape[0]:
            y = seg.shape[0]

        if x < 0:
            x = 0
        elif x > seg.shape[1]:
            x = seg.shape[1]
            
        if (y - self.target_size[0]/2 > 0 and y + self.target_size[0]/2 - 1 < seg.shape[0]):
            y0 = int(y - self.target_size[0] / 2)
        elif y + self.target_size[0]/2 - 1 > seg.shape[0] and seg.shape[0] - self.target_size[0] > 0:
            y0 = seg.shape[0] - self.target_size[0]
        else:
            y0 = 0
            
        y1 = y0 + self.target_size[0] if y0 + self.target_size[0] < seg.shape[0] else seg.shape[0]
        
        if (x - self.target_size[1]/2 > 0 and x + self.target_size[1]/2 - 1 < seg.shape[1]):
            x0 = int(x - self.target_size[1] / 2)
        elif x + self.target_size[1]/2 - 1 > seg.shape[1] and seg.shape[1] - self.target_size[1] > 0:
            x0 = seg.shape[1] - self.target_size[1]
        else: 
            x0 = 0
            
        x1 = x0 + self.target_size[1] if x0 + self.target_size[1] < seg.shape[1] else seg.shape[1]   

        self.y = [y0,y1]
        self.x = [x0,x1]



    def read_image(self,fp):

        img = cv2.imread(str(fp),cv2.IMREAD_ANYDEPTH)
        img = img / self.dtype[str(img.dtype)]

        if self.crop:
            img = img[self.y[0]:self.y[1],self.x[0]:self.x[1]]

            if img.shape[0] < self.target_size[0]:
                img = np.pad(img,((0,self.target_size[0] - img.shape[0]),(0,0)))

            if img.shape[1] < self.target_size[1]:
                img = np.pad(img,((0,0),(0,self.target_size[1] - img.shape[1])))
                
        else:
            img = cv2.resize(img,(self.target_size[1],self.target_size[0]))

        img = (255 * img).astype(np.uint8)

        return img

    def read_gt(self,fp,counter,track_file):

        nb = re.findall('\d+',fp.stem)[-1]
        pad = len(nb)
        data_set_nb = fp.parts[-2]
        gt_fp = fp.parents[1] / (data_set_nb + '_GT') / 'TRA' / (f'man_track{int(nb):0{pad}}{fp.suffix}')

        # Read inputs and outputs
        # gt = tifffile.imread(gt_fp).astype(np.uint16)
        gt = cv2.imread(str(gt_fp),cv2.IMREAD_UNCHANGED).astype(np.uint16)

        # Crop or resize inputs and outputs to target_size
        if self.crop:
            prev_gt = prev_gt[self.y[0]:self.y[1],self.x[0]:self.x[1]]
            cur_gt = cur_gt[self.y[0]:self.y[1],self.x[0]:self.x[1]]

            if prev_gt.shape[0] < self.target_size[0]:
                prev_gt = np.pad(prev_gt,((0,self.target_size[0] - prev_gt.shape[0]),(0,0)))
                cur_gt = np.pad(cur_gt,((0,self.target_size[0] - cur_gt.shape[0]),(0,0)))

            if cur_gt.shape[1] < self.target_size[1]:
                prev_gt = np.pad(prev_gt,((0,0),(0,self.target_size[1] - prev_gt.shape[1])))
                cur_gt = np.pad(cur_gt,((0,0),(0,self.target_size[1] - cur_gt.shape[1])))
        else:      
            
            gt_resized = np.zeros((self.target_size),dtype=np.uint16)
            cellnbs = np.unique(gt)
            cellnbs = cellnbs[cellnbs != 0]
            for cellnb in cellnbs:
                mask_cellnb = ((gt == cellnb)*255).astype(np.uint8)
                mask_cellnb_resized = cv2.resize(mask_cellnb,(self.target_size[1],self.target_size[0]),interpolation= cv2.INTER_NEAREST)

                contours, _ = cv2.findContours(mask_cellnb_resized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                if (mask_cellnb_resized > 127).sum() > self.min_area and sum([contour.size >= 6 for contour in contours]) > 0:
                    gt_resized[mask_cellnb_resized > 127] = cellnb
                else:
                    raise NotImplementedError

                    #TODO need to remove cell from man_track.txt if it's too small in terms of area or contour.size

            gt = gt_resized

        return gt

def create_anno(mask,cellnb,image_id,annotation_id,dataset_name):
   
    mask_sc = mask == cellnb 

    mask_sc_label = label(mask_sc)
    mask_sc_ids = np.unique(mask_sc_label)[1:]

    if len(mask_sc_ids) > 0:

        mask_sc = mask_sc_label == mask_sc_ids[0]

        area = float((mask_sc > 0.0).sum())
        seg = polygonFromMask(mask_sc)
                                    
        y, x = np.where(mask_sc != 0)

        X = np.min(x) 
        Y = np.min(y)
        width = (np.max(x) - np.min(x)) 
        height = (np.max(y) - np.min(y)) 
        bbox = (X,Y,width,height)
        empty = False

    else: #empty frame
        area = 0
        seg = []
        bbox = (0,0,0,0)
        empty = True

        assert cellnb == -1

    annotation = {
        'id': annotation_id,
        'image_id': image_id,
        'bbox': bbox,
        'segmentation': seg,
        'area': area,
        'category_id': 1,
        'track_id': cellnb,
        'dataset_name': dataset_name,
        'ignore': 0,
        'iscrowd': 0,
        'empty': empty,
    }

    return annotation
