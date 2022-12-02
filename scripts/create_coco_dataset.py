import cv2 
from pathlib import Path
import json
import numpy as np                                 
from tqdm import tqdm

import numpy as np
from itertools import groupby
import random
from skimage.measure import label

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

def create_anno(mask,cell,image_id,track_id,annotation_id,category_id,min_area=20):
   
    mask_sc = mask == cell  

    mask_sc_label = label(mask_sc)
    mask_sc_ids = np.unique(mask_sc_label)[1:]

    check_min_area = [np.sum(mask_sc_label == mask_sc_id) > min_area for mask_sc_id in mask_sc_ids]

    if len(check_min_area) == 2 and sum(check_min_area) < 2:
        mask_sc_ids = np.array([mask_sc_ids[i] for i in range(len(check_min_area)) if check_min_area[i]])

    if len(mask_sc_ids) > 0:
        mask_sc = mask_sc_label == mask_sc_ids[0]

        object_area_1 = float((mask_sc > 0.0).sum())
        object_seg_1 = polygonFromMask(mask_sc)
                                    
        y, x = np.where(mask_sc != 0)

        X = np.min(x) 
        Y = np.min(y)
        width = (np.max(x) - np.min(x)) 
        height = (np.max(y) - np.min(y)) 
        bbox_1 = (X,Y,width,height)

        category_id_1 = category_id
        empty = False

    else:
        object_area_1 = 0
        object_seg_1 = []
        bbox_1 = (0,0,0,0)
        category_id_1 = 2
        empty = True

    if len(mask_sc_ids) > 1:

        mask_sc = mask_sc_label == mask_sc_ids[1]
    
        object_area_2 = float((mask_sc > 0.0).sum())
        object_seg_2 = polygonFromMask(mask_sc)
                                    
        y, x = np.where(mask_sc != 0)

        X = np.min(x) 
        Y = np.min(y)
        width = (np.max(x) - np.min(x)) 
        height = (np.max(y) - np.min(y)) 
        bbox_2 = (X,Y,width,height)

        category_id_2 = category_id

    else:
        object_area_2 = 0
        object_seg_2 = []
        bbox_2 = (0,0,0,0)
        category_id_2 = 2

    annotation = {
        'segmentation_1': object_seg_1,
        'segmentation_2': object_seg_2,
        'iscrowd': 0,
        'image_id': image_id,
        'category_id': category_id_1,
        'category_id_2': category_id_2,
        'id': annotation_id,
        'bbox_1': bbox_1,
        'bbox_2': bbox_2,
        'area_1': object_area_1,
        'area_2': object_area_2,
        'ignore': 0,
        'empty': empty,
    }

    if not empty:
        annotation['track_id'] = track_id

    return annotation

datapath = Path('/projectnb/dunlop/ooconnor/object_detection/trackformer_2d/data/cells/raw_data_ordered')

anno_folder = 'annotations'
(datapath.parent / anno_folder).mkdir(exist_ok=True)

if (datapath.parent / anno_folder / 'train.json').exists():
    (datapath.parent / anno_folder / 'train.json').unlink()

if (datapath.parent / anno_folder / 'val.json').exists():
    (datapath.parent / anno_folder / 'val.json').unlink()

category_id = 1
no_cell = 0
min_area = 20
mothermachine = True

img_fps = list((datapath / 'img').glob("*.png"))[:]
random.seed(1)
random.shuffle(img_fps)
train_val_split = 0.8
split = int(train_val_split*len(img_fps))
train_fps = sorted(img_fps[:split])
val_fps = sorted(img_fps[split:])

folders = ['train','val']

# Remove old data
for folder in folders:
    (datapath.parent / folder).mkdir(exist_ok=True)
    (datapath.parent / folder / 'img').mkdir(exist_ok=True)
    (datapath.parent / folder / 'gt').mkdir(exist_ok=True)

    json_path = datapath.parent / anno_folder / (f'{folder}.json')
    if json_path.exists():
        json_path.unlink()

    for fol in ['img','gt']:
        fps = (datapath.parent / folder / fol).glob('*.png')
        for fp in fps:
            fp.unlink()

(datapath.parent / 'combined').mkdir(exist_ok=True)

for idx,fps in enumerate([train_fps,val_fps]):
    # These ids will be automatically increased as we go
    annotation_id = 0
    image_id = 0
    track_id = 1
    
    metadata = {}

    licenses = []
    licenses.append({'id': 1,'name': 'MIT license', 'url':'add later'})

    categories = []
    categories.append({'id': category_id, 'name': 'cell'})

    info = {
        'contributor': 'Owen OConnor, JB Lugagne',
        'date_created':'2022',
        'description':'E Coli cells growing in mother machine',
        'version': '1.0',
        'year': '2022'
        }

    annotations = []
    images = []
    categories = []
    
    width = 32
    height = 256

    for counter,fp in enumerate(tqdm(fps)):

        previmg = cv2.imread(str(fp.parents[1] / 'previmg' / fp.name),cv2.IMREAD_ANYDEPTH)
        previmg = (previmg - np.min(previmg)) / np.ptp(previmg)
        previmg = cv2.resize(previmg,(width,height))
        previmg = (255 * previmg).astype(np.uint8)

        img = cv2.imread(str(fp.parents[1] / 'img' / fp.name),cv2.IMREAD_ANYDEPTH)
        img = (img - np.min(img)) / np.ptp(img)
        img = cv2.resize(img,(width,height))
        img = (255 * img).astype(np.uint8)

        # Read and resize inputs
        inputs = (cv2.imread(str(fp.parents[1] / 'inputs' / fp.name),cv2.IMREAD_ANYDEPTH))
        og_inputs = np.unique(label(inputs))[1:]
        inputs = cv2.resize(inputs,(width,height),interpolation= cv2.INTER_NEAREST).astype(np.uint16)
        inputs_label = np.unique(label(inputs > 0))[1:]

        # Read and resize outputs
        outputs = (cv2.imread(str(fp.parents[1] / 'outputs' / fp.name),cv2.IMREAD_ANYDEPTH))
        og_outputs = np.unique(label(outputs))[1:]
        outputs = cv2.resize(outputs,(width,height),interpolation= cv2.INTER_NEAREST).astype(np.uint16)
        outputs_label = np.unique(label(outputs > 0))[1:]

        # Check for resizing artifacts and remove them
        if len(inputs_label) != len(og_inputs) or len(outputs_label) != len(og_outputs):
            print(f'Resizing error\nRemoving: {fp.stem}')
            for data_folder in ['img','previmg','inputs','outputs']:
                (fp.parents[1] / data_folder / fp.name).unlink()
            continue

        cells_input = np.unique(inputs)[1:]
        cells_output = np.unique(outputs)[1:]

        # Remove any cells smaller than min_area as these objects are probably not cells
        cells_input = [i for i in cells_input if np.sum(inputs == i) > min_area]
        cells_output = [i for i in cells_output if np.sum(outputs == i) > min_area]

        # Divided cells into three categories
        # 1.) Cells that track from the previous to current frame
        cells_track = [i for i in cells_input if i in cells_output]
        # 2.) Cells that appear in the new frame (this should not happen for mothermachine)
        cells_new = [i for i in cells_output if i not in cells_input]
        # 3.) Cells that exit the chamber; The cell exists in the previous frame but is gone in the current frame
        cells_leave = [i for i in cells_input if i not in cells_output]

        cells_all = cells_track + cells_new + cells_leave
        
        # First check for empty chambers and make an annotation per empty chamber
        # Images with no objects are trickier to deal; this is a hack implementation

        cell = -1 # This will be used to label an empty chamber with cellnb -1

        if len(cells_input) == 0 and len(cells_output) != 0:
            print('Empty chamber in previous frame only')
            if mothermachine:
                raise Exception('Cells cannot spontaneously spawn in the current frame if they don"t exist in the previous frame')
            prev_annotation = create_anno(inputs,cell,image_id,-1,annotation_id,category_id)
            annotations.append(prev_annotation)
            annotation_id += 1

        elif len(cells_input) != 0 and len(cells_output) == 0:
            print('Empty chamber in current frame only')
            cur_annotation = create_anno(inputs,cell,image_id+1,-1,annotation_id,category_id)
            annotations.append(cur_annotation)
            annotation_id += 1        

        elif len(cells_input) == 0 and len(cells_output) == 0:
            print('Empty chambers in previous and current frame')
            prev_annotation = create_anno(inputs,cell,image_id,-1,annotation_id,category_id)
            annotations.append(prev_annotation)
            annotation_id += 1

            cur_annotation = create_anno(inputs,cell,image_id+1,-1,annotation_id,category_id)
            annotations.append(cur_annotation)
            annotation_id += 1

        # Then iterate through all cells to create the standard annotations
        for cell in cells_all:
            if cell in cells_track:

                prev_annotation = create_anno(inputs,cell,image_id,track_id,annotation_id,category_id)
                annotations.append(prev_annotation)
                annotation_id += 1

                cur_annotation = create_anno(outputs,cell,image_id+1,track_id,annotation_id,category_id)
                annotations.append(cur_annotation)                    
                annotation_id += 1

                track_id += 1

            elif cell in cells_leave:
                prev_annotation = create_anno(inputs,cell,image_id,track_id,annotation_id,category_id)
                annotations.append(prev_annotation)
                annotation_id += 1
                track_id += 1

            elif cell in cells_new:
                cur_annotation = create_anno(outputs,cell,image_id+1,track_id,annotation_id,category_id)
                annotations.append(cur_annotation)                    
                annotation_id += 1
                track_id += 1
            else:
                print('error')

        prev_image = {
            'license': 1,
            'file_name': fp.stem + '_prev.png',
            'height': inputs.shape[0],
            'width': inputs.shape[1],
            'id': image_id,
            'frame_id': 0,
            'seq_length': 1,
            'first_frame_image_id': image_id
           }    

        images.append(prev_image)
        image_id += 1

        cur_image = {
            'license': 1,
            'file_name': fp.stem + '_cur.png',
            'height': outputs.shape[0],
            'width': outputs.shape[1],
            'id': image_id,
            'frame_id': 1,
            'seq_length': 1,
            'first_frame_image_id': image_id-1,
            }    

        images.append(cur_image)

        image_id += 1

        cv2.imwrite(str(datapath.parent / folders[idx] / 'img' / (fp.stem + '_prev.png')),previmg)
        cv2.imwrite(str(datapath.parent / folders[idx] / 'img' /(fp.stem + '_cur.png')),img)

        cv2.imwrite(str(datapath.parent / folders[idx] / 'gt' / (fp.stem + '_prev.png')),inputs)
        cv2.imwrite(str(datapath.parent / folders[idx] / 'gt' / (fp.stem + '_cur.png')),outputs)

        cv2.imwrite(str(datapath.parent / 'combined' / (fp.stem + '_prev.png')),inputs)
        cv2.imwrite(str(datapath.parent / 'combined' / (fp.stem + '_cur.png')),outputs)

    metadata = {
        'annotations': annotations,
        'images': images,
        'categories': categories,
        'licenses': licenses,
        'info': info,
        'sequences': 'cells',
    }
        
    with open(datapath.parent / anno_folder / (f'{folders[idx]}.json'), 'w') as f:
        json.dump(metadata,f, cls=NpEncoder)
        


print(no_cell)