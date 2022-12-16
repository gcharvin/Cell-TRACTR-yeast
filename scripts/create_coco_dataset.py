import cv2 
from pathlib import Path
import json
import numpy as np                                 
from tqdm import tqdm

import numpy as np
from itertools import groupby
import random
from skimage.measure import label
import re 

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


def read_image(filepath,width=32,height=256):

    img = cv2.imread(str(filepath),cv2.IMREAD_ANYDEPTH)
    img = (img - np.min(img)) / np.ptp(img)
    img = cv2.resize(img,(width,height))
    img = (255 * img).astype(np.uint8)

    return img

def read_gts(fp,reset = False, width=32, height=256):

    # Read and resize inputs
    prev_gt = (cv2.imread(str(fp.parents[1] / 'inputs' / fp.name),cv2.IMREAD_ANYDEPTH))
    orginal_size_prev_gt = np.unique(label(prev_gt))[1:]
    prev_gt = cv2.resize(prev_gt,(width,height),interpolation= cv2.INTER_NEAREST).astype(np.uint16)

    # Read and resize outputs
    cur_gt = (cv2.imread(str(fp.parents[1] / 'outputs' / fp.name),cv2.IMREAD_ANYDEPTH))
    original_size_cur_gt = np.unique(label(cur_gt))[1:]
    cur_gt = cv2.resize(cur_gt,(width,height),interpolation= cv2.INTER_NEAREST).astype(np.uint16)

    prev_gt_label = np.unique(label(prev_gt > 0))[1:]
    cur_gt_label = np.unique(label(cur_gt > 0))[1:]
    # Check for resizing artifacts and remove them
    if len(prev_gt_label) != len(orginal_size_prev_gt) or len(cur_gt_label) != len(original_size_cur_gt):
        print(f'Resizing error\nRemoving: {fp.stem}')
        for data_folder in ['img','previmg','inputs','outputs']:
            (fp.parents[1] / data_folder / fp.name).unlink()
        
        reset = True

    return prev_gt, cur_gt, reset

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


def compile_annotations(prev_gt,cur_gt,prev_annotations,cur_annotations,track_id,anno_ids):

    prev_annotation_id, cur_annotation_id = anno_ids

    cells_prev = np.unique(prev_gt)[1:]
    cells_cur = np.unique(cur_gt)[1:]

    # Remove any cells smaller than min_area as these objects are probably not cells
    cells_prev = [i for i in cells_prev if np.sum(prev_gt == i) > min_area]
    cells_cur = [i for i in cells_cur if np.sum(cur_gt == i) > min_area]

    # Divided cells into three categories
    # 1.) Cells that track from the previous to current frame
    cells_track = [i for i in cells_prev if i in cells_cur]
    # 2.) Cells that appear in the new frame (this should not happen for mothermachine)
    cells_new = [i for i in cells_cur if i not in cells_prev]
    # 3.) Cells that exit the chamber; The cell exists in the previous frame but is gone in the current frame
    cells_leave = [i for i in cells_prev if i not in cells_cur]

    cells_all = cells_track + cells_new + cells_leave
    
    # First check for empty chambers and make an annotation per empty chamber
    # Images with no objects are trickier to deal; this is a hack implementation

    cell = -1 # This will be used to label an empty chamber with cellnb -1

    if len(cells_prev) == 0 and len(cells_cur) != 0:
        print('Empty chamber in previous frame only')
        if mothermachine:
            raise Exception('Cells cannot spontaneously spawn in the current frame if they don"t exist in the previous frame')
        prev_annotation = create_anno(prev_gt,cell,image_id,-1,prev_annotation_id,category_id)
        prev_annotations.append(prev_annotation)
        prev_annotation_id += 1

    elif len(cells_prev) != 0 and len(cells_cur) == 0:
        print('Empty chamber in current frame only')
        cur_annotation = create_anno(cur_gt,cell,image_id+1,-1,cur_annotation_id,category_id)
        cur_annotations.append(cur_annotation)
        cur_annotation_id += 1        

    elif len(cells_prev) == 0 and len(cells_cur) == 0:
        print('Empty chambers in previous and current frame')
        prev_annotation = create_anno(prev_gt,cell,image_id,-1,prev_annotation_id,category_id)
        prev_annotations.append(prev_annotation)
        prev_annotation_id += 1

        cur_annotation = create_anno(cur_gt,cell,image_id+1,-1,cur_annotation_id,category_id)
        cur_annotations.append(cur_annotation)
        cur_annotation_id += 1

    # Then iterate through all cells to create the standard annotations
    for cell in cells_all:
        if cell in cells_track:

            prev_annotation = create_anno(prev_gt,cell,image_id,track_id,prev_annotation_id,category_id)
            prev_annotations.append(prev_annotation)
            prev_annotation_id += 1

            cur_annotation = create_anno(cur_gt,cell,image_id+1,track_id,cur_annotation_id,category_id)
            cur_annotations.append(cur_annotation)                    
            cur_annotation_id += 1

            track_id += 1

        elif cell in cells_leave:
            prev_annotation = create_anno(prev_gt,cell,image_id,track_id,prev_annotation_id,category_id)
            prev_annotations.append(prev_annotation)
            prev_annotation_id += 1
            track_id += 1

        elif cell in cells_new:
            cur_annotation = create_anno(cur_gt,cell,image_id+1,track_id,cur_annotation_id,category_id)
            cur_annotations.append(cur_annotation)                    
            cur_annotation_id += 1
            track_id += 1
        else:
            print('error')

    anno_ids = [prev_annotation_id,cur_annotation_id]

    return cur_annotations, prev_annotations, track_id, anno_ids

datapath = Path('/projectnb/dunlop/ooconnor/object_detection/cell-trackformer/data/cells/new_dataset')

anno_folder = 'annotations'
folders = ['train','val']

(datapath / anno_folder).mkdir(exist_ok=True)

category_id = 1
no_cell = 0
min_area = 20
mothermachine = True

img_fps = list((datapath / 'raw_data' / 'img').glob("*.png"))[:]

# Configure metadata
meta_data = np.zeros((len(img_fps),5),dtype=int)
for idx,fp in enumerate(img_fps):

    fn = fp.stem

    numbers = list(map(int,re.findall('\d+',fn)))

    meta_data[idx] = numbers

fns = []

exps = np.unique(meta_data[:,0])

for exp in exps:
    exp_data = meta_data[meta_data[:,0] == exp][:,1:]

    training_sets = np.unique(exp_data[:,0])

    for training_set in training_sets:
        ts_data = exp_data[exp_data[:,0] == training_set][:,1:]

        positions = np.unique(ts_data[:,0])

        for position in positions:
            pos_data = ts_data[ts_data[:,0] == position][:,1:]

            chambers = np.unique(pos_data[:,0])

            for chamber in chambers:

                cha_data = pos_data[pos_data[:,0] == chamber][:,1:]

                frames = np.unique(cha_data[:,0])

                for frame in frames:

                    if frame - 1 in frames and frame + 1 in frames:

                        ind = np.where((meta_data[:,0] == exp) * (meta_data[:,1] == training_set) * (meta_data[:,2] == position) * (meta_data[:,3] == chamber) * (meta_data[:,4] == frame) == True)[0][0]

                        fns.append(img_fps[ind].name)

random.seed(1)
random.shuffle(fns)
train_val_split = 0.8
split = int(train_val_split*len(fns))
train_fns = sorted(fns[:split])
val_fns = sorted(fns[split:])


# Remove old data
for folder in folders:

    for time_ref in ['fut_','cur_','prev_','prev_prev_']:
        for img_type in ['img','gt']:
            (datapath/ folder / (time_ref + img_type)).mkdir(exist_ok=True)

            delete_fps = (datapath / folder / (time_ref + img_type)).glob('*.png')
            for delete_fp in delete_fps:
                delete_fp.unlink()

    (datapath / anno_folder / folder).mkdir(exist_ok=True)

    json_paths = (datapath / anno_folder / folder).glob('*.json')
    for json_path in json_paths:
        json_path.unlink()


for idx,dataset_fns in enumerate([train_fns,val_fns]):
    # These ids will be automatically increased as we go
    annotation_id = 0
    image_id = 0
    track_id = 1
    
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

    annotations = [[[],[]] for _ in range(3)]
    annotation_ids = [[0,0] for _ in range(3)]
    # 0 - prev_prev
    # 1 - prev_cur
    # 2 - prev
    # 3 - cur
    # 4 - fut_prev
    # 5 - fut

    images = []

    categories = []
    
    width = 32
    height = 256

    for counter,fn in enumerate(tqdm(dataset_fns)):

        prev_img = read_image(datapath / 'raw_data' / 'previmg' / fn)
        cur_img = read_image(datapath / 'raw_data' / 'img' / fn)

        prev_gt, cur_gt, reset = read_gts(datapath / 'raw_data' / 'inputs' / fn)


        framenb = (re.findall('\d+',fn)[-1])
        framenb_plus1 = f'{int(framenb)+1:06d}'
        framenb_minus1 = f'{int(framenb)-1:06d}'

        fn_plus1 = fn.replace(framenb,framenb_plus1)
        fn_minus1 = fn.replace(framenb,framenb_minus1)

        prev_prev_img = read_image(datapath / 'raw_data' / 'previmg' / fn_minus1)
        fut_img = read_image(datapath / 'raw_data' / 'img' / fn_plus1)

        prev_prev_gt, prev_cur_gt, reset = read_gts(datapath / 'raw_data' / 'inputs' / fn_minus1)
        fut_prev_gt, fut_gt, reset = read_gts(datapath / 'raw_data' / 'inputs' / fn_plus1)

        if reset:
            raise Exception

        gts = [[prev_prev_gt,prev_cur_gt],[prev_gt,cur_gt],[fut_prev_gt,fut_gt]]

        for k, ((gt_0,gt_1),(anno_0,anno_1),anno_ids) in enumerate(zip(gts,annotations,annotation_ids)):
            anno_0,anno_1,track_id, anno_ids = compile_annotations(gt_0,gt_1,anno_0,anno_1,track_id,anno_ids)
            annotations[k] = [anno_0,anno_1]
            annotation_ids[k] = anno_ids


        image = {
            'license': 1,
            'file_name': fn,
            'height': prev_gt.shape[0],
            'width': prev_gt.shape[1],
            'id': image_id,
            'frame_id': 0,
            'seq_length': 1,
            'first_frame_image_id': image_id
            }    

        images.append(image)

        image_id += 1

        cv2.imwrite(str(datapath / folders[idx] / 'prev_prev_img' / fn),prev_prev_img)
        cv2.imwrite(str(datapath / folders[idx] / 'prev_cur_img' / fn),prev_img)
        
        cv2.imwrite(str(datapath / folders[idx] / 'prev_img' / fn),prev_img)
        cv2.imwrite(str(datapath / folders[idx] / 'cur_img' / fn),cur_img)

        cv2.imwrite(str(datapath / folders[idx] / 'fut_prev_img' / fn),cur_img)
        cv2.imwrite(str(datapath / folders[idx] / 'fut_img' / fn),fut_img)

        cv2.imwrite(str(datapath / folders[idx] / 'prev_gt' / fn),prev_gt)
        cv2.imwrite(str(datapath / folders[idx] / 'cur_gt' / fn),cur_gt)

        cv2.imwrite(str(datapath / folders[idx] / 'prev_prev_gt' / fn),prev_prev_gt)
        cv2.imwrite(str(datapath / folders[idx] / 'prev_cur_gt' / fn),prev_cur_gt)

        cv2.imwrite(str(datapath / folders[idx] / 'fut_gt' / fn),fut_gt)
        cv2.imwrite(str(datapath / folders[idx] / 'fut_prev_gt' / fn),fut_prev_gt)
    
    json_folders = ['prev_prev','prev_cur','prev','cur','fut_prev','fut']

    for m in range(len(json_folders)):
        metadata = {
            'annotations': annotations[m//2][m%2],
            'images': images,
            'categories': categories,
            'licenses': licenses,
            'info': info,
            'sequences': 'cells',
        }
        
        with open(datapath / anno_folder / (f'{folders[idx]}') / (f'{json_folders[m]}.json'), 'w') as f:
            json.dump(metadata,f, cls=NpEncoder)
        


print(no_cell)