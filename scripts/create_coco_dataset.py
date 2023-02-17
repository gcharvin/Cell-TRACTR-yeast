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


class reader():
    def __init__(self,mothermachine,target_size):

        self.target_size = target_size

        if mothermachine:
            self.crop = False
        else:
            self.crop = True

    def get_slices(self,seg):

        y,x = np.where(seg)
        y,x = int(np.mean(y)), int(np.mean(x))

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
        img = (img - np.min(img)) / np.ptp(img) if np.ptp(img) != 0 else np.zeros_like(img)

        if self.crop:
            img = img[self.y[0]:self.y[1],self.x[0]:self.x[1]]

            if img.shape[0] < target_size[0]:
                img = np.pad(img,((0,target_size[0] - img.shape[0]),(0,0)))

            if img.shape[1] < target_size[1]:
                img = np.pad(img,((0,0),(0,target_size[1] - img.shape[1])))
                
        else:
            img = cv2.resize(img,(self.target_size[1],self.target_size[0]))

        img = (255 * img).astype(np.uint8)

        return img

    def read_gt(self,fp):

        # Read inputs and outputs
        prev_gt = (cv2.imread(str(fp.parents[1] / 'inputs' / fp.name),cv2.IMREAD_ANYDEPTH))
        cur_gt = (cv2.imread(str(fp.parents[1] / 'outputs' / fp.name),cv2.IMREAD_ANYDEPTH))

        # Crop or resize inputs and outputs to target_size
        if self.crop:
            prev_gt = prev_gt[self.y[0]:self.y[1],self.x[0]:self.x[1]]
            cur_gt = cur_gt[self.y[0]:self.y[1],self.x[0]:self.x[1]]

            if prev_gt.shape[0] < target_size[0]:
                prev_gt = np.pad(prev_gt,((0,target_size[0] - prev_gt.shape[0]),(0,0)))
                cur_gt = np.pad(cur_gt,((0,target_size[0] - cur_gt.shape[0]),(0,0)))

            if cur_gt.shape[1] < target_size[1]:
                prev_gt = np.pad(prev_gt,((0,0),(0,target_size[1] - prev_gt.shape[1])))
                cur_gt = np.pad(cur_gt,((0,0),(0,target_size[1] - cur_gt.shape[1])))
        else:
            orginal_size_prev_gt = np.unique(label(prev_gt))[1:]
            prev_gt = cv2.resize(prev_gt,(target_size[1],target_size[0]),interpolation= cv2.INTER_NEAREST).astype(np.uint16)
       
            original_size_cur_gt = np.unique(label(cur_gt))[1:]
            cur_gt = cv2.resize(cur_gt,(target_size[1],target_size[0]),interpolation= cv2.INTER_NEAREST).astype(np.uint16)

            prev_gt_label = np.unique(label(prev_gt > 0))[1:]
            cur_gt_label = np.unique(label(cur_gt > 0))[1:]
            # Check for resizing artifacts and remove them
            if len(prev_gt_label) != len(orginal_size_prev_gt) or len(cur_gt_label) != len(original_size_cur_gt):
                print(f'Resizing error\nRemoving: {fp.stem}')
                for data_folder in ['img','previmg','inputs','outputs']:
                    (fp.parents[1] / data_folder / fp.name).unlink()
            
                raise Exception

        return prev_gt, cur_gt

def create_anno(mask,cell,image_id,track_id,annotation_id,category_id,min_area=50):
   
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

    if not empty and track_id > 0:
        annotation['track_id'] = track_id

    return annotation


def compile_annotations(gts,annotations_old,annotation_ids_old,track_id,image_id):

    prev_gt,cur_gt = gts
    prev_annotations,cur_annotations = annotations_old
    prev_annotation_id, cur_annotation_id = annotation_ids_old

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
    assert len(cells_all) == len(set(cells_all))
    
    # First check for empty chambers and make an annotation per empty chamber
    # Images with no objects are trickier to deal; this is a hack implementation

    cell = -1 # This will be used to label an empty chamber with cellnb -1

    if len(cells_prev) == 0 and len(cells_cur) != 0:
        print('No cells in previous frame only')
        if mothermachine:
            raise Exception('Cells cannot spontaneously spawn in the current frame if they don"t exist in the previous frame')
        annotation = create_anno(prev_gt,cell,image_id,-1,prev_annotation_id,category_id)
        prev_annotations.append(annotation)
        prev_annotation_id += 1

    elif len(cells_prev) != 0 and len(cells_cur) == 0:
        print('No cells in current frame only')
        annotation = create_anno(cur_gt,cell,image_id,-1,cur_annotation_id,category_id)
        cur_annotations.append(annotation)
        cur_annotation_id += 1        

    elif len(cells_prev) == 0 and len(cells_cur) == 0:
        print('No cells in previous and current frame')
        annotation = create_anno(prev_gt,cell,image_id,-1,prev_annotation_id,category_id)
        prev_annotations.append(annotation)
        prev_annotation_id += 1

        annotation = create_anno(cur_gt,cell,image_id,-1,cur_annotation_id,category_id)
        cur_annotations.append(annotation)
        cur_annotation_id += 1

    # Then iterate through all cells to create the standard annotations
    for cell in cells_all:
        if cell in cells_track:

            annotation = create_anno(prev_gt,cell,image_id,track_id,prev_annotation_id,category_id)
            prev_annotations.append(annotation)
            prev_annotation_id += 1

            annotation = create_anno(cur_gt,cell,image_id,track_id,cur_annotation_id,category_id)
            cur_annotations.append(annotation)                    
            cur_annotation_id += 1

            track_id += 1

        elif cell in cells_leave:
            annotation = create_anno(prev_gt,cell,image_id,track_id,prev_annotation_id,category_id)
            prev_annotations.append(annotation)
            prev_annotation_id += 1
            track_id += 1

        elif cell in cells_new:
            cur_annotation = create_anno(cur_gt,cell,image_id,track_id,cur_annotation_id,category_id)
            cur_annotations.append(cur_annotation)                    
            cur_annotation_id += 1
            track_id += 1
        else:
            print('error')

    updated_annotation_ids = [prev_annotation_id,cur_annotation_id]
    updated_annotations = [prev_annotations,cur_annotations]

    return updated_annotations, track_id, updated_annotation_ids

# datapath = Path('/projectnb/dunlop/ooconnor/object_detection/cell-trackformer/data/cells/new_dataset')
datapath = Path('/projectnb/dunlop/ooconnor/16bit/celltrackformer')

anno_folder = 'annotations'
(datapath / anno_folder).mkdir(exist_ok=True)

folders = ['train','val']
for folder in folders:
    (datapath / folder).mkdir(exist_ok=True)

category_id = 1
no_cell = 0
min_area = 75
mothermachine = False
target_size = (256,256)

img_fps = list((datapath / 'raw_data' / 'img').glob("*.png"))[:]

# Configure metadata
if mothermachine:
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

else:
    fns = []

    for idx,fp in enumerate(img_fps):
    
        fn = fp.stem

        framenb_str = re.findall('\d+',fn)[-1]
        pad = len(framenb_str)
        framenb = int(framenb_str)
        index_str = fn.index(framenb_str)

        basename = fn[:index_str]

        prev_basename = basename + f'{framenb-1:0{pad}d}{fp.suffix}'
        fut_basename = basename + f'{framenb+1:0{pad}d}{fp.suffix}'

        if (datapath / 'raw_data' / 'img' / prev_basename).exists() and (datapath / 'raw_data' / 'img' / fut_basename).exists():
            fns.append(img_fps[idx].name)


random.seed(1)
random.shuffle(fns)
train_val_split = 0.8
split = int(train_val_split*len(fns))
train_fns = sorted(fns[:split])
val_fns = sorted(fns[split:])


# Remove old data
for folder in folders:

    for time_ref in ['fut_','fut_prev_','cur_','prev_','prev_prev_','prev_cur_']:
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

    img_reader = reader(mothermachine=mothermachine, target_size=target_size)

    for counter,fn in enumerate(tqdm(dataset_fns)):

        prev_inputs = cv2.imread(str(datapath / 'raw_data' / 'inputs' / fn),cv2.IMREAD_ANYDEPTH)
        
        img_reader.get_slices((prev_inputs > 0) * 1)

        prev_img = img_reader.read_image(datapath / 'raw_data' / 'previmg' / fn)
        cur_img = img_reader.read_image(datapath / 'raw_data' / 'img' / fn)

        prev_gt, cur_gt = img_reader.read_gt(datapath / 'raw_data' / 'inputs' / fn)

        framenb = re.findall('\d+',fn)[-1]
        pad = len(framenb)
        framenb_plus1 = f'{int(framenb)+1:0{pad}d}'
        framenb_minus1 = f'{int(framenb)-1:0{pad}d}'

        fn_plus1 = fn.replace(framenb,framenb_plus1)
        fn_minus1 = fn.replace(framenb,framenb_minus1)

        prev_prev_img = img_reader.read_image(datapath / 'raw_data' / 'previmg' / fn_minus1)
        fut_img = img_reader.read_image(datapath / 'raw_data' / 'img' / fn_plus1)

        prev_prev_gt, prev_cur_gt = img_reader.read_gt(datapath / 'raw_data' / 'inputs' / fn_minus1)
        fut_prev_gt, fut_gt = img_reader.read_gt(datapath / 'raw_data' / 'inputs' / fn_plus1)

        gts = [[prev_prev_gt,prev_cur_gt],[prev_gt,cur_gt],[fut_prev_gt,fut_gt]]

        for k in range(len(gts)):
            updated_annotations,track_id, updated_annotation_ids = compile_annotations(gts[k],annotations[k],annotation_ids[k],track_id,image_id)
            annotations[k] = updated_annotations
            annotation_ids[k] = updated_annotation_ids


        image = {
            'license': 1,
            'file_name': fn,
            'height': prev_gt.shape[0],
            'width': prev_gt.shape[1],
            'id': image_id,
            'frame_id': 0,
            'seq_length': 1,
            }    

        images.append(image)

        image_id += 1

        cv2.imwrite(str(datapath / folders[idx] / 'prev_prev_img' / fn),prev_prev_img)
        cv2.imwrite(str(datapath / folders[idx] / 'prev_prev_gt' / fn),prev_prev_gt)

        cv2.imwrite(str(datapath / folders[idx] / 'prev_cur_img' / fn),prev_img)        
        cv2.imwrite(str(datapath / folders[idx] / 'prev_cur_gt' / fn),prev_cur_gt)

        cv2.imwrite(str(datapath / folders[idx] / 'prev_img' / fn),prev_img)
        cv2.imwrite(str(datapath / folders[idx] / 'prev_gt' / fn),prev_gt)

        cv2.imwrite(str(datapath / folders[idx] / 'cur_img' / fn),cur_img)
        cv2.imwrite(str(datapath / folders[idx] / 'cur_gt' / fn),cur_gt)
        
        cv2.imwrite(str(datapath / folders[idx] / 'fut_img' / fn),fut_img)
        cv2.imwrite(str(datapath / folders[idx] / 'fut_gt' / fn),fut_gt)

        cv2.imwrite(str(datapath / folders[idx] / 'fut_prev_img' / fn),cur_img)
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