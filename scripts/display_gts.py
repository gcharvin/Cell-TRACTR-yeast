import json
import numpy as np
import cv2
from pathlib import Path
from pycocotools import mask as coco_mask
import matplotlib.pyplot as plt

datapath = Path('/projectnb/dunlop/ooconnor/object_detection/cell-trackformer/data/cells/new_dataset')

json_folders = ['prev','cur']
json_folders = ['prev_prev','prev_cur']
json_folders = ['fut_prev','fut']

# Opening JSON file
f = open(datapath  / 'annotations' / 'train' / (f'{json_folders[0]}.json'))
data_prev = json.load(f)

f = open(datapath / 'annotations' / 'train' / (f'{json_folders[1]}.json'))
data_cur = json.load(f)
# returns JSON object as 
# a dictionary


num_display = 5
alpha=0.4
anno_prev = data_prev['annotations']
images_prev = data_prev['images']

anno_cur = data_cur['annotations']
images_cur = data_cur['images']
np.random.seed(1)
img_ids = np.random.choice(range(len(images_cur)),num_display)

(datapath / 'generator').mkdir(exist_ok=True)

fps = (datapath / 'generator').glob('*.png')
for fp in fps:
    fp.unlink()


class Tool():
    def __init__(self,anno_prev,anno_cur):
        self.anno_prev = anno_prev
        self.anno_cur = anno_cur
        self.crop = [np.inf,np.inf,0,0]

    def update(self,height,width,colors,img_id):
        self.height = height
        self.width = width
        self.colors = colors
        self.img_id = img_id

    def update_crop(self,bounding_box):
        x0,y0,w,h = bounding_box
        
        self.crop[0] = int(np.min((self.crop[0],x0-10)))
        self.crop[1] = int(np.min((self.crop[1],y0-10)))

        self.crop[2] = int(np.max((self.crop[2],x0+w+10)))
        self.crop[3] = int(np.max((self.crop[3],y0+h+10)))

    def forward(self,images,track_id,prev_or_cur:str,color=None):

        if prev_or_cur == 'prev':
            anno = self.anno_prev
        elif prev_or_cur == 'cur':
            anno = self.anno_cur
        else:
            raise Exception('You need to enter "prev" or "cur" for prev_or_cur variable')

        img_bbox,img_mask,img_bbox_mask, bw_mask = images

        bounding_box = [ann['bbox_1'] for ann in anno if ann['image_id'] == self.img_id and ann['track_id'] == track_id][0]
        color = (0,0,0) if not color else color
        img_bbox = cv2.rectangle(
            img_bbox,
            (int(np.clip(bounding_box[0],0,self.width)), int(np.clip(bounding_box[1],0,self.height))),
            (int(np.clip(bounding_box[0] + bounding_box[2],0,self.width)), int(np.clip(bounding_box[1] + bounding_box[3],0,self.height))),
            color= color,
            thickness = 1)

        img_bbox_mask = cv2.rectangle(
            img_bbox_mask,
            (int(np.clip(bounding_box[0],0,self.width)), int(np.clip(bounding_box[1],0,self.height))),
            (int(np.clip(bounding_box[0] + bounding_box[2],0,self.width)), int(np.clip(bounding_box[1] + bounding_box[3],0,self.height))),
            color= color,
            thickness = 1)

        cell_1 = bounding_box

        bounding_box = [ann['bbox_2'] for ann in anno if ann['image_id'] == self.img_id and ann['track_id'] == track_id][0]

        if bounding_box[-1] > 0:
            img_bbox = cv2.rectangle(
                img_bbox,
                (int(np.clip(bounding_box[0],0,self.width)), int(np.clip(bounding_box[1],0,self.height))),
                (int(np.clip(bounding_box[0] + bounding_box[2],0,self.width)), int(np.clip(bounding_box[1] + bounding_box[3],0,self.height))),
                color= color,
                thickness = 1)

            img_bbox_mask = cv2.rectangle(
                img_bbox_mask,
                (int(np.clip(bounding_box[0],0,self.width)), int(np.clip(bounding_box[1],0,self.height))),
                (int(np.clip(bounding_box[0] + bounding_box[2],0,self.width)), int(np.clip(bounding_box[1] + bounding_box[3],0,self.height))),
                color= color,
                thickness = 1)

            cell_2 = bounding_box

            img_bbox = cv2.arrowedLine(
                img_bbox,
                (int(cell_1[0] + cell_1[2] // 2), int(cell_1[1] + cell_1[3] // 2)),
                (int(cell_2[0] + cell_2[2] // 2), int(cell_2[1] + cell_2[3] // 2)),
                color=(1, 1, 1),
                thickness=1,
            )

            img_bbox_mask = cv2.arrowedLine(
                img_bbox_mask,
                (int(cell_1[0] + cell_1[2] // 2), int(cell_1[1] + cell_1[3] // 2)),
                (int(cell_2[0] + cell_2[2] // 2), int(cell_2[1] + cell_2[3] // 2)),
                color=(1, 1, 1),
                thickness=1,
            )
        self.update_crop(bounding_box)

        seg = [ann['segmentation_1'] +  ann['segmentation_2'] for ann in anno if ann['image_id'] == self.img_id and ann['track_id'] == track_id][0]

        rles = coco_mask.frPyObjects(seg, self.height, self.width)
        mask = coco_mask.decode(rles)
        for i in range(mask.shape[-1]):
            mask_color = np.repeat(mask[:,:,i:i+1],3,axis=-1)
            mask_color = mask_color * color
            img_mask[mask_color > 1] = img_mask[mask_color>1] * (1-alpha) + mask_color[mask_color>1]  * alpha
            img_bbox_mask[mask_color > 1] = img_bbox_mask[mask_color>1] * (1-alpha) + mask_color[mask_color>1]  * alpha

        bw_mask += np.sum(mask,axis=-1).astype(np.uint8)*255

        return [img_bbox,img_mask,img_bbox_mask,bw_mask]


save_mask = True
save_bbox = True
save_bbox_mask = True
save_mask_bw = True
crop = False
tool = Tool(anno_prev,anno_cur)
np.random.seed(1)

for img_id in img_ids:
    
    height = images_cur[img_id]['height']
    width = images_cur[img_id]['width']

    fn = images_prev[img_id]['file_name']

    assert fn == images_cur[img_id]['file_name']

    fn = Path(fn)

    previmg = cv2.imread(str(datapath / 'train' / (f'{json_folders[0]}_img') / fn))
    img = cv2.imread(str(datapath / 'train' / (f'{json_folders[1]}_img') / fn))

    prev_gt = cv2.imread(str(datapath / 'train' / (f'{json_folders[0]}_gt') / fn),cv2.IMREAD_UNCHANGED)
    gt = cv2.imread(str(datapath / 'train' / (f'{json_folders[1]}_gt') / fn),cv2.IMREAD_UNCHANGED)

    previmg_bbox, previmg_mask, previmg_bbox_mask = [previmg.copy() for _ in range(3)]
    img_bbox, img_mask, img_bbox_mask = [img.copy() for _ in range(3)]
    
    track_ids_prev = [ann['track_id'] for ann in anno_prev if ann['image_id'] == img_id]
    track_ids_cur = [ann['track_id'] for ann in anno_cur if ann['image_id'] == img_id]

    track_ids_both = [track_id_prev for track_id_prev in track_ids_prev if track_id_prev in track_ids_cur]
    
    # np.random.seed(4)
    colors = [tuple((255*np.random.random(3))) for _ in range(len(set(track_ids_prev + track_ids_cur)))]
    
    tool.update(height,width,colors,img_id)

    prev_bw_mask = np.zeros((img.shape[:2]),dtype=np.uint8)
    bw_mask = np.zeros((img.shape[:2]),dtype=np.uint8)

    cur_images = [img_bbox,img_mask,img_bbox_mask,bw_mask]
    prev_images = [previmg_bbox,previmg_mask,previmg_bbox_mask,prev_bw_mask]

    for idx,track_id in enumerate(track_ids_both):
        prev_images = tool.forward(prev_images,track_id,prev_or_cur='prev',color=colors[idx])
        cur_images = tool.forward(cur_images,track_id,prev_or_cur='cur',color=colors[idx])

    track_leave = [id for id in track_ids_prev if id not in track_ids_cur]

    for track_id in track_leave:
        prev_images = tool.forward(prev_images,track_id,prev_or_cur='prev',color=colors[len(track_ids_both)])

    track_new = [id for id in track_ids_cur if id not in track_ids_prev]

    for track_id in track_new:
        cur_images = tool.forward(cur_images,track_id,prev_or_cur='cur',color=None)

    previmg_bbox,previmg_mask,previmg_bbox_mask,prev_bw_mask = prev_images
    img_bbox,img_mask,img_bbox_mask,bw_mask = cur_images

    if crop:
        previmg_bbox = previmg_bbox[tool.crop[1]:tool.crop[3],tool.crop[0]:tool.crop[2]]
        previmg_mask = previmg_mask[tool.crop[1]:tool.crop[3],tool.crop[0]:tool.crop[2]]
        previmg_bbox_mask = previmg_bbox_mask[tool.crop[1]:tool.crop[3],tool.crop[0]:tool.crop[2]]
        prevseg_bw = prevseg_bw[tool.crop[1]:tool.crop[3],tool.crop[0]:tool.crop[2]]
        previmg = previmg[tool.crop[1]:tool.crop[3],tool.crop[0]:tool.crop[2]]
        img_bbox = img_bbox[tool.crop[1]:tool.crop[3],tool.crop[0]:tool.crop[2]]
        img_mask = img_mask[tool.crop[1]:tool.crop[3],tool.crop[0]:tool.crop[2]]
        img_bbox_mask = img_bbox_mask[tool.crop[1]:tool.crop[3],tool.crop[0]:tool.crop[2]]
        seg_bw = seg_bw[tool.crop[1]:tool.crop[3],tool.crop[0]:tool.crop[2]]
        img = img[tool.crop[1]:tool.crop[3],tool.crop[0]:tool.crop[2]]
        gt = gt[tool.crop[1]:tool.crop[3],tool.crop[0]:tool.crop[2]]
        prev_gt = prev_gt[tool.crop[1]:tool.crop[3],tool.crop[0]:tool.crop[2]]
        
        
    if save_bbox:
        cv2.imwrite(str(datapath / 'generator' / (fn.stem + '_bbox.png')),np.concatenate((previmg_bbox,img_bbox),axis=1))

    if save_mask:
        cv2.imwrite(str(datapath / 'generator' / (fn.stem + '_mask.png')),np.concatenate((previmg_mask,img_mask),axis=1))

    if save_bbox:
        cv2.imwrite(str(datapath / 'generator' / (fn.stem + '_bbox_mask.png')),np.concatenate((previmg_bbox_mask,img_bbox_mask),axis=1))

    if save_mask_bw:
        cv2.imwrite(str(datapath / 'generator' / (fn.stem + '_mask_bw.png')),np.concatenate((prev_bw_mask,bw_mask),axis=1))
    save_gt = True
    if save_gt:
        gt_cells = set(np.concatenate((np.unique(gt)[1:],np.unique(prev_gt)[1:])))
        blank_gt = np.repeat(np.zeros_like(gt)[...,None],3,axis=-1).astype(np.uint8)
        blank_prev_gt = np.repeat(np.zeros_like(prev_gt)[...,None],3,axis=-1).astype(np.uint8)
        for c,gt_cell in enumerate(gt_cells):
            blank_gt[gt == gt_cell] = colors[c]
            blank_prev_gt[prev_gt == gt_cell] = colors[c]

        alpha = 0.4

        blank_prev_gt = alpha * blank_prev_gt + (1-alpha) * previmg
        blank_gt = alpha * blank_gt + (1-alpha) * img

        cv2.imwrite(str(datapath / 'generator' / (fn.stem + '_gt.png')),np.concatenate((blank_prev_gt,blank_gt),axis=1))

    cv2.imwrite(str(datapath / 'generator' / (fn.stem + '_og_img.png')),np.concatenate((previmg,img),axis=1)[:,:,0])