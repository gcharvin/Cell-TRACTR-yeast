import json
import numpy as np
import cv2
from pathlib import Path
from pycocotools import mask as coco_mask
import matplotlib.pyplot as plt

datapath = Path('/projectnb/dunlop/ooconnor/object_detection/trackformer_2d/data/cells/annotations')

# Opening JSON file
f = open(datapath / 'train.json')
data = json.load(f)

# returns JSON object as 
# a dictionary


num_display = 20
alpha=0.4
anno = data['annotations']
images = data['images']
images_frame0 = [img['id'] for img in images if img['frame_id'] == 0]
np.random.seed(1)
img_ids = np.random.choice(images_frame0,num_display)

(datapath.parent / 'generator').mkdir(exist_ok=True)


class Tool():
    def __init__(self,anno):
        self.anno = anno
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

    def forward(self,img_bbox,img_mask,img_bbox_mask,track_id,frame_id,color=None):

        bounding_box = [ann['bbox_1'] for ann in anno if ann['image_id'] == self.img_id + frame_id and ann['track_id'] == track_id][0]
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

        bounding_box = [ann['bbox_2'] for ann in anno if ann['image_id'] == self.img_id + frame_id and ann['track_id'] == track_id][0]

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

        seg = [ann['segmentation_1'] +  ann['segmentation_2'] for ann in anno if ann['image_id'] == self.img_id + frame_id and ann['track_id'] == track_id][0]

        rles = coco_mask.frPyObjects(seg, self.height, self.width)
        mask = coco_mask.decode(rles)
        for i in range(mask.shape[-1]):
            mask_color = np.repeat(mask[:,:,i:i+1],3,axis=-1)
            mask_color = mask_color * color
            img_mask[mask_color > 1] = img_mask[mask_color>1] * (1-alpha) + mask_color[mask_color>1]  * alpha
            img_bbox_mask[mask_color > 1] = img_bbox_mask[mask_color>1] * (1-alpha) + mask_color[mask_color>1]  * alpha

        return img_bbox,img_mask,img_bbox_mask, np.sum(mask,axis=-1).astype(np.uint8)*255


save_mask = True
save_bbox = True
save_bbox_mask = True
save_mask_bw = True
crop = False
tool = Tool(anno)
np.random.seed(1)

for img_id in img_ids:
    
    height = images[img_id]['height']
    width = images[img_id]['width']

    assert images[img_id]['frame_id'] == 0
    assert images[img_id+1]['frame_id'] == 1

    prevfn = images[img_id]['file_name']
    curfn = images[img_id+1]['file_name']

    assert prevfn[:-9] == curfn[:-8]
    
    previmg = cv2.imread(str(datapath.parent / 'train' / 'img' / prevfn))
    img = cv2.imread(str(datapath.parent / 'train' / 'img' / curfn))

    previmg_bbox, previmg_mask, previmg_bbox_seg = [previmg.copy() for _ in range(3)]
    img_bbox, img_mask, img_bbox_seg = [img.copy() for _ in range(3)]
    
    track_ids_prev = [ann['track_id'] for ann in anno if ann['image_id'] == img_id]
    track_ids_cur = [ann['track_id'] for ann in anno if ann['image_id'] == img_id+1]
    track_ids_both = [track_id_prev for track_id_prev in track_ids_prev if track_id_prev in track_ids_cur]
    
    # np.random.seed(4)
    colors = [tuple((255*np.random.random(3))) for _ in range(len(set(track_ids_prev + track_ids_cur)))]
    
    tool.update(height,width,colors,img_id)

    prevseg_bw = np.zeros((img.shape[:2]),dtype=np.uint8)
    seg_bw = np.zeros((img.shape[:2]),dtype=np.uint8)

    for idx,track_id in enumerate(track_ids_both):
        previmg_bbox,previmg_mask,previmg_bbox_mask,prevseg = tool.forward(previmg_bbox, previmg_mask, previmg_bbox_seg,track_id,frame_id=0,color=colors[idx])
        img_bbox,img_mask,img_bbox_mask,seg = tool.forward(img_bbox, img_mask, img_bbox_seg,track_id,frame_id=1,color=colors[idx])

        prevseg_bw += prevseg
        seg_bw += seg

    track_leave = [id for id in track_ids_prev if id not in track_ids_cur]

    for track_id in track_leave:
        previmg_bbox,previmg_mask,previmg_bbox_mask,prevseg = tool.forward(previmg_bbox, previmg_mask, previmg_bbox_seg,track_id,frame_id=0,color=colors[len(track_ids_both)])
        prevseg_bw += prevseg

    track_new = [id for id in track_ids_cur if id not in track_ids_prev]

    for track_id in track_new:
        img_bbox,img_mask,img_bbox_mask,seg = tool.forward(img_bbox, img_mask, img_bbox_seg,track_id,frame_id=1,color=None)
        seg_bw += seg

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
        


        
    if save_bbox:
        cv2.imwrite(str(datapath.parent / 'generator' / (curfn[:-8] + '_bbox.png')),np.concatenate((previmg_bbox,img_bbox),axis=1))

    if save_mask:
        cv2.imwrite(str(datapath.parent / 'generator' / (curfn[:-8] + '_mask.png')),np.concatenate((previmg_mask,img_mask),axis=1))

    if save_bbox:
        cv2.imwrite(str(datapath.parent / 'generator' / (curfn[:-8] + '_bbox_mask.png')),np.concatenate((previmg_bbox_mask,img_bbox_mask),axis=1))

    if save_mask_bw:
        cv2.imwrite(str(datapath.parent / 'generator' / (curfn[:-8] + '_mask_bw.png')),np.concatenate((prevseg_bw,seg_bw),axis=1))

    cv2.imwrite(str(datapath.parent / 'generator' / (curfn[:-8] + '_og_img.png')),np.concatenate((previmg,img),axis=1)[:,:,0])