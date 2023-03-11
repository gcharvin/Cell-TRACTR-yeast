import torch
from torch.utils.data import DataLoader, DistributedSampler
import sacred
import numpy as np
import cv2
from pathlib import Path
import sys

cell_trackformer = Path(__file__).parents[1] / 'src'
sys.path.insert(0,str(cell_trackformer))

from trackformer.datasets import build_dataset
import trackformer.util.misc as utils
from trackformer.util.misc import nested_dict_to_namespace

ex = sacred.Experiment('train')
ex.add_config('/projectnb/dunlop/ooconnor/object_detection/cell-trackformer/cfgs/train.yaml')
ex.add_named_config('deformable', '/projectnb/dunlop/ooconnor/object_detection/cell-trackformer/cfgs/train_deformable.yaml')

@ex.main
def load_config(_config, _run):
    """ We use sacred only for config loading from YAML files. """
    sacred.commands.print_config(_run)

config = ex.run_commandline().config
args = nested_dict_to_namespace(config)

dataset_train = build_dataset(split='train', args=args)

sampler_train = torch.utils.data.RandomSampler(dataset_train)

batch_sampler_train = torch.utils.data.BatchSampler(
    sampler_train, batch_size = 1, drop_last=True)

data_loader_train = DataLoader(
    dataset_train,
    batch_sampler=batch_sampler_train,
    collate_fn=utils.collate_fn,
    num_workers=0)

class Tool():
    def __init__(self,targets):
        self.targets = targets
        self.crop = [np.inf,np.inf,0,0]
        self.alpha = 0.3

    def update(self,height,width,colors):
        self.height = height
        self.width = width
        self.colors = colors

    def update_crop(self,bounding_box):
        x0,y0,w,h = bounding_box
        
        self.crop[0] = int(np.min((self.crop[0],x0-10)))
        self.crop[1] = int(np.min((self.crop[1],y0-10)))

        self.crop[2] = int(np.max((self.crop[2],x0+w+10)))
        self.crop[3] = int(np.max((self.crop[3],y0+h+10)))

    def forward(self,bounding_box,seg,img_bbox,img_mask,img_bbox_mask,color=None):

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

        self.update_crop(bounding_box)

        mask = seg.numpy()
        mask_color = np.repeat(mask[...,None],3,axis=-1)
        mask_color[mask_color[...,0]>0] = color
        img_mask[mask_color > 1] = img_mask[mask_color>1] * (1-self.alpha) + mask_color[mask_color>1]  * self.alpha
        img_bbox_mask[mask_color > 1] = img_bbox_mask[mask_color>1] * (1-self.alpha) + mask_color[mask_color>1]  * self.alpha

        return img_bbox,img_mask,img_bbox_mask, (mask*255).astype(np.uint8)

datapath = Path(args.output_dir)
(datapath.parent / 'generator').mkdir(exist_ok=True)
print(datapath.parent)
fps = (datapath.parent / 'generator').glob('*.png')
for fp in fps:
    fp.unlink()

save_mask = True
save_bbox = True
save_bbox_mask = True
save_mask_bw = True
crop = False

np.random.seed(1)

for i, (samples, targets) in enumerate(data_loader_train):
    tool = Tool(targets)
    
    height = targets[0]['size'][0]
    width = targets[0]['size'][1]
    
    for b in range(len(targets)):
        previmg = targets[b]['prev_target']['image'].permute(1,2,0).numpy()
        img = targets[b]['image'].permute(1,2,0).numpy()

        previmg = (previmg - np.min(previmg)) / np.ptp(previmg) * 255
        img = (img - np.min(img)) / np.ptp(img) * 255

        previmg_bbox, previmg_mask, previmg_bbox_seg = [previmg.copy() for _ in range(3)]
        img_bbox, img_mask, img_bbox_seg = [img.copy() for _ in range(3)]
        
        track_ids_cur = targets[b]['track_ids'].numpy()
        track_ids_prev = targets[b]['prev_target']['track_ids'].numpy()
        track_ids_both = [track_id_prev for track_id_prev in track_ids_prev if track_id_prev in track_ids_cur]
        
        np.random.seed(1)
        colors = [tuple((255*np.random.random(3))) for _ in range(len(set(list(np.unique(track_ids_prev)) + list(np.unique(track_ids_cur)))))]
        
        tool.update(height,width,colors)

        prevseg_bw = np.zeros((img.shape[:2]),dtype=np.uint8)
        seg_bw = np.zeros((img.shape[:2]),dtype=np.uint8)

        boxes = targets[b]['boxes']
        prevboxes = targets[b]['prev_target']['boxes']

        boxes[:,::2] = boxes[:,::2] * width
        boxes[:,1::2] = boxes[:,1::2] * height
        boxes[:,0] -= boxes[:,2] / 2
        boxes[:,1] -= boxes[:,3] / 2

        prevboxes[:,::2] = prevboxes[:,::2] * width
        prevboxes[:,1::2] = prevboxes[:,1::2] * height
        prevboxes[:,0] -= prevboxes[:,2] / 2
        prevboxes[:,1] -= prevboxes[:,3] / 2

        for idx,track_id in enumerate(track_ids_both):
            ind = np.where(track_ids_prev == track_id)[0][0]
            bounding_box = prevboxes[ind][:4]
            mask = targets[b]['prev_target']['masks'][ind,0]
            previmg_bbox,previmg_mask,previmg_bbox_mask,prevseg = tool.forward(bounding_box,mask,previmg_bbox, previmg_mask, previmg_bbox_seg,color=colors[idx])
            prevseg_bw += prevseg

            ind = np.where(track_ids_cur == track_id)[0][0]
            bounding_box = boxes[ind][:4]    
            mask = targets[b]['masks'][ind,0]
            img_bbox,img_mask,img_bbox_mask,seg = tool.forward(bounding_box,mask,img_bbox, img_mask, img_bbox_seg,color=colors[idx])
            seg_bw += seg

            if prevboxes[ind][-1] > 0:
                boudning_boxes = boxes[ind][4:]
                mask = targets[b]['masks'][ind,1]
                img_bbox,img_mask,img_bbox_mask,seg = tool.forward(bounding_box,mask,img_bbox, img_mask, img_bbox_seg,color=colors[idx])
                seg_bw += seg

        track_leave = [id for id in track_ids_prev if id not in track_ids_cur]

        for track_id in track_leave:
            ind = np.where(track_ids_prev == track_id)[0][0]
            bounding_box = prevboxes[ind][:4]
            mask = targets[b]['prev_target']['masks'][ind,0]
            previmg_bbox,previmg_mask,previmg_bbox_mask,prevseg = tool.forward(bounding_box,mask,previmg_bbox, previmg_mask, previmg_bbox_seg,color=None)
            prevseg_bw += prevseg

        track_new = [id for id in track_ids_cur if id not in track_ids_prev]

        for track_id in track_new:
            ind = np.where(track_ids_cur == track_id)[0][0]
            bounding_box = boxes[ind][:4]
            mask = targets[b]['masks'][ind,0]
            img_bbox,img_mask,img_bbox_mask,seg = tool.forward(bounding_box,mask,img_bbox, img_mask, img_bbox_seg,color=None)
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
            cv2.imwrite(str(datapath.parent / 'generator' / (f'Sample{i:06}_batch{b:01d}_bbox.png')),np.concatenate((previmg_bbox,img_bbox),axis=1))

        if save_mask:
            cv2.imwrite(str(datapath.parent / 'generator' / (f'Sample{i:06}_batch{b:01d}_mask.png')),np.concatenate((previmg_mask,img_mask),axis=1))

        if save_bbox:
            cv2.imwrite(str(datapath.parent / 'generator' / (f'Sample{i:06}_batch{b:01d}_bbox_mask.png')),np.concatenate((previmg_bbox_mask,img_bbox_mask),axis=1))

        if save_mask_bw:
            cv2.imwrite(str(datapath.parent / 'generator' / (f'Sample{i:06}_batch{b:01d}_mask_bw.png')),np.concatenate((prevseg_bw,seg_bw),axis=1))

        cv2.imwrite(str(datapath.parent / 'generator' / (f'Sample{i:06}_batch{b:01d}_og_img.png')),np.concatenate((previmg,img),axis=1))

