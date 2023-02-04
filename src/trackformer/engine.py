# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable
import numpy as np
import PIL
import ffmpeg
import re
import torch
import cv2
from tqdm import tqdm
from .util import misc as utils
from .datasets.transforms import Normalize,ToTensor,Compose



def calc_loss_for_training_methods(training_method:str,
                                   outputs,
                                   groups,
                                   targets,
                                   criterion,
                                   epoch,
                                   args):
    outputs_TM = {}
    outputs_TM['aux_outputs'] = [{} for _ in range(len(outputs['aux_outputs']))]

    groups.append(groups[-1] + targets[0][training_method]['num_queries'])

    outputs_TM = utils.split_outputs(outputs,groups[-2:],outputs_TM,update_masks=args.masks)

    targets_TM = [target[training_method] for target in targets]

    if epoch > args.epoch_to_start_using_flexible_divisions:
        targets_TM = utils.update_early_or_late_track_divisions(outputs_TM,targets_TM)

    loss_dict_TM = criterion(outputs_TM, targets_TM)

    return outputs_TM, loss_dict_TM, groups

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loaders: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, args, num_plots=10, interval = 50):
    dataset = 'train'
    model.train()
    criterion.train()

    if epoch < 10:
        tm_threshold = 0.4
    elif epoch < 20:
        tm_threshold = 0.3
    elif epoch < 40:
        tm_threshold = 0.25
    else:
        tm_threshold = 0.2

    ids = np.random.randint(0,len(data_loaders[0]),num_plots)
    ids = np.concatenate((ids,[0]))

    metrics_dict = {}

    for i,((prev_prev_samples,prev_prev_targets), (prev_cur_samples,prev_cur_targets), (prev_samples,prev_targets), (cur_samples,cur_targets), (fut_prev_samples,fut_prev_targets), (fut_samples,fut_targets)) in enumerate(zip(*data_loaders)):
        samples = cur_samples
        targets = cur_targets
        targets_og = [{},{}]

        for t,target in enumerate(targets):
            target['cur_target'] = cur_targets[t].copy()
        
        for t,target in enumerate(targets):
    
            targets_og[t]['boxes'] = target['boxes'].to(args.device).clone()
            targets_og[t]['prev_boxes'] = prev_targets[t]['boxes'].to(args.device).clone()
            targets_og[t]['prev_prev_boxes'] = prev_prev_targets[t]['boxes'].to(args.device).clone()
            targets_og[t]['fut_boxes'] = fut_targets[t]['boxes'].to(args.device).clone()

            if 'masks' in target:
                targets_og[t]['masks'] = target['masks'].to(args.device).clone()
                targets_og[t]['prev_masks'] = prev_targets[t]['masks'].to(args.device).clone()
                targets_og[t]['prev_prev_masks'] = prev_prev_targets[t]['masks'].to(args.device).clone()
                targets_og[t]['fut_masks'] = fut_targets[t]['masks'].to(args.device).clone()

            target['prev_prev_target'] = prev_prev_targets[t]
            target['prev_prev_image'] = prev_prev_samples.tensors[t]

            target['prev_cur_target'] = prev_cur_targets[t]
            target['prev_cur_image'] = prev_cur_samples.tensors[t]

            target['prev_target'] = prev_targets[t]
            target['prev_image'] = prev_samples.tensors[t]

            target['fut_prev_target'] = fut_prev_targets[t]
            target['fut_prev_image'] = fut_prev_samples.tensors[t]

            target['fut_target'] = fut_targets[t]
            target['fut_image'] = fut_samples.tensors[t]

            assert target['image_id'] == prev_prev_targets[t]['image_id']
            assert prev_targets[t]['image_id'] == fut_prev_targets[t]['image_id']

        samples = samples.to(device)
        targets = [utils.nested_dict_to_device(t, device) for t in targets]

        outputs, targets, features, memory, hs, mask_features, prev_outputs = model(samples,targets,tm_threshold=tm_threshold,epoch=epoch)

        training_methods = outputs['training_methods'] # group_object, dn_object, dn_track

        meta_data = {}
    
        groups = [0, targets[0]['track_queries_mask'].shape[0]]

        for training_method in training_methods:
            meta_data[training_method] = {}
            outputs_TM, loss_dict_TM, groups = calc_loss_for_training_methods(training_method, outputs, groups, targets, criterion, epoch, args)

            meta_data[training_method]['outputs'] = outputs_TM
            meta_data[training_method]['loss_dict'] = loss_dict_TM

        outputs = utils.split_outputs(outputs,groups[:2],new_outputs=None,update_masks=args.masks)

        if epoch > args.epoch_to_start_using_flexible_divisions:
            targets = utils.update_early_or_late_track_divisions(outputs,targets)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        loss_dict_keys = list(loss_dict.keys())

        for training_method in training_methods:
            for loss_dict_key in loss_dict_keys:
                if loss_dict_key in ['loss_ce_enc','loss_bbox_enc','loss_giou_enc','loss_mask_enc','loss_dice_enc']: # enc loss only calculated once since dn_track / dn_object will not affect
                    continue
                assert (loss_dict_key + '_' + training_method) in weight_dict.keys()
                loss_dict[loss_dict_key + '_' + training_method] = meta_data[training_method]['loss_dict'][loss_dict_key] 

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        loss_dict['loss'] = losses

        if not math.isfinite(losses.item()):
            print(f"Loss is {losses.item()}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if args.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
        optimizer.step()

        if i == 0:
            lr = np.zeros((1,len(optimizer.param_groups)))
            for p,param_group in enumerate(optimizer.param_groups):
                lr[0,p] = param_group['lr']

        # Compute the segmentation and tracking metrics
        cls_threshold = 0.5
        iou_threshold = 0.75
        if 'track_query_match_ids' in targets[0]:
            acc_dict = utils.calc_track_acc(outputs,targets,cls_thresh=cls_threshold,iou_thresh=iou_threshold)
        else:
            acc_dict = utils.calc_bbox_acc(outputs,targets,cls_thresh=cls_threshold,iou_thresh=iou_threshold)

        metrics_dict = utils.update_metrics_dict(metrics_dict,acc_dict,loss_dict,weight_dict,i,lr)

        dict_shape = metrics_dict['loss_ce'].shape
        for metrics_dict_key in metrics_dict.keys():
            if metrics_dict_key == 'lr':
                continue
            assert metrics_dict[metrics_dict_key].shape[:2] == dict_shape, 'Metrics needed to be added per epoch'


        if i in ids and (epoch % 5 == 0 or epoch == 1):
            utils.plot_results(outputs, prev_outputs, targets,samples.tensors, targets_og, args.output_dir, train=True, filename = f'Epoch{epoch:03d}_Step{i:06d}.png', meta_data=meta_data)

        if i > 0 and i % interval == 0:
            utils.display_loss(metrics_dict,i,len(data_loaders[0]),epoch=epoch,dataset=dataset)

    utils.save_metrics_pkl(metrics_dict,args.output_dir,dataset=dataset,epoch=epoch)  





@torch.no_grad()
def evaluate(model, criterion, data_loaders, device, output_dir: str, 
             args, epoch: int = None, train=False, interval=50):
    model.eval()
    criterion.eval()
    dataset = 'val'
    num_plots = 10
    ids = np.random.randint(0,len(data_loaders[0]),num_plots)
    ids = np.concatenate((ids,[0]))

    if epoch < 10:
        tm_threshold = 0.4
    elif epoch < 20:
        tm_threshold = 0.3
    elif epoch < 40:
        tm_threshold = 0.25
    else:
        tm_threshold = 0.2

    metrics_dict = {}
    for i,((prev_prev_samples,prev_prev_targets), (prev_cur_samples,prev_cur_targets), (prev_samples,prev_targets), (cur_samples,cur_targets), (fut_prev_samples,fut_prev_targets), (fut_samples,fut_targets)) in enumerate(zip(*data_loaders)):
        
        samples = cur_samples
        targets = cur_targets

        targets_og = [{},{}]

        for t,target in enumerate(targets):
            target['cur_target'] = cur_targets[t].copy()

        for t,target in enumerate(targets):

            targets_og[t]['boxes'] = target['boxes'].to(args.device).clone()
            targets_og[t]['prev_boxes'] = prev_targets[t]['boxes'].to(args.device).clone()
            targets_og[t]['prev_prev_boxes'] = prev_prev_targets[t]['boxes'].to(args.device).clone()
            targets_og[t]['fut_boxes'] = fut_targets[t]['boxes'].to(args.device).clone()

            if 'masks' in target:
                targets_og[t]['masks'] = target['masks'].to(args.device).clone()
                targets_og[t]['prev_masks'] = prev_targets[t]['masks'].to(args.device).clone()
                targets_og[t]['prev_prev_masks'] = prev_prev_targets[t]['masks'].to(args.device).clone()
                targets_og[t]['fut_masks'] = fut_targets[t]['masks'].to(args.device).clone()

            target['prev_prev_target'] = prev_prev_targets[t]
            target['prev_prev_image'] = prev_prev_samples.tensors[t]

            target['prev_cur_target'] = prev_cur_targets[t]
            target['prev_cur_image'] = prev_cur_samples.tensors[t]

            target['prev_target'] = prev_targets[t]
            target['prev_image'] = prev_samples.tensors[t]

            target['fut_prev_target'] = fut_prev_targets[t]
            target['fut_prev_image'] = fut_prev_samples.tensors[t]

            target['fut_target'] = fut_targets[t]
            target['fut_image'] = fut_samples.tensors[t]

            assert target['image_id'] == prev_prev_targets[t]['image_id']
            assert prev_targets[t]['image_id'] == fut_prev_targets[t]['image_id']
        

        samples = samples.to(device)
        targets = [utils.nested_dict_to_device(t, device) for t in targets]

        outputs, targets, features, memory, hs, mask_features, prev_outputs = model(samples,targets,tm_threshold=tm_threshold,epoch=epoch)

        training_methods = outputs['training_methods'] # group_object, dn_object, dn_track

        meta_data = {}

        groups = [0, targets[0]['track_queries_mask'].shape[0]]

        for training_method in training_methods:
            meta_data[training_method] = {}
            outputs_TM, loss_dict_TM, groups = calc_loss_for_training_methods(training_method, outputs, groups, targets, criterion, epoch, args)

            meta_data[training_method]['outputs'] = outputs_TM
            meta_data[training_method]['loss_dict'] = loss_dict_TM

        outputs = utils.split_outputs(outputs,groups[:2],new_outputs=None,update_masks=args.masks)

        if epoch > args.epoch_to_start_using_flexible_divisions:
            targets = utils.update_early_or_late_track_divisions(outputs,targets)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        loss_dict_keys = list(loss_dict.keys())

        for training_method in training_methods:
            for loss_dict_key in loss_dict_keys:
                if loss_dict_key in ['loss_ce_enc','loss_bbox_enc','loss_giou_enc','loss_mask_enc','loss_dice_enc']: # enc loss only calculated once since dn_track / dn_object will not affect
                    continue
                loss_dict[loss_dict_key + '_' + training_method] = meta_data[training_method]['loss_dict'][loss_dict_key] 

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        loss_dict['loss'] = losses

        # Compute the segmentation and tracking metrics
        cls_threshold = 0.5
        iou_threshold = 0.75
        if 'track_query_match_ids' in targets[0]:
            acc_dict = utils.calc_track_acc(outputs,targets,cls_thresh=cls_threshold,iou_thresh=iou_threshold)
        else:
            acc_dict = utils.calc_bbox_acc(outputs,targets,cls_thresh=cls_threshold,iou_thresh=iou_threshold)

        metrics_dict = utils.update_metrics_dict(metrics_dict,acc_dict,loss_dict,weight_dict,i)

        if i in ids and (epoch % 5 == 0 or epoch == 1) and train:
            utils.plot_results(outputs, prev_outputs, targets,samples.tensors, targets_og, savepath = output_dir, train=False, filename = f'Epoch{epoch:03d}_Step{i:06d}.png', meta_data=meta_data)

        if i > 0 and i % interval == 0:
            utils.display_loss(metrics_dict,i,len(data_loaders[0]),epoch=epoch,dataset=dataset)

    utils.save_metrics_pkl(metrics_dict,args.output_dir,dataset=dataset,epoch=epoch)  



@torch.no_grad()
class pipeline():
    def __init__(self,model, fps, device, output_dir, args, track=True, use_NMS=False, display_masks=False):
        self.model = model
        self.model.tracking()

        self.use_NMS = use_NMS
        self.masks = display_masks and args.masks

        self.predictions_folder = 'predictions' 
        
        if self.use_NMS:
            self.predictions_folder += '_NMS'



        self.output_dir = output_dir
        (self.output_dir / self.predictions_folder).mkdir(exist_ok=True)

        self.normalize = Compose([
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.threshold = 0.5
        self.target_size = (256,32)
        self.num_queries = args.num_queries
        self.device = device
        self.use_dab = args.use_dab

        self.write_video = True
        self.track = track

        if self.track:
            self.colors = np.array([tuple((255*np.random.random(3))) for _ in range(1000)]) # Assume max 1000 cells in one chamber
        else:
            self.colors = np.array([tuple((np.zeros(3))) for _ in range(1000)])

        if args.two_stage:
            (self.output_dir / self.predictions_folder / 'enc_outputs').mkdir(exist_ok=True)

        self.oq_div = True # Can object queries detect divisions

        self.display_decoder_aux = True

        if self.display_decoder_aux:
            (self.output_dir / self.predictions_folder / 'decoder_bbox_outputs').mkdir(exist_ok=True)
            (self.output_dir / self.predictions_folder / 'ref_pts_outputs').mkdir(exist_ok=True)
            self.num_decoder_frames = 1

        self.display_object_query_boxes = True

        if self.display_object_query_boxes:
            self.query_box_locations = [np.zeros((1,4)) for i in range(args.num_queries)]

        self.fps_split = [[]]
        self.videoname_list = []

        for fidx,fp in enumerate(fps):
            frame_nb = re.findall('\d+',fp.stem)[-1]
            filename = fp.name.replace(frame_nb+'.png','')

            if fidx == 0 or filename != old_filename:
                self.videoname_list.append(filename)

            if fidx > 0 and filename != old_filename:
                self.fps_split.append([fp])
            else:
                self.fps_split[-1].append(fp)

            old_filename = filename

        max_len = max([len(fp_split) for fp_split in self.fps_split])

        self.color_stack = np.zeros((max_len,self.target_size[0],self.target_size[1]*len(self.videoname_list),3))

    def update_query_box_locations(self,pred_boxes,keep,keep_div):
        # This is only used to display to reference points for object queries that are detected
        # Get x,y location of all detected object queries 
        all_oq_boxes = pred_boxes[-self.num_queries:].cpu().numpy()
        oq_boxes = all_oq_boxes[keep[-self.num_queries:],:4]

        if keep_div[-self.num_queries:].sum():
            oq_div_boxes = all_oq_boxes[keep_div[-self.num_queries:],4:]
            oq_boxes = np.concatenate((oq_boxes,oq_div_boxes),axis=0)

        oq_boxes[:,1::2] = np.clip(oq_boxes[:,1::2] * self.target_size[0], 0, self.target_size[0])
        oq_boxes[:,0::2] = np.clip(oq_boxes[:,::2] * self.target_size[1], 0, self.target_size[1])

        oq_indices = keep[-self.num_queries:].nonzero()[0]
        for oq_ind, oq_box in zip(oq_indices,oq_boxes):
            self.query_box_locations[oq_ind] = np.append(self.query_box_locations[oq_ind], oq_box[None],axis=0)


    def split_up_divided_cells(self):

        self.div_track = -1 * np.ones((len(self.track_indices) + len(self.div_indices)),dtype=np.int) # keeps track of which cells were the result of cell division

        for div_ind in self.div_indices:
            ind = np.where(self.track_indices==div_ind)[0][0]

            self.max_cellnb += 1             
            self.cells = np.concatenate((self.cells[:ind+1],[self.max_cellnb],self.cells[ind+1:])) # add daughter cellnb after mother cellnb
            self.track_indices = np.concatenate((self.track_indices[:ind],self.track_indices[ind:ind+1],self.track_indices[ind:])) # order doesn't matter here since they are the same track indices

            self.div_track[ind:ind+2] = div_ind # we add 1 because div_track is set as np.zeros so an ind of 0 would blend in with the starting point

        self.new_cells = self.cells == 0

        if 0 in self.cells:
            self.max_cellnb += 1   
            self.cells[self.cells==0] = np.arange(self.max_cellnb,self.max_cellnb+sum(self.cells==0),dtype=np.int)
            
            assert np.max(self.cells) >= self.max_cellnb
            self.max_cellnb = np.max(self.cells)

    def update_div_boxes(self,boxes,masks=None):
        # boxes where div_indices were repeat; now they need to be rearrange because only the first box is sent to decoder
        self.unique_divs = np.unique(self.div_track[self.div_track != -1])
        for unique_div in self.unique_divs:
            div_ids = (self.div_track == unique_div).nonzero()[0]
            boxes[div_ids[1],:4] = boxes[div_ids[0],4:]
            # e.g. [15, 45, 16, 22, 14, 62, 15, 20]  -->  [15, 45, 16, 22, 14, 62, 15, 20]  -->  [15, 45, 16, 22]  --> fed to decoder
            #      [15, 45, 16, 22, 14, 62, 15, 20]  -->  [14, 62, 15, 20, 14, 62, 15, 20]  -->  [14, 62, 15, 20]  --> fed to decoder

            if masks is not None:
                masks[div_ids[1],:1] = masks[div_ids[0],1:] 

        return boxes, masks

    def NMS(self, pred_boxes, keep, keep_div):

        keep, keep_div = torch.tensor(keep), torch.tensor(keep_div)
        ind_keep = torch.where(keep == True)[0]
        track_keep = [i_keep for i_keep in ind_keep if i_keep < len(keep) - self.num_queries]
        new_keep = [i_keep for i_keep in ind_keep if i_keep >= len(keep) - self.num_queries]
        ind_div_keep = torch.where(keep_div == True)[0]
        track_div_keep = [i_keep for i_keep in ind_div_keep if i_keep < len(keep) - self.num_queries]

        track_boxes = torch.cat((pred_boxes[track_keep,:4],pred_boxes[track_div_keep,4:]),axis=0)

        for ind in new_keep:
            box = pred_boxes[ind,:4]

            for track_box in track_boxes:
                iou = utils.calc_iou(box,track_box)

                if iou > 0.75:
                    keep[ind] = False
                    break

        return keep.numpy(), keep_div.numpy()

    def forward(self):
        for r,fps_ROI in enumerate(self.fps_split):
            print(self.videoname_list[r])
            print(f'video {r+1}/{len(self.fps_split)}')
            self.max_cellnb = 0
            targets = [{}]
            prev_features = None
            
            if self.display_decoder_aux:
                random_nbs = np.random.choice(len(fps_ROI),self.num_decoder_frames)
                random_nbs = np.concatenate((random_nbs,random_nbs+1)) # so we can see two consecutive frames

            for i, fp in enumerate(tqdm(fps_ROI)):

                img = PIL.Image.open(fp,mode='r').resize((self.target_size[1],self.target_size[0])).convert('RGB')
                previmg = PIL.Image.open(fps_ROI[i-1],mode='r').resize((self.target_size[1],self.target_size[0])).convert('RGB') if i > 0 else None # saved for analysis later

                samples = self.normalize(img)[0][None]
                samples = samples.to(self.device)

                if not self.track: # If object detction only, we don't feed information from the previous frame
                    targets = [{}]
                    self.max_cellnb = 0
                    prev_features = None

                outputs, targets, prev_features, memory, hs, mask_features, prev_outputs = self.model(samples,targets=targets,prev_features=prev_features)

                pred_logits = outputs['pred_logits'][0].sigmoid().detach().cpu().numpy()
                pred_boxes = outputs['pred_boxes'][0].detach()

                keep = (pred_logits[:,0] > self.threshold)
                keep_div = (pred_logits[:,1] > self.threshold)
                keep_div[-self.num_queries:] = False # disregard any divisions predicted by object queries; model should have learned not to do this anyways

                if self.use_NMS and len(keep) > self.num_queries and keep[-self.num_queries:].sum() > 0:
                    keep, keep_div = self.NMS(pred_boxes, keep, keep_div)

                if i > 0:
                    prevcells = np.copy(self.cells)

                self.cells = np.zeros((keep.sum()),dtype=np.int)
                masks = None

                # If no objects are detected, skip
                if self.display_object_query_boxes and keep[-self.num_queries:].sum() > 0:
                    self.update_query_box_locations(pred_boxes,keep,keep_div)

                if sum(keep) > 0:
                    self.track_indices = keep.nonzero()[0] # Get query indices (object or track) where a cell was detected / tracked; this is used to create track queries for next  frame
                    self.div_indices = keep_div.nonzero()[0] # Get track query indices where a cell division was tracked; object queries should not be able to detect divisions

                    if pred_logits.shape[0] > self.num_queries: # If track queries are fed to the model
                        tq_keep = pred_logits[:len(prevcells),0] > self.threshold
                        self.cells[:sum(tq_keep)] = prevcells[tq_keep]
                    else:
                        prevcells = None

                    self.split_up_divided_cells()

                    targets[0]['track_query_hs_embeds'] = outputs['hs_embed'][0,self.track_indices] # For div_indices, hs_embeds will be the same; no update
                    boxes = pred_boxes[self.track_indices] # For div_indices, the boxes will be repeated and will properly updated below

                    if self.masks:
                        masks = outputs['pred_masks'][0,self.track_indices].sigmoid().detach()
                    
                    boxes,masks = self.update_div_boxes(boxes,masks)
                    boxes = boxes[:,:4] # only one cell is tracked at a time
        
                    if masks is not None: # gets rid of the division mask which we rearrange above
                        masks = masks[:,0]

                    targets[0]['track_query_boxes'] = boxes

                    if i == 0: # self.new_cells is used to visually highlight errors specifically for the mother machine. This is because no new cells will ever appear so I know this is an erorr
                        self.new_cells = None

                else:
                    self.track_indices
                    self.div_track = None
                    boxes = None
                    self.new_cells = None
                    prevcells = None

                assert boxes.shape[0] == len(self.cells)

                if self.track:
                    color_frame = utils.plot_tracking_results(img,boxes,masks,self.colors[self.cells-1],self.cells,self.div_track,self.new_cells,self.track)
                else:
                    color_frame = utils.plot_tracking_results(img,boxes,masks,self.colors[:len(self.cells)],self.cells,self.div_track,None,self.track)

                self.color_stack[i,:,r*self.target_size[1]:(r+1)*self.target_size[1]] = color_frame         

                if self.display_decoder_aux and i in random_nbs:

                    if 'enc_outputs' in outputs:
                        enc_frame = np.array(img).copy()
                        enc_outputs = outputs['enc_outputs']
                        enc_pred_logits = enc_outputs['pred_logits']
                        enc_pred_boxes = enc_outputs['pred_boxes']

                        logits_topk, ind_topk = torch.topk(enc_pred_logits[0,:,0].sigmoid(),self.num_queries)
                        boxes_topk = enc_pred_boxes[0,ind_topk]

                        t0,t1,t2,t3 = 0.1,0.3,0.5,0.8
                        boxes_list = []
                        boxes_list.append(boxes_topk[logits_topk < t0])
                        boxes_list.append(boxes_topk[(logits_topk > t0) * (logits_topk < t1)])
                        boxes_list.append(boxes_topk[(logits_topk > t1) * (logits_topk < t2)])
                        boxes_list.append(boxes_topk[(logits_topk > t2) * (logits_topk < t3)])
                        boxes_list.append(boxes_topk[logits_topk > t3])

                        enc_frames = []
                        for boxes in boxes_list:
                            enc_frame = np.array(img).copy()
                            enc_frames.append(utils.plot_tracking_results(enc_frame,boxes,None,self.colors,None,None,None,self.track))
                        
                        enc_frames = np.concatenate((enc_frames),axis=1)

                        cv2.imwrite(str(self.output_dir / self.predictions_folder / 'enc_outputs' / (f'encoder_frame_{fp.name}')),enc_frames)


                    references = outputs['references'].detach()

                    aux_outputs = outputs['aux_outputs'] # output from first 5 layers of decoder
                    aux_outputs = [{'pred_boxes':references[0]}] + aux_outputs # add the initial anchors / reference points
                    aux_outputs.append({'pred_boxes':outputs['pred_boxes'].detach()}) # add the last layer of the decoder which is the final prediction

                    img = np.array(img)

                    colors = self.colors[self.cells-1] if self.track else self.colors[:len(self.cells)] 

                    cells_exit_ids = torch.tensor([[cidx,c] for cidx,c in enumerate(prevcells) if c not in self.cells]) if prevcells is not None else None

                    if self.track:
                        previmg_copy = previmg.copy()
                    for a,aux_output in enumerate(aux_outputs):
                        all_boxes = aux_output['pred_boxes'][0].detach()
                        img_copy = img.copy()
                        
                        if len(self.track_indices) > 0:
                            
                            if a > 0: # initial reference points are all single boxes; this applies to the outputs of decoder
                                track_boxes = all_boxes[self.track_indices]
                                for unique_div in self.unique_divs:
                                    div_ids = (self.div_track == unique_div).nonzero()[0]
                                    track_boxes[div_ids[1],:4] = track_boxes[div_ids[0],4:]
                                track_boxes = track_boxes[:,:4]
                                div_track = self.div_track
                                box_colors = colors
                                new_cells = self.new_cells if self.track else None

                            else:
                                track_boxes = all_boxes[np.unique(self.track_indices[self.track_indices < len(keep) - self.num_queries])]
                                box_colors = self.colors[prevcells-1] if prevcells is not None else colors
                                div_track = np.ones((track_boxes.shape[0])) * -1
                                new_cells = None

                                if self.track:
                                    assert track_boxes.shape[0] <= box_colors.shape[0]
                                    previmg_anchor_boxes = utils.plot_tracking_results(previmg_copy,track_boxes,None,box_colors,prevcells,div_track,new_cells,self.track)

                            if a == 0 and not self.use_dab: # if x,y reference points are used
                                for ridx in range(track_boxes.shape[0]):
                                    x,y = track_boxes[ridx]
                                    img_copy = cv2.circle(img_copy, (int(x*self.target_size[1]),int(y*self.target_size[0])), radius=1, color=colors[ridx], thickness=-1)
                            else:
                                img_copy = utils.plot_tracking_results(img_copy,track_boxes,None,box_colors,self.cells,div_track,new_cells,self.track)
                        
                        if a == 0 and self.track:
                            decoder_frame = np.concatenate((previmg,img,previmg_anchor_boxes,img_copy),axis=1)
                        elif a == 0:
                            decoder_frame = np.concatenate((img,img_copy),axis=1)
                        else:
                            decoder_frame = np.concatenate((decoder_frame,img_copy),axis=1)

                        # Plot all predictions regardless of cls label
                        if a == len(aux_outputs) - 1:
                            img_copy = img.copy()
                            
                            color_queries = np.array([(np.array([0.,0.,255.])) for _ in range(self.num_queries)])

                            if cells_exit_ids is not None and cells_exit_ids.shape[0] > 0: # Plot the track query that left the chamber
                                boxes_exit = all_boxes[cells_exit_ids[:,0],:4]
                                boxes = torch.cat((all_boxes[-self.num_queries:,:4],boxes_exit,track_boxes))
                                colors_prev = self.colors[cells_exit_ids[:,1] - 1] 
                                colors_prev = colors_prev[None] if colors_prev.ndim == 1 else colors_prev
                                
                                all_colors = np.concatenate((color_queries,colors_prev,colors),axis=0)
                                # div_track_all = np.ones((all_boxes.shape[0] + len(self.div_indices))) * -1 # all boxes does not contain div boxes separated
                                div_track_all = np.ones((boxes.shape[0])) * -1 # all boxes does not contain div boxes separated
                                div_track_all[-len(track_boxes):] = self.div_track

                                # print(f'boxes: {len(boxes)}\nall_boxes: {len(all_boxes)}\ntrack_boxes: {len(track_boxes)}\nboxes_exit: {len(boxes_exit)}\ndiv_indices: {len(self.div_indices)}\n{self.div_indices}')
                                assert len(div_track_all) == len(boxes)

                            else: # all cells / track queries stayed in the chamber
                                boxes = torch.cat((all_boxes[-self.num_queries:,:4],track_boxes))
                                all_colors = np.concatenate((color_queries,colors),axis=0)
                                div_track_all = np.concatenate((np.ones((self.num_queries))*-1,self.div_track))

                                assert len(div_track_all) == len(boxes)

                            new_cell_thickness = np.zeros_like(div_track_all).astype(bool)
                            new_cell_thickness[self.num_queries:] = True # set all track queries with a thickened boudning box so it's easier to see
                            img_final_box = utils.plot_tracking_results(img_copy,boxes,None,all_colors,self.cells,div_track_all,new_cell_thickness,self.track)

                            color_red = np.array([(np.array([0.,0.,255.])) for _ in range(boxes.shape[0])]) # plot all bounding boxes as red
                            img_final_all_box = utils.plot_tracking_results(img_copy,boxes,None,color_red,self.cells,None,None,self.track)
                            
                            decoder_frame = np.concatenate((decoder_frame,img_final_box,img_final_all_box),axis=1)

                    method = 'object_detection' if not self.track else 'track'
                    cv2.imwrite(str(self.output_dir / self.predictions_folder / 'decoder_bbox_outputs' / (f'{method}_decoder_frame_{fp.name}')),decoder_frame)

                    img_ref_pts_init_object = np.copy(np.array(img))
                    img_ref_pts_init_track = np.copy(np.array(img))
                    if self.track:
                        previmg_ref_pts_init_track = np.copy(np.array(previmg))
                    img_ref_pts_init_all = np.copy(np.array(img))
                    img_ref_pts_final_all = np.copy(np.array(img))

                    for index,ref in enumerate(references[:,0]): # batch size of 1
                        img_ref_pts_update = np.copy(np.array(img))

                        for ridx in range(len(ref)):
                            if self.use_dab:
                                x,y,_,_ = ref[ridx]
                            else:
                                x,y = ref[ridx]
                            
                            if self.track:
                                if ridx < len(prevcells) and prevcells[ridx] in self.cells: # cell tracked from previous frame
                                    color = self.colors[prevcells[ridx] - 1]
                                    radius = 2
                                elif ridx < len(prevcells) and prevcells[ridx] not in self.cells: # track query not detected
                                    color = (255,255,255)
                                    radius = 2
                                elif ridx >= len(prevcells) and pred_logits[ridx,0] > self.threshold: # object queries detected
                                    color = (255,0,0)
                                    radius = 1
                                elif ridx >= len(prevcells) and pred_logits[ridx,0] < self.threshold: # object queries not used
                                    color = (0,0,255)
                                    radius = 1
                            else:
                                if pred_logits[ridx,0] > self.threshold: # object queries detected
                                    color = (255,0,0)
                                    radius = 1
                                elif pred_logits[ridx,0] < self.threshold: # object queries not used
                                    color = (0,0,255)
                                    radius = 1

                            img_ref_pts_update = cv2.circle(img_ref_pts_update, (int(x*self.target_size[1]),int(y*self.target_size[0])), radius=radius, color=color, thickness=-1)

                            if index == 0:
                                if self.track and prevcells is not None and ridx < len(prevcells):
                                    img_ref_pts_init_track = cv2.circle(img_ref_pts_init_track, (int(x*self.target_size[1]),int(y*self.target_size[0])), radius=radius, color=color, thickness=-1)
                                    previmg_ref_pts_init_track = cv2.circle(previmg_ref_pts_init_track, (int(x*self.target_size[1]),int(y*self.target_size[0])), radius=radius, color=color, thickness=-1)
                                elif prevcells is None or ridx >= len(prevcells):
                                    img_ref_pts_init_object = cv2.circle(img_ref_pts_init_object, (int(x*self.target_size[1]),int(y*self.target_size[0])), radius=radius, color=color, thickness=-1)

                                img_ref_pts_init_all = cv2.circle(img_ref_pts_init_all, (int(x*self.target_size[1]),int(y*self.target_size[0])), radius=1, color=(0,0,255), thickness=-1)

                            if index == len(references) - 1:
                                img_ref_pts_final_all = cv2.circle(img_ref_pts_final_all, (int(x*self.target_size[1]),int(y*self.target_size[0])), radius=1, color=(0,0,255), thickness=-1)

                        ref_frames = np.concatenate((ref_frames,img_ref_pts_update),axis=1) if index > 0 else img_ref_pts_update

                    if self.track:
                        ref_frames = np.concatenate((img,img_ref_pts_init_all,previmg_ref_pts_init_track,img_ref_pts_init_track,img_ref_pts_init_object,ref_frames,img_ref_pts_final_all),axis=1)
                    else:
                        ref_frames = np.concatenate((img,img_ref_pts_init_all,ref_frames,img_ref_pts_final_all),axis=1)

                    cv2.imwrite(str(self.output_dir / self.predictions_folder / 'ref_pts_outputs' / (f'{method}_ref_pts_{fp.name}')),ref_frames)
                    

        if self.write_video:
            crf = 20
            verbose = 1
            method = 'track' if self.track else 'object_detection'
            name_mask = 'mask_' if self.masks else ''
            filename = self.output_dir / self.predictions_folder / (f'{self.videoname_list[r]}_{method}_{name_mask}video.mp4')
            print(filename)
            height, width, _ = self.color_stack[0].shape
            if height % 2 == 1:
                height -= 1
            if width % 2 == 1:
                width -= 1
            quiet = [] if verbose else ["-loglevel", "error", "-hide_banner"]
            process = (
                ffmpeg.input(
                    "pipe:",
                    format="rawvideo",
                    pix_fmt="rgb24",
                    s="{}x{}".format(width, height),
                    r=7,
                )
                .output(
                    str(filename),
                    pix_fmt="yuv420p",
                    vcodec="libx264",
                    crf=crf,
                    preset="veryslow",
                )
                .global_args(*quiet)
                .overwrite_output()
                .run_async(pipe_stdin=True)
            )

            # Write frames:
            for frame in self.color_stack:
                process.stdin.write(frame[:height, :width].astype(np.uint8).tobytes())

            # Close file stream:
            process.stdin.close()

            # Wait for processing + close to complete:
            process.wait()

        if self.display_object_query_boxes:

            scale = 8
            wspacer = 5 * scale
            hspacer = 20 * scale

            max_area = [np.max(boxes[:,2] * boxes[:,3]) for boxes in self.query_box_locations]
            num_boxes_used = np.sum(np.array(max_area) > 0)
            query_frames = np.ones((self.target_size[0]*scale + hspacer, (self.target_size[1]*scale + wspacer) * num_boxes_used,3),dtype=np.uint8) * 255
            where_boxes = np.where(np.array(max_area) > 0)[0]

            for j,ind in enumerate(where_boxes):
                img_empty = cv2.imread(str(self.output_dir.parents[1] / 'empty_chamber' / 'img.png'))
                img_empty = cv2.resize(img_empty,(self.target_size[1]*scale,self.target_size[0]*scale))
            
                for box in self.query_box_locations[ind][1:]:
                    img_empty = cv2.circle(img_empty, (int(box[0]*scale),int(box[1]*scale)), radius=1*scale, color=(255,0,0), thickness=-1)

                img_empty = np.concatenate((np.ones((hspacer,self.target_size[1]*scale,3),dtype=np.uint8)*255,img_empty),axis=0)
                shift = 5 if ind + 1 >= 10 else 12
                img_empty = cv2.putText(img_empty,f'{ind+1}',org=(shift*scale,15*scale),fontFace=cv2.FONT_HERSHEY_COMPLEX,fontScale=4,color=(0,0,0),thickness=4)
                query_frames[:,j*(self.target_size[1]*scale+wspacer): j*(self.target_size[1]*scale+wspacer) + self.target_size[1]*scale] = img_empty

            cv2.imwrite(str(self.output_dir / self.predictions_folder / (f'{method}_object_query_box_locations.png')),query_frames)
        


@torch.no_grad()
def print_worst(model, criterion, data_loaders_train, data_loaders_val, device, output_dir, args, track=True):
    model.eval()
    model._tracking = True
    criterion.eval()

    if track:
        save_folder = 'save_worst_predictions_track'
        tm_threshold = 0

        if args.use_prev_prev_frame:
            save_folder += '_prev_prev_frame_fix' 
    else:
        save_folder = 'save_worst_predcitions_object_det'
        tm_threshold = 1

    (output_dir / save_folder).mkdir(exist_ok=True)
    (output_dir / save_folder / 'train_outputs').mkdir(exist_ok=True)
    (output_dir / save_folder / 'eval_outputs').mkdir(exist_ok=True)

    (output_dir / save_folder / 'train_outputs' / 'standard').mkdir(exist_ok=True)
    (output_dir / save_folder / 'eval_outputs' / 'standard').mkdir(exist_ok=True)

    (output_dir / save_folder / 'train_outputs' / 'enc_outputs').mkdir(exist_ok=True)
    (output_dir / save_folder / 'eval_outputs' / 'enc_outputs').mkdir(exist_ok=True)


    for didx, data_loaders in enumerate([data_loaders_train,data_loaders_val]):
        store_loss = torch.zeros((len(data_loaders[0])))
        for idx,((prev_prev_samples,prev_prev_targets), (prev_cur_samples,prev_cur_targets), (prev_samples,prev_targets), (cur_samples,cur_targets), (fut_prev_samples,fut_prev_targets), (fut_samples,fut_targets)) in enumerate(tqdm(zip(*data_loaders))):

            samples = cur_samples
            targets = cur_targets
            targets_og = [{},{}]

            for t,target in enumerate(targets):
                target['cur_target'] = cur_targets[t].copy()
            
            for t,target in enumerate(targets):
        
                targets_og[t]['boxes'] = target['boxes'].to(args.device).clone()
                targets_og[t]['prev_boxes'] = prev_targets[t]['boxes'].to(args.device).clone()
                targets_og[t]['prev_prev_boxes'] = prev_prev_targets[t]['boxes'].to(args.device).clone()
                targets_og[t]['fut_boxes'] = fut_targets[t]['boxes'].to(args.device).clone()

                if 'masks' in target:
                    targets_og[t]['masks'] = target['masks'].to(args.device).clone()
                    targets_og[t]['prev_masks'] = prev_targets[t]['masks'].to(args.device).clone()
                    targets_og[t]['prev_prev_masks'] = prev_prev_targets[t]['masks'].to(args.device).clone()
                    targets_og[t]['fut_masks'] = fut_targets[t]['masks'].to(args.device).clone()

                target['prev_prev_target'] = prev_prev_targets[t]
                target['prev_prev_image'] = prev_prev_samples.tensors[t]

                target['prev_cur_target'] = prev_cur_targets[t]
                target['prev_cur_image'] = prev_cur_samples.tensors[t]

                target['prev_target'] = prev_targets[t]
                target['prev_image'] = prev_samples.tensors[t]

                target['fut_prev_target'] = fut_prev_targets[t]
                target['fut_prev_image'] = fut_prev_samples.tensors[t]

                target['fut_target'] = fut_targets[t]
                target['fut_image'] = fut_samples.tensors[t]

                assert target['image_id'] == prev_prev_targets[t]['image_id']
                assert prev_targets[t]['image_id'] == fut_prev_targets[t]['image_id']

            samples = samples.to(device)
            targets = [utils.nested_dict_to_device(t, device) for t in targets]

            outputs, targets, features, memory, hs, mask_features, prev_outputs = model(samples,targets,tm_threshold=tm_threshold)

            targets = utils.update_early_or_late_track_divisions(outputs,targets)

            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict

            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            store_loss[idx] = losses.item()


        worst_ind = torch.argsort(store_loss)[-50:]

        for idx,((prev_prev_samples,prev_prev_targets), (prev_cur_samples,prev_cur_targets), (prev_samples,prev_targets), (cur_samples,cur_targets), (fut_prev_samples,fut_prev_targets), (fut_samples,fut_targets)) in enumerate(zip(*data_loaders)):
            
            if idx not in worst_ind:
                continue

            samples = cur_samples
            targets = cur_targets
            targets_og = [{},{}]

            for t,target in enumerate(targets):
                target['cur_target'] = cur_targets[t].copy()
            
            for t,target in enumerate(targets):
        
                targets_og[t]['boxes'] = target['boxes'].to(args.device).clone()
                targets_og[t]['prev_boxes'] = prev_targets[t]['boxes'].to(args.device).clone()
                targets_og[t]['prev_prev_boxes'] = prev_prev_targets[t]['boxes'].to(args.device).clone()
                targets_og[t]['fut_boxes'] = fut_targets[t]['boxes'].to(args.device).clone()

                if 'masks' in target:
                    targets_og[t]['masks'] = target['masks'].to(args.device).clone()
                    targets_og[t]['prev_masks'] = prev_targets[t]['masks'].to(args.device).clone()
                    targets_og[t]['prev_prev_masks'] = prev_prev_targets[t]['masks'].to(args.device).clone()
                    targets_og[t]['fut_masks'] = fut_targets[t]['masks'].to(args.device).clone()

                target['prev_prev_target'] = prev_prev_targets[t]
                target['prev_prev_image'] = prev_prev_samples.tensors[t]

                target['prev_cur_target'] = prev_cur_targets[t]
                target['prev_cur_image'] = prev_cur_samples.tensors[t]

                target['prev_target'] = prev_targets[t]
                target['prev_image'] = prev_samples.tensors[t]

                target['fut_prev_target'] = fut_prev_targets[t]
                target['fut_prev_image'] = fut_prev_samples.tensors[t]

                target['fut_target'] = fut_targets[t]
                target['fut_image'] = fut_samples.tensors[t]

                assert target['image_id'] == prev_prev_targets[t]['image_id']
                assert prev_targets[t]['image_id'] == fut_prev_targets[t]['image_id']

            samples = samples.to(device)
            targets = [utils.nested_dict_to_device(t, device) for t in targets]

            outputs, targets, features, memory, hs, mask_features, prev_outputs = model(samples,targets,tm_threshold=tm_threshold)

            targets = utils.update_early_or_late_track_divisions(outputs,targets)

            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict

            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            if not np.round(losses.item(),3) == np.round(store_loss[idx],3):
                print(idx, np.round(losses.item(),3),np.round(store_loss[idx],3))

            utils.plot_results(outputs, prev_outputs, targets,samples.tensors, targets_og, args.output_dir / save_folder, train=True if didx == 0 else False, filename = f'Loss_{store_loss[idx]:06.2f}_ind{idx}_.png')



        
@torch.no_grad()
def print_attn_maps(model, fps, device, output_dir, args):

    model.tracking()
    (output_dir / 'predictions').mkdir(exist_ok=True)
    normalize = Compose([
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    target_size = (256,32)
    fp = fps[0]

    img = PIL.Image.open(fp,mode='r').resize((target_size[1],target_size[0])).convert('RGB')
    samples = normalize(img)[0][None]
    samples = samples.to(device)

    # use lists to store the outputs via up-values
    conv_features, enc_attn_weights, dec_attn_weights = [], [], []

    hooks = [
        model.backbone[-2].register_forward_hook(
            lambda self, input, output: conv_features.append(output)
        ),
        model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
            lambda self, input, output: enc_attn_weights.append(output)
        ),
        model.transformer.decoder.layers[-1].cross_attn.register_forward_hook(
            lambda self, input, output: dec_attn_weights.append(output)
        ),
    ]

    # propagate through the model
    outputs = model(samples)

    for hook in hooks:
        hook.remove()

    # don't need the list anymore
    conv_features = conv_features[0]
    enc_attn_weights = enc_attn_weights[0]
    dec_attn_weights = dec_attn_weights[0]
