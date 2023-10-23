# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable
import numpy as np
import PIL
import torch
import cv2
from pathlib import Path
from tqdm import tqdm
from .util import data_viz
from .util import misc as utils
from .util import box_ops
from .util import data_viz
from .datasets.transforms import Normalize,ToTensor,Compose
import ffmpeg
        

def calc_loss_for_training_methods(training_method:str,
                                   outputs,
                                   groups,
                                   targets,
                                   criterion,
                                   epoch,
                                   args):
    outputs_TM = {}
    outputs_TM['aux_outputs'] = [{} for _ in range(len(outputs['aux_outputs']))]

    # groups.append(groups[-1] + targets[0][training_method]['num_queries'])
    num_queries = targets[0][training_method]['num_queries']
    
    if training_method == 'cur_target':
        if 'enc_outputs' in outputs:
            outputs_TM['enc_outputs'] = outputs['enc_outputs']

    # outputs_TM = utils.split_outputs(outputs,groups[-2:],outputs_TM)
    outputs_TM = utils.split_outputs(outputs,num_queries,outputs_TM)

    loss_dict_TM = criterion(outputs_TM, targets, training_method, epoch)

    return outputs_TM, loss_dict_TM, groups

def train_one_epoch(model: torch.nn.Module, 
                    criterion: torch.nn.Module,
                    data_loader: Iterable, 
                    optimizer: torch.optim.Optimizer,
                    epoch: int, 
                    args,  
                    interval: int = 50):

    dataset = 'train'
    model.train()
    criterion.train()

    track_thresh = 0.1

    ids = np.random.randint(0,len(data_loader),args.num_plots)
    ids = np.concatenate((ids,[0]))

    metrics_dict = {}

    for i, (samples, targets) in enumerate(data_loader):

        track = torch.rand(1).item() > track_thresh
        samples = samples.to(args.device)
        targets = [utils.nested_dict_to_device(t, args.device) for t in targets]

        outputs, targets, _, _, _, prev_outputs = model(samples,targets,track,epoch=epoch)

        del _

        if prev_outputs is not None:
            for key in prev_outputs.keys():
                if 'pred' in key:
                    prev_outputs[key] = prev_outputs[key].cpu()

        training_methods =  outputs['training_methods'] # dn_object, dn_track, dn_enc

        meta_data = {}
        groups = [0]

        for training_method in training_methods:
            outputs_TM, loss_dict_TM, groups = calc_loss_for_training_methods(training_method, outputs, groups, targets, criterion, epoch, args)

            meta_data[training_method] = {}
            meta_data[training_method]['outputs'] = outputs_TM
            meta_data[training_method]['loss_dict'] = loss_dict_TM

        outputs = meta_data['cur_target']['outputs'] 
        loss_dict = meta_data['cur_target']['loss_dict']

        weight_dict = criterion.weight_dict

        loss_dict_keys = list(loss_dict.keys())

        for training_method in training_methods:
            if training_method == 'cur_target':
                continue 
            for loss_dict_key in loss_dict_keys:
                if loss_dict_key in ['loss_ce_enc','loss_bbox_enc','loss_giou_enc','loss_mask_enc','loss_dice_enc'] or 'CoMOT' in loss_dict_key or loss_dict_key not in meta_data[training_method]['loss_dict']: # enc loss only calculated once since dn_track / dn_object will not affect
                    continue
                assert (loss_dict_key + '_' + training_method) in weight_dict.keys()
                loss_dict[loss_dict_key + '_' + training_method] = meta_data[training_method]['loss_dict'][loss_dict_key] 

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys())
        
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

        cur_targets = [target['cur_target'] for target in targets]

        if targets[0]['track']:
            acc_dict = utils.calc_track_acc(outputs,cur_targets,args)
        else:
            acc_dict = utils.calc_bbox_acc(outputs,cur_targets,args)

        metrics_dict = utils.update_metrics_dict(metrics_dict,acc_dict,loss_dict,weight_dict,i,lr)

        dict_shape = metrics_dict['loss_ce'].shape
        for metrics_dict_key in metrics_dict.keys():
            if metrics_dict_key == 'lr':
                continue
            assert metrics_dict[metrics_dict_key].shape[:2] == dict_shape, 'Metrics needed to be added per epoch'

        if i in ids and (epoch % 5 == 0 or epoch == 1):
            data_viz.plot_results(outputs, prev_outputs, targets,samples.tensors, args.output_dir, folder=dataset + '_outputs', filename = f'Epoch{epoch:03d}_Step{i:06d}.png', args=args,meta_data=meta_data)
 
        if i > 0 and (i % interval == 0 or i == len(data_loader) - 1):
            utils.display_loss(metrics_dict,i,len(data_loader),epoch=epoch,dataset=dataset)

    return metrics_dict

@torch.no_grad()
def evaluate(model, criterion, data_loader, args, epoch: int = None, interval=50):

    model.eval()
    criterion.eval()
    dataset = 'val'
    ids = np.random.randint(0,len(data_loader),args.num_plots)
    ids = np.concatenate((ids,[0]))

    track_thresh = 0.1

    metrics_dict = {}
    for i, (samples, targets) in enumerate(data_loader):
         
        track = torch.rand(1).item() > track_thresh

        samples = samples.to(args.device)
        targets = [utils.nested_dict_to_device(t, args.device) for t in targets]

        outputs, targets, _, _, _, prev_outputs = model(samples,targets,track=track,epoch=epoch)

        training_methods = outputs['training_methods'] # dn_object, dn_track, dn_enc

        meta_data = {}
        groups = [0]

        for training_method in training_methods:
            outputs_TM, loss_dict_TM, groups = calc_loss_for_training_methods(training_method, outputs, groups, targets, criterion, epoch, args)

            meta_data[training_method] = {}
            meta_data[training_method]['outputs'] = outputs_TM
            meta_data[training_method]['loss_dict'] = loss_dict_TM

        outputs = meta_data['cur_target']['outputs']
        loss_dict = meta_data['cur_target']['loss_dict']
        weight_dict = criterion.weight_dict

        loss_dict_keys = list(loss_dict.keys())

        for training_method in training_methods:
            if training_method == 'cur_target':
                continue
            for loss_dict_key in loss_dict_keys:
                if loss_dict_key in ['loss_ce_enc','loss_bbox_enc','loss_giou_enc','loss_mask_enc','loss_dice_enc'] or 'CoMOT' in loss_dict_key or loss_dict_key not in meta_data[training_method]['loss_dict']: # enc loss only calculated once since dn_track / dn_object will not affect
                    continue
                loss_dict[loss_dict_key + '_' + training_method] = meta_data[training_method]['loss_dict'][loss_dict_key] 

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys())
        loss_dict['loss'] = losses

        cur_targets = [target['cur_target'] for target in targets]

        if targets[0]['track']:
            acc_dict = utils.calc_track_acc(outputs,cur_targets,args)
        else:
            acc_dict = utils.calc_bbox_acc(outputs,cur_targets,args)

        metrics_dict = utils.update_metrics_dict(metrics_dict,acc_dict,loss_dict,weight_dict,i)

        if i in ids and (epoch % 5 == 0 or epoch == 1):
            data_viz.plot_results(outputs, prev_outputs, targets,samples.tensors, args.output_dir, folder=dataset + '_outputs', filename = f'Epoch{epoch:03d}_Step{i:06d}.png', args=args,meta_data=meta_data)

        if i > 0 and (i % interval == 0  or i == len(data_loader) - 1):
            utils.display_loss(metrics_dict,i,len(data_loader),epoch=epoch,dataset=dataset)

    return metrics_dict



@torch.no_grad()
class pipeline():
    def __init__(self,model, fps, device, output_dir, args, track=True, use_NMS=False, display_masks=False,display_all_aux_outputs=None):
        self.model = model
        self.model.tracking()

        self.args = args
        self.use_NMS = use_NMS
        self.masks = display_masks and args.masks
        self.eval_ctc = args.eval_ctc
        self.output_dir = output_dir
        self.display_all_aux_outputs = display_all_aux_outputs

        if not self.eval_ctc:
            self.predictions_folder = 'predictions' 
            (self.output_dir / self.predictions_folder).mkdir(exist_ok=True)

            self.predictions_folder += '/track' if track else '/object_detection'
            self.predictions_folder += '_mask' if self.masks else ''
            self.predictions_folder += '_NMS' if self.use_NMS else ''

            (self.output_dir / self.predictions_folder).mkdir(exist_ok=True)
        else:
            self.predictions_folder = ''

        if args.hooks:
            self.use_hooks = True
            (self.output_dir / self.predictions_folder / 'attn_weight_maps').mkdir(exist_ok=True)
        else:
            self.use_hooks = False


        self.normalize = Compose([
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.threshold = 0.5
        self.target_size = args.target_size

        if isinstance(self.target_size[0],str):
            string_joined = ''.join([str(s) for s in self.target_size if s.isdigit() or s == ','])
            self.target_size = tuple(int(num) for num in string_joined.split(','))

        self.num_queries = args.num_queries
        self.device = device
        self.use_dab = args.use_dab
        self.two_stage = args.two_stage

        self.write_video = True
        self.track = track

        np.random.seed(24)

        if self.track:
            self.colors = np.array([tuple((255*np.random.random(3))) for _ in range(10000)]) # Assume max 1000 cells in one chamber
            self.colors[:6] = np.array([[0.,0.,255.],[0.,255.,0.],[255.,0.,0.],[255.,0.,255.],[0.,255.,255.],[255.,255.,0.]])
        else:
            self.colors = np.array([tuple((np.zeros(3))) for _ in range(10000)])
            
        self.all_colors = np.array([tuple((255*np.random.random(3))) for _ in range(10000)])
        self.all_colors[:6] = np.array([[0.,0.,255.],[0.,255.,0.],[255.,0.,0.],[255.,0.,255.],[0.,255.,255.],[255.,255.,0.]])

        if args.two_stage:
            # if not args.eval_ctc:
            (self.output_dir / self.predictions_folder / 'enc_outputs').mkdir(exist_ok=True)

            final_fmap = np.array([self.target_size[0] // 32, self.target_size[1] // 32])
            # TODO need to finish
            self.enc_map = [np.zeros((final_fmap[0]*2**f,final_fmap[1]*2**f)) for f in range(args.num_feature_levels)][::-1]

        self.oq_div = False # Can object queries detect divisions

        # if not self.eval_ctc or self.display_all_aux_outputs is not None and self.display_all_aux_outputs:
        #     self.display_decoder_aux = True
        #     if self.display_decoder_aux:
        #         (self.output_dir / self.predictions_folder / 'decoder_bbox_outputs').mkdir(exist_ok=True)
        #         self.num_decoder_frames = 1
        #         if args.two_stage:
        #             (self.output_dir / self.predictions_folder / 'enc_outputs').mkdir(exist_ok=True)
                    
        # else:
        #     self.display_decoder_aux = False          

        self.display_decoder_aux = True
        if self.display_decoder_aux:
            (self.output_dir / self.predictions_folder / 'decoder_bbox_outputs').mkdir(exist_ok=True)
            self.num_decoder_frames = 1
            if args.two_stage:
                (self.output_dir / self.predictions_folder / 'enc_outputs').mkdir(exist_ok=True)

        if self.eval_ctc:
            self.display_object_query_boxes = False
        else:
            if self.two_stage:
                self.display_object_query_boxes = False
            else:
                self.display_object_query_boxes = True

        if self.display_object_query_boxes:
            self.query_box_locations = [np.zeros((1,4)) for i in range(args.num_queries)]

        self.fps_split = fps
        self.videoname_list = []
        
        for fp in fps:
            if isinstance(fp,list):
                self.videoname_list.append(fp[0].parts[-2])
            elif fp.parts[-2] not in self.videoname_list:
                self.videoname_list.append(fp.parts[-2])

        if isinstance(self.fps_split[0],Path):
            max_len = len(self.fps_split)
            self.fps_split = [self.fps_split]
        else:
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

        self.div_track = -1 * np.ones((len(self.track_indices) + len(self.div_indices)),dtype=np.uint16) # keeps track of which cells were the result of cell division

        for div_ind in self.div_indices:
            ind = np.where(self.track_indices==div_ind)[0][0]

            self.max_cellnb += 1             
            self.cells = np.concatenate((self.cells[:ind+1],[self.max_cellnb],self.cells[ind+1:])) # add daughter cellnb after mother cellnb
            self.track_indices = np.concatenate((self.track_indices[:ind],self.track_indices[ind:ind+1],self.track_indices[ind:])) # order doesn't matter here since they are the same track indices

            self.div_track[ind:ind+2] = div_ind 

        self.new_cells = self.cells == 0

        if 0 in self.cells:
            self.max_cellnb += 1   
            self.cells[self.cells==0] = np.arange(self.max_cellnb,self.max_cellnb+sum(self.cells==0),dtype=np.uint16)
            
            assert np.max(self.cells) >= self.max_cellnb
            self.max_cellnb = np.max(self.cells)

    def update_div_boxes(self,boxes,masks=None):
        # boxes where div_indices were repeat; now they need to be rearrange because only the first box is sent to decoder
        unique_divs = np.unique(self.div_track[self.div_track != -1])
        for unique_div in unique_divs:
            div_ids = (self.div_track == unique_div).nonzero()[0]
            boxes[div_ids[1],:4] = boxes[div_ids[0],4:]
            # e.g. [[15, 45, 16, 22, 14, 62, 15, 20]   -->  [[15, 45, 16, 22, 14, 62, 15, 20]   -->  [[15, 45, 16, 22]  --> fed to decoder
            #       [15, 45, 16, 22, 14, 62, 15, 20]]  -->   [14, 62, 15, 20, 14, 62, 15, 20]]  -->   [14, 62, 15, 20]]  --> fed to decoder

            if masks is not None:
                masks[div_ids[1],:1] = masks[div_ids[0],1:] 

        return boxes, masks

    def NMS(self, pred_boxes, keep, keep_div, pred_masks):
        ''' Non-Max Suppresion'''
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

                if iou > 0.5:
                    keep[ind] = False
                    break

        track_ious = torch.triu(box_ops.generalized_box_iou(box_ops.box_cxcywh_to_xyxy(track_boxes),box_ops.box_cxcywh_to_xyxy(track_boxes),return_iou_only=True),diagonal=1)

        topk_nonself_matches = torch.where(track_ious > 0.5)

        if len(topk_nonself_matches[0] > 0):
            for midx in range(len(topk_nonself_matches[0])):
                box_1_ind = topk_nonself_matches[0][midx]
                box_2_ind = topk_nonself_matches[1][midx]

                if box_1_ind < len(track_keep) and box_2_ind < len(track_keep):
                    if pred_masks is not None: # here we pick the track query that has a matching mask; if mask is incorrect, then bbox is probably incorrect
                        mask_1 = pred_masks[track_keep[box_1_ind],0]
                        mask_2 = pred_masks[track_keep[box_2_ind],0]

                        if (mask_1 > 0.5).sum() == 0:
                            keep[track_keep[box_1_ind]] = False
                            keep_div[track_keep[box_1_ind]] = False
                            continue
                        elif (mask_2 > 0.5).sum() == 0:
                            keep[track_keep[box_2_ind]] = False
                            keep_div[track_keep[box_2_ind]] = False
                            continue

                        mask_1_where = torch.where(mask_1 > 0.5)
                        min_y_1, max_y_1 = mask_1_where[0].min(), mask_1_where[0].max() 
                        min_x_1, max_x_1  = mask_1_where[1].min(), mask_1_where[1].max()
                        height_1 = max_y_1 - min_y_1
                        width_1 = max_x_1 - min_x_1
                        center_y_1 = height_1 / 2 + min_y_1
                        center_x_1 = width_1 / 2 + min_x_1

                        mask_1_box = torch.tensor([[center_x_1,center_y_1,width_1,height_1]],device=self.device)
                        mask_1_box[:,::2] = mask_1_box[:,::2] / mask_1.shape[-1]
                        mask_1_box[:,1::2] = mask_1_box[:,1::2] / mask_1.shape[-2]

                        mask_2_where = torch.where(mask_2 > 0.5)
                        min_y_2, max_y_2 = mask_2_where[0].min(), mask_2_where[0].max() 
                        min_x_2, max_x_2  = mask_2_where[1].min(), mask_2_where[1].max()
                        height_2 = max_y_2 - min_y_2
                        width_2 = max_x_2 - min_x_2
                        center_y_2 = height_2 / 2 + min_y_2
                        center_x_2 = width_2 / 2 + min_x_2

                        mask_2_box = torch.tensor([[center_x_2,center_y_2,width_2,height_2]],device=self.device)
                        mask_2_box[:,::2] = mask_2_box[:,::2] / mask_2.shape[-1]
                        mask_2_box[:,1::2] = mask_2_box[:,1::2] / mask_2.shape[-2]

                        iou_1 = box_ops.generalized_box_iou(box_ops.box_cxcywh_to_xyxy(track_boxes[box_1_ind]),box_ops.box_cxcywh_to_xyxy(mask_1_box),return_iou_only=True)
                        iou_2 = box_ops.generalized_box_iou(box_ops.box_cxcywh_to_xyxy(track_boxes[box_2_ind]),box_ops.box_cxcywh_to_xyxy(mask_2_box),return_iou_only=True)

                        if iou_1 > iou_2:
                            keep[track_keep[box_2_ind]] = False
                            keep_div[track_keep[box_2_ind]] = False
                        else:
                            keep[track_keep[box_1_ind]] = False
                            keep_div[track_keep[box_1_ind]] = False

                    else: # without masks, it is impossible to tell which is correct
                        keep[track_keep[box_2_ind]] = False
                        keep_div[track_keep[box_2_ind]] = False
                elif box_1_ind < len(track_keep) and box_2_ind >= len(track_keep): # track query and div overlap
                    keep_div[track_div_keep[box_2_ind - len(track_keep)]] = False
                elif box_2_ind < len(track_keep) and box_1_ind >= len(track_keep): # track query and div overlap
                    keep_div[track_div_keep[box_1_ind - len(track_keep)]] = False
                elif box_1_ind > len(track_keep) and box_1_ind > len(track_keep): # div query and div overlap
                    keep_div[track_div_keep[box_2_ind - len(track_keep)]] = False

        return keep.numpy(), keep_div.numpy()

    def forward(self):
        for r,fps_ROI in enumerate(self.fps_split):
            print(self.videoname_list[r])
            print(f'video {r+1}/{len(self.fps_split)}')

            self.max_cellnb = 0
            targets = None
            prev_features = None

            if self.eval_ctc:
                ctc_data = None
            
            if self.display_decoder_aux:
                random_nbs = np.random.choice(np.arange(1,len(fps_ROI)),self.num_decoder_frames)
                random_nbs = np.concatenate((random_nbs,random_nbs+1)) # so we can see two consecutive frames

            for i, fp in enumerate(tqdm(fps_ROI)):

                if self.use_hooks and ((self.display_decoder_aux and i in random_nbs) or (self.display_all_aux_outputs and i > 0)):
                    dec_attn_outputs = []
                    hooks = [self.model.decoder.layers[0].self_attn.register_forward_hook(lambda self, input, output: dec_attn_outputs.append(output)),
                        self.model.decoder.layers[-1].self_attn.register_forward_hook(lambda self, input, output: dec_attn_outputs.append(output))]


                img = cv2.imread(str(fp),cv2.IMREAD_ANYDEPTH)
                img_dtype = img.dtype
                img_shape = img.shape
                if img_dtype == 'uint16':
                    img = ((img - np.min(img)) / np.ptp(img))
                    img = cv2.resize(img,(self.target_size[1],self.target_size[0]))
                    img = (255 * img).astype(np.uint8)
                    img = PIL.Image.fromarray(img).convert('RGB')
                else:
                    img = PIL.Image.open(fp,mode='r')
                    img_shape = img.size
                    img = img.resize((self.target_size[1],self.target_size[0])).convert('RGB')

                if i > 0:
                    if img_dtype == 'uint16':
                        previmg = cv2.imread(str(fps_ROI[i-1]),cv2.IMREAD_ANYDEPTH)
                        previmg = ((previmg - np.min(previmg)) / np.ptp(previmg))
                        previmg = cv2.resize(previmg,(self.target_size[1],self.target_size[0]))
                        previmg = (255 * previmg).astype(np.uint8)
                        previmg = PIL.Image.fromarray(previmg).convert('RGB')
                    else:
                        previmg = PIL.Image.open(fps_ROI[i-1],mode='r').resize((self.target_size[1],self.target_size[0])).convert('RGB') if i > 0 else None # saved for analysis later

                samples = self.normalize(img)[0][None]
                samples = samples.to(self.device)

                if not self.track: # If object detction only, we don't feed information from the previous frame
                    targets = None
                    self.max_cellnb = 0
                    prev_features = None

                with torch.no_grad():
                    outputs, targets, prev_features, memory, hs, _ = self.model(samples,targets=targets,prev_features=prev_features)

                if targets is None:
                    targets = [{'cur_target':{}}]

                pred_logits = outputs['pred_logits'][0].sigmoid().detach().cpu().numpy()
                pred_boxes = outputs['pred_boxes'][0].detach()

                keep = (pred_logits[:,0] > self.threshold)
                keep_div = (pred_logits[:,1] > self.threshold)
                keep_div[-self.num_queries:] = False # disregard any divisions predicted by object queries; model should have learned not to do this anyways
                keep_div[~keep] = False

                if self.use_NMS and len(keep) > self.num_queries:
                    if 'pred_masks' in outputs:
                        pred_masks = outputs['pred_masks'][0].sigmoid().detach()
                    else:
                        pred_masks = None

                    keep, keep_div = self.NMS(pred_boxes, keep, keep_div,pred_masks)

                if i > 0:
                    prevcells = np.copy(self.cells)

                self.cells = np.zeros((keep.sum()),dtype=np.uint16)
                masks = None

                # If no objects are detected, skip
                if self.display_object_query_boxes and keep[-self.num_queries:].sum() > 0:
                    self.update_query_box_locations(pred_boxes,keep,keep_div)

                if sum(keep) > 0:
                    self.track_indices = keep.nonzero()[0] # Get query indices (object or track) where a cell was detected / tracked; this is used to create track queries for next  frame
                    self.div_indices = keep_div.nonzero()[0] # Get track query indices where a cell division was tracked; object queries should not be able to detect divisions

                    if pred_logits.shape[0] > self.num_queries: # If track queries are fed to the model
                        tq_keep = keep[:len(prevcells)]
                        self.cells[:sum(tq_keep)] = prevcells[tq_keep]
                    else:
                        prevcells = None

                    self.split_up_divided_cells()

                    targets[0]['cur_target']['track_query_hs_embeds'] = outputs['hs_embed'][0,self.track_indices] # For div_indices, hs_embeds will be the same; no update
                    boxes = pred_boxes[self.track_indices] # For div_indices, the boxes will be repeated and will properly updated below

                    if self.masks:
                        masks = outputs['pred_masks'][0,self.track_indices].sigmoid().detach()

                    boxes,masks = self.update_div_boxes(boxes,masks)
                    boxes = boxes[:,:4] # only one cell is tracked at a time
        
                    if masks is not None: # gets rid of the division mask which we rearrange above
                        masks = masks[:,0]

                    if self.args.init_boxes_from_masks:
                        h, w = masks.shape[-2:]

                        mask_boxes = utils.mask_to_bbox(masks > 0.5)

                        # mask_boxes = box_ops.masks_to_boxes(masks > 0.5).cuda()
                        # mask_boxes = box_ops.box_xyxy_to_cxcywh(mask_boxes) / torch.as_tensor([w, h, w, h],dtype=torch.float,device=self.device)
                        targets[0]['cur_target']['track_query_boxes'] = mask_boxes

                        if self.args.iterative_masks:
                            track_query_masks = torch.cat((masks[:,None],torch.zeros_like(masks[:,None])),axis=1)
                            targets[0]['cur_target']['track_query_masks'] = track_query_masks

                            assert track_query_masks.shape[0] == boxes.shape[0]
                            assert track_query_masks.shape[0] == outputs['hs_embed'][0,self.track_indices].shape[0]

                    else:
                        targets[0]['cur_target']['track_query_boxes'] = boxes

                    if i == 0: # self.new_cells is used to visually highlight errors specifically for the mother machine. This is because no new cells will ever appear so I know this is an erorr
                        self.new_cells = None

                else:
                    self.track_indices = None
                    self.div_track = None
                    if prevcells is not None:
                        self.div_track = -1 * np.ones(len(prevcells),dtype=np.uint16)
                    else:
                        self.div_track = np.ones(0,dtype=np.uint16)
                    boxes = None
                    self.new_cells = None
                    # prevcells = None
                    masks = None

                    targets = None


                if boxes is not None: # No cells
                    assert boxes.shape[0] == len(self.cells)

                if self.track:
                    color_frame = utils.plot_tracking_results(img,boxes,masks,self.colors[self.cells-1],self.div_track,self.new_cells)
                else:
                    color_frame = utils.plot_tracking_results(img,boxes,masks,self.colors[:len(self.cells)],self.div_track,None)

                color_frame = cv2.putText(
                    color_frame,
                    text = f'{i:03d}', 
                    org=(0,10), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale = 0.4,
                    color = (255,255,255),
                    thickness=1,
                    )

                self.color_stack[i,:,r*self.target_size[1]:(r+1)*self.target_size[1]] = color_frame         

                if self.use_hooks and ((self.display_decoder_aux and i in random_nbs) or (self.display_all_aux_outputs and i > 0)):
                    for hook in hooks:
                        hook.remove()

                    scale = 10

                    for dec_attn_output,layer in zip(dec_attn_outputs,['first','last']):
                        dec_attn_weight_maps = dec_attn_output[1].cpu().numpy() # [output, attention_map]
                        dec_attn_weight_maps = np.repeat(dec_attn_weight_maps[...,None],3,axis=-1)
                        for dec_attn_weight_map in dec_attn_weight_maps: # per batch
                            dec_attn_weight_map_color = np.zeros((dec_attn_weight_map.shape[0]+1,dec_attn_weight_map.shape[1]+1,3))
                            dec_attn_weight_map_color[-dec_attn_weight_map.shape[0]:,-dec_attn_weight_map.shape[1]:] = dec_attn_weight_map
                            dec_attn_weight_map_color = ((dec_attn_weight_map_color / np.max(dec_attn_weight_map_color)) * 255).astype(np.uint8)

                            for tidx in range(len(keep) - self.num_queries):
                                dec_attn_weight_map_color[tidx+1,0] = self.all_colors[prevcells[tidx]-1]
                                dec_attn_weight_map_color[0,tidx+1] = self.all_colors[prevcells[tidx]-1]                                

                            dec_attn_weight_map_color_resize = cv2.resize(dec_attn_weight_map_color,(dec_attn_weight_map.shape[1]*scale,dec_attn_weight_map.shape[0]*scale),interpolation=cv2.INTER_NEAREST)
                            cv2.imwrite(str(self.output_dir / self.predictions_folder / 'attn_weight_maps' / (f'self_attn_weight_map_{fp.stem}_{layer}_layer.png')),dec_attn_weight_map_color_resize)

                            for q in range(len(keep) - self.num_queries,len(keep)):
                                dec_attn_weight_map_color[q+1,0] = self.all_colors[-q]
                                dec_attn_weight_map_color[0,q+1] = self.all_colors[-q]

                            
                            dec_attn_weight_map_color_resize = cv2.resize(dec_attn_weight_map_color,(dec_attn_weight_map.shape[1]*scale,dec_attn_weight_map.shape[0]*scale),interpolation=cv2.INTER_NEAREST)
                            cv2.imwrite(str(self.output_dir / self.predictions_folder / 'attn_weight_maps' / (f'self_attn_weight_map_{fp.stem}_oq_color_{layer}_layer.png')),dec_attn_weight_map_color_resize)


                if self.eval_ctc:

                    mask_threshold = 0.5

                    if sum(keep) > 0:
                        # Convert tensor to numpy and resize array to target size
                        masks = masks.cpu().numpy()
                        masks = np.transpose(masks,(1,2,0))
                        masks = cv2.resize(masks,img_shape)
                        if masks.ndim == 2: # cv2.resize will drop last dim if it is 1
                            masks = masks[:,:,None]
                        masks = np.transpose(masks,(-1,0,1))

                        masks_filt = np.zeros((masks.shape))
                        argmax = np.argmax(masks,axis=0)
                        
                        for m in range(masks.shape[0]):
                            masks_filt[m,argmax==m] = masks[m,argmax==m]
                            
                        masks_filt = masks_filt > mask_threshold

                        mask = np.zeros(masks.shape[-2:],dtype=np.uint16)
                        
                        for m, cell in enumerate(self.cells):
                            if masks_filt[m].sum() > 0:
                                mask[masks_filt[m] > 0] = cell
                            else: # Embedding was predicted to be a cell but therer is no cell mask so we remove it
                                keep_cells = self.cells != cell
                                track_ind = self.track_indices[~keep_cells][0]
                          
                                if track_ind in self.div_indices:          
                                    keep_div[track_ind] = False
                                    self.div_indicies = self.div_indices[self.div_indices != track_ind]

                                    other_div_cell = self.cells[(self.div_track == track_ind) * (keep_cells)]

                                    self.div_track[self.div_track == track_ind] = -1
                                    self.cells[self.cells == other_div_cell] = prevcells[track_ind]
                                
                                self.cells = self.cells[keep_cells]
                                keep[track_ind] = False
                                self.track_indices = self.track_indices[keep_cells]
                                self.div_track = self.div_track[keep_cells]

                                targets[0]['cur_target']['track_query_hs_embeds'] = targets[0]['cur_target']['track_query_hs_embeds'][keep_cells]
                                targets[0]['cur_target']['track_query_boxes'] = targets[0]['cur_target']['track_query_boxes'][keep_cells]

                                if 'track_query_masks' in targets[0]['cur_target']:
                                    targets[0]['cur_target']['track_query_masks'] = targets[0]['cur_target']['track_query_masks'][keep_cells]

                        mask_cells = np.unique(mask)
                        mask_cells = mask_cells[mask_cells != 0]

                    if ctc_data is None:
                        ctc_data = []
                        for cell in self.cells:
                            ctc_data.append(np.array([cell,i,i,0]))
                        ctc_data = np.stack(ctc_data)
                        ctc_cells = np.copy(self.cells)
                    else:
                        max_cellnb = ctc_data.shape[0]

                        ctc_cells_new = np.copy(self.cells)
                        mask_copy = np.copy(mask)

                        if prevcells is not None:
                            for c,cell in enumerate(prevcells):
                                if cell in self.cells:
                                    ctc_cell = ctc_cells[c]
                                    if self.div_track[self.cells == cell] != -1:
                                        div_ind = self.div_track[self.cells == cell]
                                        div_cells = self.cells[self.div_track == div_ind]
                                        max_cellnb += 1
                                        new_cell_1 = np.array([max_cellnb,i,i,ctc_cell])[None]
                                        ctc_cells_new[self.cells == div_cells[0]] = max_cellnb 
                                        mask[mask_copy == div_cells[0]] = max_cellnb
                                        max_cellnb += 1
                                        new_cell_2 = np.array([max_cellnb,i,i,ctc_cell])[None]
                                        ctc_data = np.concatenate((ctc_data,new_cell_1,new_cell_2),axis=0)
                                        ctc_cells_new[self.cells == div_cells[1]] = max_cellnb
                                        mask[mask_copy == div_cells[1]] = max_cellnb
                                        assert div_cells[0] in mask_copy and div_cells[1] in mask_copy

                                    else:
                                        ctc_data[ctc_cell-1,2] = i
                                        ctc_cells_new[self.cells == cell] = ctc_cells[prevcells == cell]
                                        mask[mask_copy == cell] = ctc_cell
                                        assert cell in mask_copy

                        for c,cell in enumerate(self.cells):
                            if prevcells is None or cell not in prevcells and self.div_track[c] == -1:
                                max_cellnb += 1
                                new_cell = np.array([max_cellnb,i,i,0])
                                ctc_data = np.concatenate((ctc_data,new_cell[None]),axis=0)

                                ctc_cells_new[self.cells == cell] = max_cellnb     
                                mask[mask_copy == cell] = max_cellnb

                        ctc_cells = ctc_cells_new

                    cv2.imwrite(str(self.output_dir / self.predictions_folder / f'mask{i:03d}.tif'),mask)
                            

                if ((self.display_decoder_aux and i in random_nbs) or (self.display_all_aux_outputs and i > 0)):

                    if 'enc_outputs' in outputs:
                        enc_frame = np.array(img).copy()
                        enc_outputs = outputs['enc_outputs']
                        logits_topk = enc_outputs['pred_logits'][0,:,0].sigmoid()
                        boxes_topk = enc_outputs['pred_boxes'][0]
                        topk_proposals = enc_outputs['topk_proposals'][0]

                        enc_colors = np.array([(np.array([0.,0.,0.])) for _ in range(self.num_queries)])
                        used_object_queries = torch.where(outputs['pred_logits'][0,-self.num_queries:,0].sigmoid() > self.threshold)[0].sort()[0]
                        num_tracked_cells = len(self.cells) - len(used_object_queries)

                        if self.track:
                            if len(used_object_queries) > 0:
                                counter = 0
                                for pidx in used_object_queries:
                                        enc_colors[pidx] = self.all_colors[self.cells[num_tracked_cells+counter]-1]
                                        counter += 1
                            else:
                                enc_colors = np.array([tuple((255*np.random.random(3))) for _ in range(logits_topk.shape[0])])
                        else:
                            enc_colors[:len(self.cells)] = self.all_colors[self.cells-1]

                        t0,t1,t2,t3 = 0.1,0.3,0.5,0.8
                        boxes_list = []
                        boxes_list.append(boxes_topk[logits_topk > t3])
                        boxes_list.append(boxes_topk[(logits_topk > t2) * (logits_topk < t3)])
                        boxes_list.append(boxes_topk[(logits_topk > t1) * (logits_topk < t2)])
                        boxes_list.append(boxes_topk[(logits_topk > t0) * (logits_topk < t1)])
                        boxes_list.append(boxes_topk[logits_topk < t0])

                        num_per_box = [box.shape[0] for box in boxes_list]
                        all_enc_boxes = []
                        enc_frames = []
                        for b,boxes in enumerate(boxes_list):
                            enc_frame = np.array(img).copy()
                            enc_frames.append(utils.plot_tracking_results(enc_frame,boxes,None,enc_colors[sum(num_per_box[:b]):sum(num_per_box[:b+1])],None,None))
                            all_enc_boxes.append(boxes[sum(num_per_box[:b]):sum(num_per_box[:b+1])])
                        
                        enc_frame = np.array(img).copy()
                        all_enc_boxes = torch.cat(boxes_list)
                        all_enc_boxes = torch.cat((all_enc_boxes[np.sum(enc_colors,-1) == 0],all_enc_boxes[np.sum(enc_colors,-1) > 0]))
                        enc_colors = np.concatenate((enc_colors[np.sum(enc_colors,-1) == 0],enc_colors[np.sum(enc_colors,-1) > 0]))
                        enc_frames.append(utils.plot_tracking_results(enc_frame,all_enc_boxes,None,enc_colors,None,None))

                        if len(used_object_queries) > 0:
                            enc_colors = np.array([(np.array([0.,0.,0.])) for _ in range(self.num_queries)])
                            enc_frames.append(utils.plot_tracking_results(enc_frame,all_enc_boxes,None,enc_colors,None,None))

                        enc_frames = np.concatenate((enc_frames),axis=1)

                        cv2.imwrite(str(self.output_dir / self.predictions_folder / 'enc_outputs' / (f'encoder_frame_{fp.stem}.png')),enc_frames)

                        if self.enc_map is not None:
                            #TODO update for future
                            proposals_img = np.array(img)
                            spatial_shapes = enc_outputs['spatial_shapes']

                            fmaps_cum_size = torch.tensor([spatial_shape[0] * spatial_shape[1] for spatial_shape in spatial_shapes]).cumsum(0)

                            proposals = topk_proposals[enc_outputs['pred_logits'][0,:,0].sigmoid() > 0.5].clone().cpu()
                            topk_proposals = topk_proposals.cpu()

                            for proposal in proposals:
                                proposal_ind = torch.where(topk_proposals == proposal)[0][0]
                                f = 0
                                fmap = fmaps_cum_size[f]
                                proposal_clone = proposal.clone()

                                while proposal_clone >= fmap and f < len(fmaps_cum_size) - 1:
                                    f += 1 
                                    fmap = fmaps_cum_size[f]

                                if f > 0:
                                    proposal_clone -= fmaps_cum_size[f-1]

                                y = torch.floor_divide(proposal_clone,spatial_shapes[f,1])
                                x = torch.remainder(proposal_clone,spatial_shapes[f,1])

                                self.enc_map[f][y,x] += 1
                                
                                small_mask = np.zeros(spatial_shapes[f].cpu().numpy(),dtype=np.uint8)
                                small_mask[y,x] += 1

                                resize_mask = cv2.resize(small_mask,self.target_size[::-1])

                                y, x = np.where(resize_mask == 1)

                                X = np.min(x) 
                                Y = np.min(y)
                                width = (np.max(x) - np.min(x)) 
                                height = (np.max(y) - np.min(y)) 

                                if len(used_object_queries) > 0:
                                    proposals_img = cv2.rectangle(proposals_img,(X,Y),(int(X+width),int(Y+height)),color=self.all_colors[self.cells[self.track_indices == proposal_ind]-1],thickness=1)
                                else:
                                    proposals_img = cv2.rectangle(proposals_img,(X,Y),(int(X+width),int(Y+height)),color=enc_colors[proposal == topk_proposals][0],thickness=1)


                            cv2.imwrite(str(self.output_dir / self.predictions_folder / 'enc_outputs' / (f'encoder_frame_{fp.stem}_proposals.png')),proposals_img)

                    references = outputs['references'].detach()

                    aux_outputs = outputs['aux_outputs'] # output from the intermedaite layers of decoder
                    aux_outputs = [{'pred_boxes':references[0]}] + aux_outputs # add the initial anchors / reference points
                    aux_outputs.append({'pred_boxes':outputs['pred_boxes'].detach(),'pred_masks':outputs['pred_masks'].detach(),'pred_logits':outputs['pred_logits'].detach()}) # add the last layer of the decoder which is the final prediction

                    img = np.array(img)

                    cells_exit_ids = torch.tensor([[cidx,c] for cidx,c in enumerate(prevcells.astype(np.int64)) if c not in self.cells]) if prevcells is not None else None

                    if self.track:
                        previmg_copy = previmg.copy()
                        img_box = img.copy()
                        img_mask = img.copy()
                    for a,aux_output in enumerate(aux_outputs):
                        all_boxes = aux_output['pred_boxes'][0].detach()
                        img_copy = img.copy()
                        
                        if self.track_indices is not None:
                            tq_ind_only = self.track_indices < (len(keep) - self.num_queries)
                            track_indices_tq_only = self.track_indices[tq_ind_only]
                            div_track = self.div_track[tq_ind_only]                        
                        else:
                            track_indices_tq_only = None
                        
                        if a > 0:
                            all_logits = aux_output['pred_logits'][0].detach()
                            all_masks = aux_output['pred_masks'][0].detach()
                            object_boxes = all_boxes[-self.num_queries:,:4]
                            object_masks = all_masks[-self.num_queries:,0]
                            object_logits = all_logits[-self.num_queries:,0]

                            object_indices = torch.where(object_logits > 0.5)[0]
                            pred_object_boxes = object_boxes[object_indices]
                            pred_object_masks = object_masks[object_indices]

                            object_indices += (len(keep) - self.num_queries)

                            if len(object_indices) > 0:
                                img_object = utils.plot_tracking_results(img_copy,pred_object_boxes,pred_object_masks,self.all_colors[-object_indices.cpu()],None,None)
                            else:
                                img_object = img_copy

                        if track_indices_tq_only is not None:
                            
                            if a > 0: # initial reference points are all single boxes; this applies to the outputs of decoder
                                if len(track_indices_tq_only) > 0:
                                    track_boxes = all_boxes[track_indices_tq_only]
                                    track_masks = all_masks[track_indices_tq_only]
                                    track_logits = all_logits[track_indices_tq_only]
                                    # unique_divs = np.unique(self.div_track[self.div_track != -1])
                                    unique_divs = np.unique(div_track[div_track != -1])
                                    for unique_div in unique_divs:
                                        div_ids = (div_track == unique_div).nonzero()[0]
                                        # div_ids = (self.div_track == unique_div).nonzero()[0]
                                        track_boxes[div_ids[1],:4] = track_boxes[div_ids[0],4:]
                                        track_masks[div_ids[1],:1] = track_masks[div_ids[0],1:]
                                        track_logits[div_ids[1],:1] = track_logits[div_ids[0],1:]

                                    track_boxes = track_boxes[:,:4]
                                    track_masks = track_masks[:,0]
                                    track_logits = track_logits[:,0]
                                    # div_track = self.div_track

                                    track_box_colors = self.all_colors[self.cells-1]
                                    new_cells = self.new_cells if self.track else None
                                else:
                                    track_boxes = torch.zeros((0,4),dtype=all_boxes.dtype,device=all_boxes.device)

                            elif a == 0:
                                track_masks = None
                                # track_boxes = all_boxes[np.unique(track_indices_tq_only[track_indices_tq_only < len(keep) - self.num_queries])]
                                track_boxes = all_boxes[np.arange(len(keep) - self.num_queries)]
                                object_boxes = all_boxes[-self.num_queries:]
                                track_box_colors = self.all_colors[prevcells-1] if prevcells is not None else self.all_colors[self.cells-1]
                                object_box_colors = np.array([(np.array([0.,0.,0.])) for _ in range(self.num_queries)])
                                div_track = np.ones((track_boxes.shape[0])) * -1
                                new_cells = None

                                all_black = object_box_colors.copy()

                                if self.track:
                                    assert track_boxes.shape[0] <= track_box_colors.shape[0]
                                    previmg_track_anchor_boxes = utils.plot_tracking_results(previmg_copy,track_boxes,None,track_box_colors,div_track,new_cells)

                                    all_colors = np.concatenate((object_box_colors,track_box_colors),axis=0)
                                    all_boxes_rev = torch.cat((object_boxes,track_boxes),axis=0)
                                    previmg_all_anchor_boxes = utils.plot_tracking_results(img_copy,all_boxes_rev,None,all_colors,None,None)

                                else:
                                    object_box_colors[-track_box_colors.shape[0]:] = track_box_colors
                                    if self.two_stage:
                                        object_boxes = torch.cat((object_boxes[track_box_colors.shape[0]:],object_boxes[:track_box_colors.shape[0]]))
                                    else:
                                        unused_oqs = np.array([oq_id for oq_id in range(self.num_queries) if oq_id not in track_indices_tq_only])
                                        object_boxes = torch.cat((object_boxes[unused_oqs],object_boxes[track_indices_tq_only]))

                                img_object_anchor_boxes_all = utils.plot_tracking_results(img_copy,object_boxes,None,object_box_colors,None,None)
                                img_object_anchor_boxes_not_track_only = utils.plot_tracking_results(img_copy,object_boxes,None,all_black,None,None)

                                oq_indices = np.array([q for q in range(len(keep)) if q not in track_indices_tq_only])
                                oq_colors = self.all_colors[-oq_indices]
                                img_object_anchor_boxes_not_track_color_only = utils.plot_tracking_results(img_copy,all_boxes[oq_indices],None,oq_colors,None,None)

                                if not self.track:
                                    img_object_anchor_boxes_track_only = utils.plot_tracking_results(img_copy,object_boxes[-len(track_indices_tq_only):],None,track_box_colors,None,None)
                                    img_object_anchor_boxes = np.concatenate((img_object_anchor_boxes_all,img_object_anchor_boxes_not_track_only,img_object_anchor_boxes_track_only,img_object_anchor_boxes_not_track_color_only),axis=1)
                                else:
                                    img_object_anchor_boxes = np.concatenate((img_object_anchor_boxes_all,img_object_anchor_boxes_not_track_only,img_object_anchor_boxes_not_track_color_only),axis=1)

                                if i == 1:
                                    all_prev_logits = prev_outputs['pred_logits'][0].detach().sigmoid()
                                    all_prev_boxes = prev_outputs['pred_boxes'][0]
                                    all_prev_masks = prev_outputs['pred_masks'][0,:,0]

                                    prevkeep = all_prev_logits[:,0] > self.threshold
                                    prev_boxes = all_prev_boxes[prevkeep]
                                    prev_masks = all_prev_masks[prevkeep]
                                    prev_colors = self.all_colors[prevcells-1]

                                    first_img_box = utils.plot_tracking_results(previmg_copy,prev_boxes,None,prev_colors,None,None)
                                    first_img_mask = utils.plot_tracking_results(previmg_copy,None,prev_masks,prev_colors,None,None)
                                    first_img_box_and_mask = utils.plot_tracking_results(previmg_copy,prev_boxes,prev_masks,prev_colors,None,None)

                                    img_object_anchor_boxes = np.concatenate((first_img_box,first_img_mask,first_img_box_and_mask,img_object_anchor_boxes),axis=1)

                            img_track = utils.plot_tracking_results(img_copy,track_boxes,track_masks,track_box_colors,div_track,new_cells)

                            if a == len(aux_outputs) - 1:
                                img_box = utils.plot_tracking_results(img_box,track_boxes,None,track_box_colors,div_track,new_cells)
                                img_mask = utils.plot_tracking_results(img_mask,None,track_masks,track_box_colors,div_track,new_cells)

                        else:
                            track_boxes = torch.zeros((0,4),dtype=all_boxes.dtype,device=all_boxes.device)
                            img_track = img_copy
                        
                    
                        if a == 0 and self.track:
                            if track_indices_tq_only is not None:
                                decoder_frame = np.concatenate((previmg,img,previmg_all_anchor_boxes,img_object_anchor_boxes,previmg_track_anchor_boxes,img_track),axis=1)
                            else:
                                decoder_frame = np.concatenate((previmg,img,img_track,previmg_all_anchor_boxes),axis=1)
                        elif a == 0:
                            decoder_frame = np.concatenate((img,img_track,img_object_anchor_boxes),axis=1)
                        else:
                            decoder_frame = np.concatenate((decoder_frame,img_track,img_object),axis=1)

                        if a == len(aux_outputs)-1:
                            decoder_frame = np.concatenate((decoder_frame,img_box,img_mask),axis=1)

                        # Plot all predictions regardless of cls label
                        if a == len(aux_outputs) - 1:
                            img_copy = img.copy()
                            
                            color_queries = np.array([(np.array([0.,0.,0.])) for _ in range(self.num_queries)])

                            if  cells_exit_ids is not None and cells_exit_ids.shape[0] > 0: # Plot the track query that left the chamber
                                boxes_exit = all_boxes[cells_exit_ids[:,0],:4]
                                boxes = torch.cat((all_boxes[-self.num_queries:,:4],boxes_exit,track_boxes))
                                colors_prev = self.all_colors[cells_exit_ids[:,1] - 1] 
                                colors_prev = colors_prev[None] if colors_prev.ndim == 1 else colors_prev
                                
                                all_colors = np.concatenate((color_queries,colors_prev,self.all_colors[self.cells-1]),axis=0)
                                div_track_all = np.ones((boxes.shape[0])) * -1 # all boxes does not contain div boxes separated
                                if len(track_boxes) > 0:
                                    div_track_all[-len(track_boxes):] = div_track

                                assert len(div_track_all) == len(boxes)

                            else: # all cells / track queries stayed in the chamber
                                boxes = torch.cat((all_boxes[-self.num_queries:,:4],track_boxes))
                                all_colors = np.concatenate((color_queries,self.all_colors[self.cells-1]),axis=0)
                                # div_track_all = np.concatenate((np.ones((self.num_queries))*-1,self.div_track))
                                div_track_all = np.concatenate((np.ones((self.num_queries))*-1,div_track))

                                assert len(div_track_all) == len(boxes)

                            new_cell_thickness = np.zeros_like(div_track_all).astype(bool)
                            # new_cell_thickness[self.num_queries:] = True # set all track queries with a thickened boudning box so it's easier to see
                            if cells_exit_ids is not None and cells_exit_ids.shape[0] > 0:
                                new_cell_thickness[self.num_queries:-track_boxes.shape[0]] = False
                            img_final_box = utils.plot_tracking_results(img_copy,boxes,None,all_colors,div_track_all,new_cell_thickness)

                            color_black = np.array([(np.array([0.,0.,0.])) for _ in range(boxes.shape[0])]) # plot all bounding boxes as black
                            img_final_all_box = utils.plot_tracking_results(img_copy,boxes,None,color_black,None,None)

                            if  cells_exit_ids is not None and cells_exit_ids.shape[0] > 0: # Plot the track query that left the chamber
                                all_colors[len(color_queries):-len(self.cells)] = np.array([0.,0.,0.])
                                img_final_all_box = utils.plot_tracking_results(img_copy,boxes,None,all_colors,None,None)

                            oq_indices = np.array([q for q in range(len(keep)) if track_indices_tq_only is None or q not in track_indices_tq_only])
                            oq_colors = self.all_colors[-oq_indices]
                            img_oq_only_color = utils.plot_tracking_results(img_copy,all_boxes[oq_indices,:4],None,oq_colors,None,None)
                            
                            decoder_frame = np.concatenate((decoder_frame,img_final_box,img_final_all_box,img_oq_only_color),axis=1)

                    if self.args.CoMOT:
                        img_CoMOT_color = img.copy()
                        pred_logits_aux = outputs['aux_outputs'][-2]['pred_logits'][0,-self.num_queries:,0].sigmoid().cpu().numpy()
                        aux_object_ind = np.where(pred_logits_aux > 0.5)[0] + (len(keep) - self.num_queries)

                        if len(aux_object_ind) > 0:
                            aux_boxes = outputs['aux_outputs'][-2]['pred_boxes'][0,aux_object_ind,:4]
                            aux_masks = outputs['aux_outputs'][-2]['pred_masks'][0,aux_object_ind,0]

                            img_CoMOT_color = utils.plot_tracking_results(img_CoMOT_color,aux_boxes,aux_masks,self.all_colors[-aux_object_ind],None,None)


                        decoder_frame = np.concatenate((decoder_frame,img_CoMOT_color),axis=1)

                    method = 'object_detection' if not self.track else 'track'
                    cv2.imwrite(str(self.output_dir / self.predictions_folder / 'decoder_bbox_outputs' / (f'{method}_decoder_frame_{fp.stem}.png')),decoder_frame)

                else:
                    if 'enc_outputs' in outputs and self.enc_map is not None:
                        
                        enc_outputs = outputs['enc_outputs']
                        topk_proposals = enc_outputs['topk_proposals'][0]
                        spatial_shapes = enc_outputs['spatial_shapes']

                        fmaps_cum_size = torch.tensor([spatial_shape[0] * spatial_shape[1] for spatial_shape in spatial_shapes]).cumsum(0)

                        # proposals = topk_proposals[enc_outputs['pred_logits'][0,topk_proposals,0].sigmoid() > 0.5].clone()

                        for proposal in topk_proposals:
                            f = 0
                            fmap = fmaps_cum_size[f]

                            while proposal >= fmap and f < len(fmaps_cum_size) - 1:
                                f += 1 
                                fmap = fmaps_cum_size[f]

                            if f > 0:
                                proposal -= fmaps_cum_size[f-1]

                            y = torch.floor_divide(proposal,spatial_shapes[f,1])
                            x = torch.remainder(proposal,spatial_shapes[f,1])

                            self.enc_map[f][y,x] += 1

                torch.cuda.empty_cache()
                    
                if sum(keep) == 0:
                    prevcells = None

                prev_outputs = outputs.copy()

        if 'enc_outputs' in outputs:
            spacer = 1
            scale = self.target_size[0] / self.enc_map[0].shape[0]
            enc_maps = []
            output_enc_maps = []
            max_value = 0
            for e,enc_map in enumerate(self.enc_map):

                enc_map = cv2.resize(enc_map,(self.target_size[1],self.target_size[0]),interpolation=cv2.INTER_NEAREST)
                enc_map = np.repeat(enc_map[:,:,None],3,-1)
                max_value = max(max_value,np.max(enc_map))
                enc_maps.append(enc_map)
                border = np.zeros((self.target_size[0],spacer,3),dtype=np.uint8)
                border[:,:,0] = -1
                enc_maps.append(border)

            enc_maps = np.concatenate((enc_maps),axis=1)

            img_empty = cv2.imread(str(self.output_dir.parents[4] / 'examples' / 'empty_chamber.png'))
            

            enc_maps[enc_maps!=-1] = (enc_maps[enc_maps!=-1] / max_value) * 255
            enc_maps[enc_maps==-1] = 255
            enc_maps = enc_maps[:,:-spacer]

            if img_empty is not None:
                enc_maps = np.concatenate((img_empty,enc_maps[:,self.target_size[1]:self.target_size[1]+spacer],enc_maps),axis=1)
            else:
                enc_maps = np.concatenate((enc_maps[:,self.target_size[1]:self.target_size[1]+spacer],enc_maps),axis=1)

            cv2.imwrite(str(self.output_dir / self.predictions_folder / 'all_enc_queries_picked.png'),enc_maps.astype(np.uint8))

        if self.eval_ctc:
            np.savetxt(self.output_dir / self.predictions_folder / 'res_track.txt',ctc_data,fmt='%d')

        if self.write_video:
            crf = 20
            verbose = 1

            if self.eval_ctc or self.args.dataset == '2D':
                filename = self.output_dir / self.predictions_folder / (f'{self.videoname_list[r]}_movie.mp4')
            else:
                filename = self.output_dir / self.predictions_folder / (f'movie.mp4')                

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
                img_empty = cv2.imread(str(self.output_dir.parents[1] / 'examples' / 'empty_chamber.png'))
                img_empty = cv2.resize(img_empty,(self.target_size[1]*scale,self.target_size[0]*scale))
            
                for box in self.query_box_locations[ind][1:]:
                    img_empty = cv2.circle(img_empty, (int(box[0]*scale),int(box[1]*scale)), radius=1*scale, color=(255,0,0), thickness=-1)

                img_empty = np.concatenate((np.ones((hspacer,self.target_size[1]*scale,3),dtype=np.uint8)*255,img_empty),axis=0)
                shift = 5 if ind + 1 >= 10 else 12
                img_empty = cv2.putText(img_empty,f'{ind+1}',org=(shift*scale,15*scale),fontFace=cv2.FONT_HERSHEY_COMPLEX,fontScale=4,color=(0,0,0),thickness=4)
                query_frames[:,j*(self.target_size[1]*scale+wspacer): j*(self.target_size[1]*scale+wspacer) + self.target_size[1]*scale] = img_empty

            cv2.imwrite(str(self.output_dir / self.predictions_folder / (f'{method}_object_query_box_locations.png')),query_frames)

            if self.use_dab:
                height,width = self.target_size[0] * scale, self.target_size[1] * scale
                boxes = aux_outputs[0]['pred_boxes'][0,:,:4].detach().cpu().numpy()

                boxes[:,1::2] = boxes[:,1::2] * height
                boxes[:,::2] = boxes[:,::2] * width

                boxes[:,0] = boxes[:,0] - boxes[:,2] // 2
                boxes[:,1] = boxes[:,1] - boxes[:,3] // 2

                for j,ind in enumerate(where_boxes):

                        bounding_box = boxes[ind]

                        query_frames = cv2.rectangle(
                        query_frames,
                        (int(np.clip(bounding_box[0],0,width)) + j * (width + wspacer), int(np.clip(bounding_box[1],0,height))+hspacer),
                        (int(np.clip(bounding_box[0] + bounding_box[2],0,width)) + j * (width + wspacer), int(np.clip(bounding_box[1] + bounding_box[3],0,height))+hspacer),
                        color=tuple(np.array([50.,50.,50.])),
                        thickness = 5)

                cv2.imwrite(str(self.output_dir / self.predictions_folder / (f'{method}_object_query_box_locations_with_boxes.png')),query_frames)
        


@torch.no_grad()
def print_worst(model, criterion, data_loader_train, data_loader_val, device, args, track=True):
    model.eval()
    model._tracking = True
    criterion.eval()

    if track:
        save_folder = 'save_worst_predictions_track'

        if args.use_prev_prev_frame:
            save_folder += '_prev_prev' 

    else:
        save_folder = 'save_worst_predcitions_object_det'

    if not args.no_data_aug:
        save_folder += '_data_aug'

    train_folder = 'train_outputs'
    val_folder = 'val_outputs'

    (args.output_dir / save_folder).mkdir(exist_ok=True)
    (args.output_dir / save_folder / train_folder).mkdir(exist_ok=True)
    (args.output_dir / save_folder / val_folder).mkdir(exist_ok=True)

    folders = ['worst_loss','worst_score']

    for folder in folders:
        (args.output_dir / save_folder / train_folder / folder).mkdir(exist_ok=True)
        (args.output_dir / save_folder / val_folder / folder).mkdir(exist_ok=True)

        (args.output_dir / save_folder / train_folder / folder / 'standard').mkdir(exist_ok=True)
        (args.output_dir / save_folder / val_folder / folder / 'standard').mkdir(exist_ok=True)

        if args.two_stage:
            (args.output_dir / save_folder / train_folder / folder / 'enc_outputs').mkdir(exist_ok=True)
            (args.output_dir / save_folder / val_folder / folder / 'enc_outputs').mkdir(exist_ok=True)

        if args.dn_track:
            (args.output_dir / save_folder / train_folder / folder / 'dn_track').mkdir(exist_ok=True)
            (args.output_dir / save_folder / val_folder / folder / 'dn_track').mkdir(exist_ok=True)

        if args.dn_object:
            (args.output_dir / save_folder / train_folder / folder / 'dn_object').mkdir(exist_ok=True)
            (args.output_dir / save_folder / val_folder / folder / 'dn_object').mkdir(exist_ok=True)

        if args.dn_enc:
            (args.output_dir / save_folder / train_folder / folder / 'dn_enc').mkdir(exist_ok=True)
            (args.output_dir / save_folder / val_folder / folder / 'dn_enc').mkdir(exist_ok=True)

        if args.dn_track_group:
            (args.output_dir / save_folder / train_folder / folder / 'dn_track_group').mkdir(exist_ok=True)
            (args.output_dir / save_folder / val_folder / folder / 'dn_track_group').mkdir(exist_ok=True)

    datasets = ['train','val']

    for didx, data_loader in enumerate([data_loader_train,data_loader_val]):

        store_loss = torch.zeros((len(data_loader)))
        store_score = torch.zeros((len(data_loader)))

        for idx, (samples, targets) in enumerate(data_loader):

            # if idx > 100:
            #     continue

            samples = samples.to(device)
            targets = [utils.nested_dict_to_device(t, device) for t in targets]

            dataset_id = targets[0]['dataset_nb']
            framenb = targets[0]['cur_target']['framenb']

            # if not ((dataset_id == 73 and framenb == 10)):
            #     continue

            outputs, targets, features, memory, hs, prev_outputs = model(samples,targets,track=track)

            cur_targets = [target['cur_target'] for target in targets]

            groups = [0]

            outputs, loss_dict, groups = calc_loss_for_training_methods('cur_target', outputs, groups, targets, criterion, args.epochs, args)

            weight_dict = criterion.weight_dict

            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            store_loss[idx] = losses.item()

            if targets[0]['track']:
                acc_dict = utils.calc_track_acc(outputs,cur_targets,args)
                if cur_targets[0]['empty']:
                    score = 1.0 if acc_dict['overall_track_acc'][0,0,1] == 0 else 0.
                else:
                    score = acc_dict['overall_track_acc'][0,0,0]/acc_dict['overall_track_acc'][0,0,1]
                print(f"{datasets[didx]} {idx}/{len(data_loader)} Dataset: {dataset_id:02d} Framenb: {framenb}  track: {score:.2f}")
                store_score[idx] = score
            else:
                acc_dict = utils.calc_bbox_acc(outputs,cur_targets,args)
                if cur_targets[0]['empty']:
                    score = 1.0 if acc_dict['mask_det_acc'][0,0,1] == 0 else 0.
                else:
                    score = acc_dict['mask_det_acc'][0,0,0]/acc_dict['mask_det_acc'][0,0,1]
                score = acc_dict['mask_det_acc'][0,0,0]/acc_dict['mask_det_acc'][0,0,1]
                print(f"{datasets[didx]} {idx}/{len(data_loader)} Dataset: {dataset_id:02d} Framenb: {framenb}  det: {acc_dict['bbox_det_acc'][0,0,0]/acc_dict['bbox_det_acc'][0,0,1]:.2f}  seg: {score:.2f}")
                store_score[idx] = score

        worst_loss_ind = torch.argsort(store_loss)[-100:].flip(0)
        worst_score_ind = torch.argsort(store_score)[:50]

        print('Saving worst predctions...')

        for idx, (samples, targets) in enumerate(data_loader):
            
            if idx not in worst_loss_ind and idx not in worst_score_ind:
                continue

            samples = samples.to(device)
            targets = [utils.nested_dict_to_device(t, device) for t in targets]

            outputs, targets, features, memory, hs, prev_outputs = model(samples,targets,track=track)

            training_methods =  outputs['training_methods'] # dn_object, dn_track, dn_enc

            meta_data = {}
            groups = [0]

            for training_method in training_methods:
                outputs_TM, loss_dict_TM, groups = calc_loss_for_training_methods(training_method, outputs, groups, targets, criterion, args.epochs, args)

                meta_data[training_method] = {}
                meta_data[training_method]['outputs'] = outputs_TM
                meta_data[training_method]['loss_dict'] = loss_dict_TM

            outputs = meta_data['cur_target']['outputs'] 
            loss_dict = meta_data['cur_target']['loss_dict']

            weight_dict = criterion.weight_dict

            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            dataset_id = targets[0]['dataset_nb']
            framenb = targets[0]['cur_target']['framenb']

            if not np.round(losses.item(),3) == np.round(store_loss[idx],3):
                print(idx, np.round(losses.item(),3),np.round(store_loss[idx],3))
            # else:
            #     print(f"{idx} loss: {np.round(store_loss[idx],3)} {f'({np.where(worst_loss_ind == idx)[0][0]})' if idx in worst_loss_ind else ''} score: {np.round(store_score[idx],3)} {f'({np.where(worst_score_ind == idx)[0][0]})' if idx in worst_score_ind else ''}) ")

            if idx in worst_loss_ind:
                ind = np.where(worst_loss_ind == idx)[0][0]
                data_viz.plot_results(outputs, prev_outputs, targets,samples.tensors, args.output_dir / save_folder, folder = Path(datasets[didx] + '_outputs') / 'worst_loss', filename = f'{ind+1:03d}_Loss_{store_loss[idx]:06.2f}_dataset_{dataset_id:02d}_framenb_{framenb}.png', args=args, meta_data=meta_data)
            
            if idx in worst_score_ind:
                ind = np.where(worst_score_ind == idx)[0][0]
                data_viz.plot_results(outputs, prev_outputs, targets,samples.tensors, args.output_dir / save_folder, folder = Path(datasets[didx] + '_outputs') / 'worst_score', filename = f'{ind+1:03d}_Score_{store_score[idx]:06.2f}_dataset_{dataset_id:02d}_framenb_{framenb}.png', args=args, meta_data=meta_data)

        print(f'Done save worst predictions for {datasets[didx]} dataset')