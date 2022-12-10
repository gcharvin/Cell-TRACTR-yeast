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

def split_outputs(outputs,indices,new_outputs=None,update_masks=False):

    if new_outputs is None:
        new_outputs = outputs
    new_outputs['pred_logits'] = outputs['pred_logits'][:,indices[0]:indices[1]]
    new_outputs['pred_boxes'] = outputs['pred_boxes'][:,indices[0]:indices[1]]

    if update_masks:
        new_outputs['pred_mask'] = outputs['pred_mask'][:,indices[0]:indices[1]]

    if 'aux_outputs' in outputs:
        for lid in range(len(outputs['aux_outputs'])):
            new_outputs['aux_outputs'][lid]['pred_logits'] = outputs['aux_outputs'][lid]['pred_logits'][:,indices[0]:indices[1]]
            new_outputs['aux_outputs'][lid]['pred_boxes'] = outputs['aux_outputs'][lid]['pred_boxes'][:,indices[0]:indices[1]]

            if update_masks:
                new_outputs['aux_outputs'][lid]['pred_mask'] = outputs['aux_outputs'][lid]['pred_mask'][:,indices[0]:indices[1]]

    return new_outputs

def calc_loss_for_training_methods(training_method:str,
                                   outputs,
                                   groups,
                                   targets,
                                   criterion,
                                   masks=False):
    outputs_TM = {}
    outputs_TM['aux_outputs'] = [{} for _ in range(len(outputs['aux_outputs']))]

    groups.append(groups[-1] + targets[0][training_method]['num_queries'])

    outputs_TM = split_outputs(outputs,groups[-2:],outputs_TM,update_masks=masks)

    loss_dict_TM = criterion(outputs_TM, [target[training_method] for target in targets],return_bbox_track_acc=False)

    return outputs_TM, loss_dict_TM, groups

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, postprocessors,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, args, num_plots=10, interval = 50):
    dataset = 'train'
    model.train()
    criterion.train()

    ids = np.random.randint(0,len(data_loader),num_plots)
    ids = np.concatenate((ids,[0]))

    metrics_dict = {}

    for i, (samples, targets) in enumerate(data_loader):

        samples = samples.to(device)
        targets = [utils.nested_dict_to_device(t, device) for t in targets]

        outputs, targets, features, memory, hs, prev_outputs = model(samples,targets)

        training_methods = outputs['training_methods'] # group_object, dn_object, dn_track

        meta_data = {}
    
        groups = [0]
        groups.append(groups[-1] + len(targets[0]['track_queries_mask']))

        for training_method in training_methods:
            meta_data[training_method] = {}
            outputs_TM, loss_dict_TM, groups = calc_loss_for_training_methods(training_method, outputs, groups, targets, criterion, args.masks)

            meta_data[training_method]['outputs'] = outputs_TM
            meta_data[training_method]['loss_dict'] = loss_dict_TM

        outputs = split_outputs(outputs,groups[:2],new_outputs=None,update_masks=args.masks)


        loss_dict, acc_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        loss_dict_keys = list(loss_dict.keys())

        for training_method in training_methods:
            for loss_dict_key in loss_dict_keys:
                loss_dict[loss_dict_key + '_' + training_method] = meta_data[training_method]['loss_dict'][loss_dict_key] 
                weight_dict[loss_dict_key + '_' + training_method] = weight_dict[loss_dict_key] * args.group_object_coef

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

        metrics_dict = utils.update_metrics_dict(metrics_dict,acc_dict,loss_dict,weight_dict,i)

        dict_shape = metrics_dict['loss_ce'].shape
        for metrics_dict_key in metrics_dict.keys():
            assert metrics_dict[metrics_dict_key].shape[:2] == dict_shape, 'Metrics needed to be added per epoch'


        if i in ids and (epoch % 5 == 0 or epoch == 1):
            utils.plot_results(outputs, prev_outputs, targets,samples, args.output_dir, train=True, filename = f'Epoch{epoch:03d}_Step{i:06d}.png', meta_data=meta_data)

        if i > 0 and i % interval == 0:
            utils.display_loss(metrics_dict,i,len(data_loader),epoch=epoch,dataset=dataset)

    utils.save_metrics_pkl(metrics_dict,args.output_dir,dataset=dataset)  


@torch.no_grad()
def evaluate(model, criterion, data_loader, device, output_dir: str, 
             args, epoch: int = None, train=False, interval=50):
    model.eval()
    criterion.eval()
    dataset = 'val'
    num_plots = 10
    ids = np.random.randint(0,len(data_loader),num_plots)
    ids = np.concatenate((ids,[0]))


    metrics_dict = {}
    for i, (samples, targets) in enumerate(data_loader):
        samples = samples.to(device)
        targets = [utils.nested_dict_to_device(t, device) for t in targets]

        outputs, targets, features, memory, hs, prev_outputs = model(samples,targets)

        training_methods = outputs['training_methods'] # group_object, dn_object, dn_track

        meta_data = {}

        groups = [0]
        groups.append(groups[-1] + len(targets[0]['track_queries_mask']))

        for training_method in training_methods:
            meta_data[training_method] = {}
            outputs_TM, loss_dict_TM, groups = calc_loss_for_training_methods(training_method, outputs, groups, targets, criterion, args.masks)

            meta_data[training_method]['outputs'] = outputs_TM
            meta_data[training_method]['loss_dict'] = loss_dict_TM

        outputs = split_outputs(outputs,groups[:2],new_outputs=None,update_masks=args.masks)

        loss_dict, acc_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        loss_dict_keys = list(loss_dict.keys())

        for training_method in training_methods:
            for loss_dict_key in loss_dict_keys:
                loss_dict[loss_dict_key + '_' + training_method] = meta_data[training_method]['loss_dict'][loss_dict_key] 
                weight_dict[loss_dict_key + '_' + training_method] = weight_dict[loss_dict_key] * args.group_object_coef

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        loss_dict['loss'] = losses
        weight_dict['loss'] = 1

        metrics_dict = utils.update_metrics_dict(metrics_dict,acc_dict,loss_dict,weight_dict,i)

        if i in ids and (epoch % 5 == 0 or epoch == 1) and train:
            utils.plot_results(outputs, prev_outputs, targets,samples, savepath = output_dir, train=False, filename = f'Epoch{epoch:03d}_Step{i:06d}.png', meta_data=meta_data)

        if i > 0 and i % interval == 0:
            utils.display_loss(metrics_dict,i,len(data_loader),epoch=epoch,dataset=dataset)

    utils.save_metrics_pkl(metrics_dict,args.output_dir,dataset=dataset)  


@torch.no_grad()
def run_pipeline(model, fps, device, output_dir, args):
    model.tracking()
    (output_dir / 'predictions').mkdir(exist_ok=True)
    normalize = Compose([
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    colors = np.array([tuple((255*np.random.random(3))) for _ in range(1000)]) # Assume max 1000 cells in one chamber
    colors[0] = [255,0,0]
    colors[1] = [0,0,255]
    colors[2] = [0,255,0]
    colors[3] = [255,255,0]
    colors[4] = [255,0,255]
    colors[5] = [0,255,255]

    threshold = 0.5

    target_size = (256,32)
    fps_split = []
    videoname_list = []
    print_query_boxes = True
    write_video = True
    track = True
    print_decoder_aux = True
    print_masks = False

    if print_decoder_aux:
        (output_dir / 'predictions' / 'decoder_bbox_outputs').mkdir(exist_ok=True)
        (output_dir / 'predictions' / 'ref_pts_outputs').mkdir(exist_ok=True)
        num_decoder_frames = 1
    counter = -1
    old_filename = ''
    for fp in fps:
        frame_nb = re.findall('\d+',fp.stem)[-1]
        filename = fp.name.replace(frame_nb+'.png','')

        if filename != old_filename:
            fps_split.append([fp])
            counter += 1
            videoname_list.append(filename)
        else:
            fps_split[counter].append(fp)

        old_filename = filename

    max_len = max([len(fp_split) for fp_split in fps_split])

    color_stack = np.zeros((max_len,target_size[0],target_size[1]*len(videoname_list),3))

    if print_query_boxes:
        query_box_locations = [np.zeros((1,4)) for i in range(args.num_queries)]

    for r,fps_ROI in enumerate(fps_split):
        print(videoname_list[r])
        print(f'video {r+1}/{len(fps_split)}')
        max_cellnb = 0
        targets = [{}]
        prev_features = None

        if print_decoder_aux:
            random_nbs = np.random.choice(len(fps_ROI),num_decoder_frames)
            random_nbs = np.concatenate((random_nbs,random_nbs+1))

        for i, fp in enumerate(tqdm(fps_ROI)):

            if not track:
                targets = [{}]
                colors = np.array([tuple((255*np.zeros(3))) for _ in range(1000)])
            img = PIL.Image.open(fp,mode='r').resize((target_size[1],target_size[0])).convert('RGB')

            previmg = PIL.Image.open(fps_ROI[i-1],mode='r').resize((target_size[1],target_size[0])).convert('RGB') if i > 0 else None

            samples = normalize(img)[0][None]

            samples = samples.to(device)
            outputs, targets, prev_features, memory, hs, prev_outputs = model(samples,targets=targets,prev_features=prev_features)

            pred_logits = outputs['pred_logits'][0].sigmoid().detach().cpu().numpy()

            keep = (pred_logits[:,0] > threshold)
            keep_div = (pred_logits[:,1] > threshold)

            #### consider updating this code; particular for object detection ####
            # keep_div[-args.num_queries:] = False
            ####

            cells = np.zeros((sum(keep)),dtype=np.int)
            masks = None

            if print_query_boxes:
                all_object_boxes = outputs['pred_boxes'][0,-args.num_queries:].detach().cpu().numpy()
                bbs = all_object_boxes[keep[-args.num_queries:],:4]

                bbs[:,1::2] = bbs[:,1::2] * target_size[0]
                bbs[:,::2] = bbs[:,::2] * target_size[1]

                bbs[:,::2] = np.clip(bbs[:,::2],0,target_size[1])
                bbs[:,1::2] = np.clip(bbs[:,1::2],0,target_size[0])

                where_keep = np.where(keep[-args.num_queries:] == True)[0]
                for k,ind in enumerate(where_keep):
                    query_box_locations[ind] = np.append(query_box_locations[ind], bbs[k:k+1],axis=0)

            if sum((pred_logits[~keep,1] > 0.5) * (pred_logits[~keep,0] > 0.5) > 0):
                print('cell division for not tracked cell')

            if sum((pred_logits[:,1] > 0.5) * (pred_logits[:,0] < 0.5)):
                print('cell tracked to second slot not first')

            if sum(keep) > 0:
                track_ind = np.where(keep==True)[0]
                track_ind_copy = track_ind.copy()
                track_div_ind = np.where(keep_div==True)[0]

                div_track = np.zeros((sum(keep) + sum(keep_div)))
                for track_div_idx in track_div_ind:
                    ind = np.where(track_ind==track_div_idx)[0][0]
                    track_ind = np.concatenate((track_ind[:ind],track_ind[ind:ind+1],track_ind[ind:]))
                
                    div_track[ind:ind+2] = track_div_idx+1

                targets[0]['track_query_hs_embeds'] = outputs['hs_embed'][0,track_ind]
                boxes = outputs['pred_boxes'][0,track_ind]

                if print_masks and 'pred_masks' in outputs:
                    masks = outputs['pred_masks'][0,track_ind].sigmoid()
                
                if len(track_div_ind) > 0:
                    for k in range(1,boxes.shape[0]):
                        if (boxes[k-1] == boxes[k]).all():
                            boxes[k,:4] = boxes[k-1,4:]

                            if masks is not None:
                                masks[k,:1] = masks[k-1,1:] 

                boxes = boxes[:,:4]
    
                if masks is not None:
                    masks = masks[:,0]

                targets[0]['track_query_boxes'] = boxes[:,:4]

                if outputs['pred_logits'].shape[1] > args.num_queries:

                    track_keep = pred_logits[:len(prevcells),0] > threshold
                    cells[:sum(track_keep)] = prevcells[track_keep]

                    for div_idx,track_div_idx in enumerate(track_div_ind):
                        idx= np.where(track_ind_copy==track_div_idx)[0][0]
                        cells = np.concatenate((cells[:idx+1+div_idx],[max_cellnb+1],cells[idx+1+div_idx:]))
                        max_cellnb += 1             

                    new_cells = cells == 0
                    cells[cells==0] = np.arange(max_cellnb+1,max_cellnb+1+sum(cells==0),dtype=np.int)
                elif sum(keep_div) > 0:
                    for div_idx,track_div_idx in enumerate(track_div_ind):
                        idx= np.where(track_ind_copy==track_div_idx)[0][0]
                        cells = np.concatenate((cells[:idx+1+div_idx],[max_cellnb+1],cells[idx+1+div_idx:]))
                        max_cellnb += 1             

                    new_cells = None
                    cells[cells==0] = np.arange(max_cellnb+1,max_cellnb+1+sum(cells==0),dtype=np.int)
                else:
                    cells[:] = np.arange(max_cellnb+1,max_cellnb+1+len(cells),dtype=np.int)
                    new_cells = None

                max_cellnb = max(max_cellnb,np.max(cells))
                if outputs['pred_logits'].shape[1] > args.num_queries:
                    store__prevcells = np.copy(prevcells)
                prevcells = np.copy(cells)
            else:
                div_track = None
                boxes = None
                new_cells = None

            assert boxes.shape[0] == len(cells)

            if track:
                color_frame = utils.plot_tracking_results(img,boxes,masks,colors[cells-1],cells,div_track,new_cells,track)
            else:
                color_frame = utils.plot_tracking_results(img,boxes,masks,colors[:len(cells)],cells,div_track,new_cells,track)

            color_stack[i,:,r*target_size[1]:(r+1)*target_size[1]] = color_frame         

            if print_decoder_aux and i in random_nbs:
                references = outputs['references']

                aux_outputs =outputs['aux_outputs']
                aux_outputs =  [{'pred_boxes':references[0]}] + aux_outputs
                aux_outputs.append({'pred_boxes':outputs['pred_boxes']})
                decoder_frame = np.zeros((target_size[0],target_size[1] * len(aux_outputs),3))

                if track:
                    color = colors[cells-1]
                else:
                    color = np.array([(np.array([255.,0.,0.])) for _ in range(len(cells))]) 

                for a,aux_output in enumerate(aux_outputs):
                    boxes = aux_output['pred_boxes'][0]

                    if a != 0:
                        if sum(keep) > 0:
                            boxes = boxes[track_ind]
                            if len(track_div_ind) > 0:
                                for k in range(1,boxes.shape[0]):
                                    if (boxes[k-1] == boxes[k]).all():
                                        boxes[k,:4] = boxes[k-1,4:]

                            boxes = boxes[:,:4]

                            decoder_frame[:,target_size[1]*a:target_size[1]*(a+1)] = utils.plot_tracking_results(img,boxes,None,color,cells,div_track,new_cells,track)
                        else:
                            decoder_frame[:,target_size[1]*a:target_size[1]*(a+1)] = np.array(img)

                    else:
                        if sum(keep) > 0:
                            boxes = boxes[track_ind]

                            img_ref_box = np.copy(np.array(img))
                            for ridx in range(boxes.shape[0]):
                                if args.use_dab:
                                    x,y,w,h = boxes[ridx]
                                else:
                                    x,y - boxes[ridx]

                                if not(div_track[ridx-1] and  div_track[ridx]):
                                    img_ref_box = cv2.circle(img_ref_box, (int(x*target_size[1]),int(y*target_size[0])), radius=1, color=color[ridx], thickness=-1)

                                decoder_frame[:,target_size[1]*a:target_size[1]*(a+1)] = img_ref_box
                        else:
                            decoder_frame[:,target_size[1]*a:target_size[1]*(a+1)] = np.array(img)

                    if a == len(aux_outputs) - 1:
                        # boxes = torch.cat((aux_output['pred_boxes'][0,-args.num_queries:,:4],boxes),axis=0)
                        boxes = aux_output['pred_boxes'][0]

                        prevcells_only_ind = torch.tensor([cidx for cidx,c in enumerate(store__prevcells) if c not in cells])
                        prevcells_only = torch.tensor([c for cidx,c in enumerate(store__prevcells) if c not in cells])

                        boxes_track = boxes[track_ind]
                        if len(track_div_ind) > 0:
                            for k in range(1,boxes.shape[0]):
                                if (boxes[k-1] == boxes[k]).all():
                                    boxes[k,:4] = boxes[k-1,4:]
                        boxes_track = boxes_track[:,:4]

                        if len(prevcells_only) > 0:
                            boxes_prev_only = boxes[prevcells_only_ind,:4]
                            boxes = torch.cat((boxes[-args.num_queries:,:4],boxes_track,boxes_prev_only))
                            color_prev = colors[prevcells_only - 1] 
                            if color_prev.ndim == 1:
                                color_prev = color_prev[None,:]
                            color = np.concatenate((np.array([(np.array([0.,0.,255.])) for _ in range(args.num_queries)]),color,color_prev),axis=0)
                            div_track_all = np.concatenate((np.zeros((args.num_queries)),div_track,torch.zeros((len(boxes_prev_only)))))

                        else:
                            boxes = torch.cat((boxes[-args.num_queries:,:4],boxes_track))
                            color = np.concatenate((np.array([(np.array([0.,0.,255.])) for _ in range(args.num_queries)]),color),axis=0)
                            div_track_all = np.concatenate((np.zeros((args.num_queries)),div_track))

                        new_cell_thickness = np.zeros_like(div_track_all).astype(bool)

                        if new_cells is not None:
                            new_cell_thickness[args.num_queries:] = True
                        img_ref_box = utils.plot_tracking_results(img,boxes,None,color,cells,div_track_all,new_cell_thickness,track)

                        color = np.array([(np.array([0.,0.,255.])) for _ in range(boxes.shape[0])])

                        img_ref_box_all_object = utils.plot_tracking_results(img,boxes,None,color,cells,None,None,track)
                        
                        decoder_frame = np.concatenate((decoder_frame,img_ref_box,img_ref_box_all_object),axis=1)

                method = 'object_detection' if not track else 'track'
                cv2.imwrite(str(output_dir / 'predictions' / 'decoder_bbox_outputs' / (f'{method}_decoder_frame_{fp.name}')),decoder_frame)

                
                ref_frames = np.zeros((target_size[0],target_size[1] * references.shape[0],3))

                img_ref_pts_init_object = np.copy(np.array(img))
                img_ref_pts_init_track = np.copy(np.array(img))
                previmg_ref_pts_init_track = np.copy(np.array(previmg))
                img_ref_pts_init_all = np.copy(np.array(img))
                img_ref_pts_final_all = np.copy(np.array(img))
                empty_img = np.copy(np.array(img))

                for index,ref in enumerate(references):
                    ref_pts = ref[0]
                    img_ref_pts_update = np.copy(np.array(img))
                    count = -1
                    counter = 0
                    if outputs['pred_logits'].shape[1] > args.num_queries:
                        prevcells_only = [c for c in store__prevcells if c not in cells]
                    for ridx in range(len(ref_pts)):
                        x = ref_pts[-(ridx+1)][0]
                        y = ref_pts[-(ridx+1)][1]
                        radius = 1
                        if track:
                            if ridx < args.num_queries:
                                color = (0,0,255)
                            elif keep[-(ridx+1)]:
                                color = colors[cells[count] - 1]
                                count -= 1
                                radius = 2
                            else:
                                color = colors[prevcells_only[counter] - 1]
                                counter += 1
                                radius = 2
                                # color = (255,0,0)
                        else:
                            if keep[-(ridx+1)]:
                                color = (255,0,0)
                            else:
                                color = (0,0,255)

                        img_ref_pts_update = cv2.circle(img_ref_pts_update, (int(x*target_size[1]),int(y*target_size[0])), radius=radius, color=color, thickness=-1)

                        if index == 0 and track and ridx >= args.num_queries:
                            img_ref_pts_init_track = cv2.circle(img_ref_pts_init_track, (int(x*target_size[1]),int(y*target_size[0])), radius=radius, color=color, thickness=-1)
                            previmg_ref_pts_init_track = cv2.circle(previmg_ref_pts_init_track, (int(x*target_size[1]),int(y*target_size[0])), radius=radius, color=color, thickness=-1)

                        if index == 0 and ridx < args.num_queries:
                            img_ref_pts_init_object = cv2.circle(img_ref_pts_init_object, (int(x*target_size[1]),int(y*target_size[0])), radius=radius, color=color, thickness=-1)

                        if index == 0:
                            img_ref_pts_init_all = cv2.circle(img_ref_pts_init_all, (int(x*target_size[1]),int(y*target_size[0])), radius=radius, color=(0,0,255), thickness=-1)

                        if index == len(references) - 1:
                            img_ref_pts_final_all = cv2.circle(img_ref_pts_final_all, (int(x*target_size[1]),int(y*target_size[0])), radius=1, color=(0,0,255), thickness=-1)

                    ref_frames[:,target_size[1]*index:target_size[1]*(index+1)] = img_ref_pts_update

                if previmg is not None and track:
                    ref_pts = references[0,0]
                    for ridx in range(ref_pts.shape[0]):
                        if args.use_dab:
                            x,y,w,h = ref_pts[ridx]
                        else:
                            x,y = ref_pts[ridx]

                        if ridx < ref_pts.shape[0] - args.num_queries:
                            color = (255,0,0)
                        
                            previmg = cv2.circle(np.array(previmg), (int(x*target_size[1]),int(y*target_size[0])), radius=2, color=color, thickness=-1)

                    ref_frames = np.concatenate((previmg,ref_frames),axis=1)


                if track:
                    ref_frames = np.concatenate((empty_img,img_ref_pts_init_all,previmg_ref_pts_init_track,img_ref_pts_init_track,img_ref_pts_init_object,ref_frames,img_ref_pts_final_all),axis=1)
                else:
                    ref_frames = np.concatenate((empty_img,img_ref_pts_init_all,ref_frames,img_ref_pts_final_all),axis=1)

                cv2.imwrite(str(output_dir / 'predictions' / 'ref_pts_outputs' / (f'{method}_ref_pts_{fp.name}')),ref_frames)
                   

    if write_video:
        crf = 20
        verbose = 1
        method = 'track' if track else 'object_detection'
        name_mask = 'mask_' if print_masks else ''
        filename = output_dir / 'predictions' / (f'{videoname_list[r]}_{method}_{name_mask}video.mp4')
        print(filename)
        height, width, _ = color_stack[0].shape
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
        for frame in color_stack:
            process.stdin.write(frame[:height, :width].astype(np.uint8).tobytes())

        # Close file stream:
        process.stdin.close()

        # Wait for processing + close to complete:
        process.wait()

    if print_query_boxes:

        scale = 8
        wspacer = 5 * scale
        hspacer = 20 * scale

        max_area = [np.max(boxes[:,2] * boxes[:,3]) for boxes in query_box_locations]
        num_boxes_used = np.sum(np.array(max_area) > 0)
        query_frames = np.ones((target_size[0]*scale + hspacer, (target_size[1]*scale + wspacer) * num_boxes_used,3),dtype=np.uint8) * 255
        where_boxes = np.where(np.array(max_area) > 0)[0]

        for j,ind in enumerate(where_boxes):
            img_empty = cv2.imread(str(output_dir.parents[1] / 'empty_chamber' / 'img.png'))
            img_empty = cv2.resize(img_empty,(target_size[1]*scale,target_size[0]*scale))
           
            for box in query_box_locations[ind][1:]:
                img_empty = cv2.circle(img_empty, (int(box[0]*scale),int(box[1]*scale)), radius=1*scale, color=(255,0,0), thickness=-1)

            img_empty = np.concatenate((np.ones((hspacer,target_size[1]*scale,3),dtype=np.uint8)*255,img_empty),axis=0)
            shift = 5 if ind + 1 >= 10 else 12
            img_empty = cv2.putText(img_empty,f'{ind+1}',org=(shift*scale,15*scale),fontFace=cv2.FONT_HERSHEY_COMPLEX,fontScale=4,color=(0,0,0),thickness=4)
            query_frames[:,j*(target_size[1]*scale+wspacer): j*(target_size[1]*scale+wspacer) + target_size[1]*scale] = img_empty

        cv2.imwrite(str(output_dir / 'predictions' / (f'{method}_object_query_box_locations.png')),query_frames)
        




        
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