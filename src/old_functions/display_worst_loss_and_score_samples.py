# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import random
from argparse import Namespace
from pathlib import Path
import numpy as np
import sacred
import torch
from torch.utils.data import DataLoader, DistributedSampler

import trackformer.util.misc as utils
from trackformer.engine import calc_loss_for_training_methods
from trackformer.models import build_model
from trackformer.util.misc import nested_dict_to_namespace
from trackformer.util import data_viz
from trackformer.datasets import build_dataset




modelname = '240326_moma_Track_no_flex_div_CoMOT_two_stage_dn_track_dn_track_group_dab_intermediate_mask_OD_decoder_layer_use_box_as_div_ref_pts_4_enc_4_dec_layers_backprop_prev'

modelpath = Path('/projectnb/dunlop/ooconnor/MOT/models/cell-trackformer/results') / modelname

ex = sacred.Experiment('pipeline')
ex.add_config(modelpath.as_posix() + '/config.yaml')

def train(args: Namespace) -> None:

    display_all_aux_outputs = False
    args.display_decoder_aux = False

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
        
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    model, criterion = build_model(args)
    model.to(device)
    model.train_model = True
    criterion.eval_only = True

    model_without_ddp = model
    model_without_ddp = utils.load_model(model_without_ddp,args)

    model.eval_prev_prev_frame = args.use_prev_prev_frame
    model.no_data_aug = args.no_data_aug
    
    dataset_train = build_dataset(split='train', args=args)
    dataset_val = build_dataset(split='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train, shuffle=False)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.SequentialSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = DataLoader(
        dataset_train,
        batch_size = 1,
        sampler=sampler_train,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers)
        
    data_loader_val = DataLoader(
        dataset_val, 
        batch_size = 1,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers)

    if args.tracking:
        save_folder = 'save_worst_predictions_track'

        if args.use_prev_prev_frame:
            save_folder += '_prev_prev' 

    else:
        save_folder = 'save_worst_predcitions_object_det'

    args.output_dir = args.output_dir / save_folder

    (args.output_dir).mkdir(exist_ok=True)
    (args.output_dir / train_folder).mkdir(exist_ok=True)
    (args.output_dir / val_folder).mkdir(exist_ok=True)

    folders = ['worst_loss','worst_score']
    train_folder = 'train_outputs'
    val_folder = 'val_outputs'

    for folder in folders:
        utils.create_folders(train_folder,val_folder,args)

    datasets = ['train','val']

    with torch.no_grad():
        for didx, data_loader in enumerate([data_loader_train,data_loader_val]):

            store_loss = torch.zeros((len(data_loader)))
            store_score = torch.zeros((len(data_loader)))

            for idx, (samples, targets) in enumerate(data_loader):

                samples = samples.to(device)
                targets = [utils.nested_dict_to_device(t, device) for t in targets]

                dataset_id = targets[0]['dataset_nb']
                framenb = targets[0]['main']['cur_target']['framenb']

                outputs, targets, _, _, _ = model(samples,targets)
                prev_outputs = outputs['prev_outputs'] if 'prev_outputs' in outputs else None
                outputs, loss_dict = calc_loss_for_training_methods(outputs, targets, criterion)

                weight_dict = criterion.weight_dict

                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
                store_loss[idx] = losses.item()
                
                main_targets = [target['main']['cur_target'] for target in targets]

                acc_dict = {}
                if args.tracking:
                    acc_dict = utils.calc_track_acc(acc_dict,outputs['main'],main_targets,args,calc_mask_acc=True)
                    if main_targets[0]['empty']:
                        score = 1.0 if acc_dict['track_mask_acc'][0,0,1] == 0 else 0.
                    else:
                        score = acc_dict['track_mask_acc'][0,0,0]/acc_dict['track_mask_acc'][0,0,1]
                    print(f"{datasets[didx]} {idx}/{len(data_loader)} Dataset: {dataset_id:02d} Framenb: {framenb}  track: {score:.2f}")
                    store_score[idx] = score

                else:
                    acc_dict = utils.calc_bbox_acc(acc_dict,outputs['main'],main_targets,args)
                    if main_targets[0]['empty']:
                        score = 1.0 if acc_dict['det_mask_acc'][0,0,1] == 0 else 0.
                    else:
                        score = acc_dict['det_mask_acc'][0,0,0]/acc_dict['det_mask_acc'][0,0,1]
                    score = acc_dict['det_mask_acc'][0,0,0]/acc_dict['det_mask_acc'][0,0,1]
                    print(f"{datasets[didx]} {idx}/{len(data_loader)} Dataset: {dataset_id:02d} Framenb: {framenb}  det: {acc_dict['det_bbox_acc'][0,0,0]/acc_dict['det_bbox_acc'][0,0,1]:.2f}  seg: {score:.2f}")
                    store_score[idx] = score

            worst_loss_ind = torch.argsort(store_loss)[-100:].flip(0)
            worst_score_ind = torch.argsort(store_score)[:50]

            print('Saving worst predctions...')

            for idx, (samples, targets) in enumerate(data_loader):
                
                if idx not in worst_loss_ind and idx not in worst_score_ind:
                    continue

                samples = samples.to(device)
                targets = [utils.nested_dict_to_device(t, device) for t in targets]

                outputs, targets, _, _, _ = model(samples,targets)
                prev_outputs = outputs['prev_outputs'] if 'prev_outputs' in outputs else None
                outputs, loss_dict = calc_loss_for_training_methods(outputs, targets, criterion)

                weight_dict = criterion.weight_dict

                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

                dataset_id = targets[0]['dataset_nb']
                framenb = targets[0]['main']['cur_target']['framenb']

                if not np.round(losses.item(),3) == np.round(store_loss[idx],3):
                    print(idx, np.round(losses.item(),3),np.round(store_loss[idx],3))

                if idx in worst_loss_ind:
                    ind = np.where(worst_loss_ind == idx)[0][0]
                    data_viz.plot_results(outputs, prev_outputs, targets,samples.tensors, args.output_dir / save_folder, folder = Path(datasets[didx] + '_outputs') / 'worst_loss', filename = f'{ind+1:03d}_Loss_{store_loss[idx]:06.2f}_dataset_{dataset_id:02d}_framenb_{framenb}.png', args=args,)
                
                if idx in worst_score_ind:
                    ind = np.where(worst_score_ind == idx)[0][0]
                    data_viz.plot_results(outputs, prev_outputs, targets,samples.tensors, args.output_dir / save_folder, folder = Path(datasets[didx] + '_outputs') / 'worst_score', filename = f'{ind+1:03d}_Score_{store_score[idx]:06.2f}_dataset_{dataset_id:02d}_framenb_{framenb}.png', args=args,)

            print(f'Done save worst predictions for {datasets[didx]} dataset')

@ex.main
def load_config(_config, _run):
    """ We use sacred only for config loading from YAML files. """
    sacred.commands.print_config(_run)


if __name__ == '__main__':
    # TODO: hierachical Namespacing for nested dict
    config = ex.run_commandline().config
    args = nested_dict_to_namespace(config)
    # args.train = Namespace(**config['train'])
    train(args)