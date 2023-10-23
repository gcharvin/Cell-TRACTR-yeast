# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import datetime
import os
import random
import time
from argparse import Namespace
from pathlib import Path
from datetime import date
import shutil

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import numpy as np
import sacred
import torch
import yaml
from torch.utils.data import DataLoader, DistributedSampler

import trackformer.util.misc as utils
from trackformer.datasets import build_dataset
from trackformer.engine import evaluate, train_one_epoch
from trackformer.models import build_model

dataset = 'moma' #['moma','2D','DIC-C2DH-HeLa','Fluo-N2DH-SIM+']
# dataset = '2D' #['moma','2D','DIC-C2DH-HeLa','Fluo-N2DH-SIM+']

ex = sacred.Experiment('train')
ex.add_config('/projectnb/dunlop/ooconnor/object_detection/cell-trackformer/cfgs/train_' + dataset + '.yaml')

def train(args: Namespace) -> None:

    if args.resume:
        args.output_dir = Path(args.resume).parent
    else:
        args.output_dir = Path(args.output_dir) / (f'{date.today().strftime("%y%m%d")}_{dataset}{"" if args.flex_div else "_no"}_flex_div{"_CoMOT" if args.CoMOT else ""}{"_object_detection_only" if args.object_detection_only else "_track"}{"_two_stage" if args.two_stage else ""}{"_dn_enc" if args.dn_enc and args.two_stage else ""}{"_dn_track" if args.dn_track else ""}{"_dn_track_group" if args.dn_track_group and args.dn_track else ""}{"_dn_object" if args.dn_object else ""}{"_dab" if args.use_dab else ""}{"_intermediate" if args.masks and args.return_intermediate_masks else ""}{"_mask" if args.masks else "_no_mask"}_{args.enc_layers}_enc_{args.dec_layers}_dec_layers')
        args.output_dir.mkdir(exist_ok=True)
    
    if args.dn_track or args.dn_object:
        assert args.use_dab, f'DAB-DETR is needed to use denoised boxes for tracking / object detection. args.use_dab is currently set to {args.use_dab}'

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.CoMOT:
        assert not args.share_bbox_layers, 'Are you sure you want to share layers when using CoMOT'

    if args.debug:
        # args.tracking_eval = False
        args.num_workers = 0

    if not args.deformable:
        assert args.num_feature_levels == 1

    if args.output_dir:
        args.output_dir = str(args.output_dir)
        yaml.dump(
            vars(args),
            open(Path(args.output_dir) / 'config.yaml', 'w'), allow_unicode=True)

    args.output_dir = Path(args.output_dir)

    print(args)

    val_output_folder = 'val_outputs'
    train_output_folder = 'train_outputs'

    args.output_dir.mkdir(exist_ok=True)
    (args.output_dir / val_output_folder).mkdir(exist_ok=True)
    (args.output_dir / train_output_folder).mkdir(exist_ok=True)

    (args.output_dir / val_output_folder / 'standard').mkdir(exist_ok=True)
    (args.output_dir / train_output_folder / 'standard').mkdir(exist_ok=True)

    if args.two_stage:
        (args.output_dir / val_output_folder / 'enc_outputs').mkdir(exist_ok=True)
        (args.output_dir / train_output_folder / 'enc_outputs').mkdir(exist_ok=True)

        if args.dn_enc:
            (args.output_dir / val_output_folder / 'dn_enc').mkdir(exist_ok=True)
            (args.output_dir / train_output_folder / 'dn_enc').mkdir(exist_ok=True) 

    if args.dn_track and not args.object_detection_only:
        (args.output_dir / val_output_folder / 'dn_track').mkdir(exist_ok=True)
        (args.output_dir / train_output_folder / 'dn_track').mkdir(exist_ok=True)     

    if args.dn_track and args.dn_track_group and not args.object_detection_only:
        (args.output_dir / val_output_folder / 'dn_track_group').mkdir(exist_ok=True)
        (args.output_dir / train_output_folder / 'dn_track_group').mkdir(exist_ok=True)  

    if args.dn_object:
        (args.output_dir / val_output_folder / 'dn_object').mkdir(exist_ok=True)
        (args.output_dir / train_output_folder / 'dn_object').mkdir(exist_ok=True)   

    if args.CoMOT:
        (args.output_dir / val_output_folder / 'CoMOT').mkdir(exist_ok=True)
        (args.output_dir / train_output_folder / 'CoMOT').mkdir(exist_ok=True)  

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

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('NUM TRAINABLE MODEL PARAMS:', n_parameters)

    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters()
                    if not match_name_keywords(n, args.lr_backbone_names + args.lr_linear_proj_names + ['layers_track_attention']) and p.requires_grad],
         "lr": args.lr,},
        {"params": [p for n, p in model_without_ddp.named_parameters()
                    if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
         "lr": args.lr_backbone},
        {"params": [p for n, p in model_without_ddp.named_parameters()
                    if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
         "lr":  args.lr * args.lr_linear_proj_mult}]
    if args.track_attention:
        param_dicts.append({
            "params": [p for n, p in model_without_ddp.named_parameters()
                       if match_name_keywords(n, ['layers_track_attention']) and p.requires_grad],
            "lr": args.lr_track})

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    
    args_drop_num = args.epochs // args.lr_drop
    lr_drop = [args.lr_drop * i for i in range(1,args_drop_num+1)] 

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_drop, gamma=0.1)

    dataset_train = build_dataset(split='train', args=args)
    dataset_val = build_dataset(split='val', args=args)

    if args.distributed:
        sampler_train = utils.DistributedWeightedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers)

    data_loader_val = DataLoader(
        dataset_val, args.batch_size,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers)

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        model_state_dict = model_without_ddp.state_dict()
        checkpoint_state_dict = checkpoint['model']
        checkpoint_state_dict = {
            k.replace('detr.', ''): v for k, v in checkpoint['model'].items()}

        for k, v in checkpoint_state_dict.items():
            if k not in model_state_dict:
                print(f'Where is {k} {tuple(v.shape)}?')

        resume_state_dict = {}
        for k, v in model_state_dict.items():
            if k not in checkpoint_state_dict:
                resume_value = v
                print(f'Load {k} {tuple(v.shape)} from scratch.')
            elif v.shape != checkpoint_state_dict[k].shape:
                checkpoint_value = checkpoint_state_dict[k]
                num_dims = len(checkpoint_value.shape)

                if 'norm' in k:
                    resume_value = checkpoint_value.repeat(2)
                elif 'multihead_attn' in k or 'self_attn' in k:
                    resume_value = checkpoint_value.repeat(num_dims * (2, ))
                elif 'reference_points' in k and checkpoint_value.shape[0] * 2 == v.shape[0]:
                    resume_value = v
                    resume_value[:2] = checkpoint_value.clone()
                elif 'linear1' in k or 'query_embed' in k:
                    resume_state_dict[k] = v
                    print(f'Load {k} {tuple(v.shape)} from scratch.')
                    continue
                elif 'linear2' in k or 'input_proj' in k:
                    resume_value = checkpoint_value.repeat((2,) + (num_dims - 1) * (1, ))
                elif 'class_embed' in k:
                    resume_value = checkpoint_value[list(range(0, 20))]
                else:
                    raise NotImplementedError(f"No rule for {k} with shape {v.shape}.")

                print(f"Load {k} {tuple(v.shape)} from resume model "
                      f"{tuple(checkpoint_value.shape)}.")
            elif args.resume_shift_neuron and 'class_embed' in k:
                checkpoint_value = checkpoint_state_dict[k]
                resume_value = checkpoint_value.clone()
                resume_value[:-1] = checkpoint_value[1:].clone()
                resume_value[-2] = checkpoint_value[0].clone()
                print(f"Load {k} {tuple(v.shape)} from resume model and "
                      "shift class embed neurons to start with label=0 at neuron=0.")
            else:
                resume_value = checkpoint_state_dict[k]

            resume_state_dict[k] = resume_value

        if args.masks and args.load_mask_head_from_model is not None:
            checkpoint_mask_head = torch.load(
                args.load_mask_head_from_model, map_location='cpu')

            for k, v in resume_state_dict.items():

                if (('bbox_attention' in k or 'mask_head' in k)
                    and v.shape == checkpoint_mask_head['model'][k].shape):
                    print(f'Load {k} {tuple(v.shape)} from mask head model.')
                    resume_state_dict[k] = checkpoint_mask_head['model'][k]

        model_without_ddp.load_state_dict(resume_state_dict)

        # RESUME OPTIM
        if not args.eval_only and args.resume_optim:
            if 'optimizer' in checkpoint:
                if args.overwrite_lrs:
                    for c_p, p in zip(checkpoint['optimizer']['param_groups'], param_dicts):
                        c_p['lr'] = p['lr']

                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'lr_scheduler' in checkpoint:
                if args.overwrite_lr_scheduler:
                    checkpoint['lr_scheduler'].pop('milestones')
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                if args.overwrite_lr_scheduler:
                    lr_scheduler.step(checkpoint['lr_scheduler']['last_epoch'])
            if 'epoch' in checkpoint:
                args.start_epoch = checkpoint['epoch'] + 1
                print(f"RESUME EPOCH: {args.start_epoch}")

            for file_name in ['metrics_train.pkl','metrics_val.pkl','training_time.txt']:
                if not (args.output_dir / file_name).exists():
                    shutil.copyfile(Path(args.resume).parent / file_name, args.output_dir / file_name)


    if args.eval_only:
        evaluate(
            model, criterion, data_loader_val, device,
            args.output_dir, args, 0)
        return
    
    val_loss = np.inf

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs + 1):
        start_epoch = time.time()
        # TRAIN
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_metrics = train_one_epoch(model, criterion, data_loader_train, optimizer, epoch, args)

        lr_scheduler.step()

        checkpoint_paths = [args.output_dir / 'checkpoint.pth']

        val_metrics = evaluate(model, criterion, data_loader_val,args, epoch)

        utils.save_metrics_pkl(train_metrics,args.output_dir,'train',epoch=epoch)  
        utils.save_metrics_pkl(val_metrics,args.output_dir,'val',epoch=epoch)  

        # MODEL SAVING
        new_val_loss = val_metrics['loss'][-1].mean()

        if new_val_loss < val_loss:
            val_loss = new_val_loss
            if args.output_dir:
                if args.save_model_interval and not epoch % args.save_model_interval:
                    checkpoint_paths.append(str(args.output_dir / f"checkpoint_epoch_{epoch}.pth"))

                for checkpoint_path in checkpoint_paths:
                    args.output_dir = str(args.output_dir)
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, str(checkpoint_path))
                    args.output_dir = Path(args.output_dir)

        total_epoch_time = time.time() - start_epoch
        print(f'Epoch took {str(datetime.timedelta(seconds=int(total_epoch_time)))}')
        with open(str(args.output_dir / "training_time.txt"), "a") as f:
            f.write(f"Epoch {epoch}: {str(datetime.timedelta(seconds=int(total_epoch_time)))}\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    with open(str(args.output_dir / "training_time.txt"), "a") as f:
        f.write(f"Total time: {total_time_str}\n")

@ex.main
def load_config(_config, _run):
    """ We use sacred only for config loading from YAML files. """
    sacred.commands.print_config(_run)


if __name__ == '__main__':
    config = ex.run_commandline().config
    args = utils.nested_dict_to_namespace(config)
    train(args)
