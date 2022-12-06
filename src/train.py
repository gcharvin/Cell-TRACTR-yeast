# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import datetime
import os
import random
import time
from argparse import Namespace
from pathlib import Path
from datetime import date

import numpy as np
import sacred
import torch
import yaml
from torch.utils.data import DataLoader, DistributedSampler

import trackformer.util.misc as utils
from trackformer.datasets import build_dataset
from trackformer.engine import evaluate, train_one_epoch
from trackformer.models import build_model


ex = sacred.Experiment('train')
# ex.add_config('../cfgs/train.yaml')
# ex.add_config(str(Path('../cfgs/train.yaml').resolve()))
ex.add_config('/projectnb/dunlop/ooconnor/object_detection/cell-trackformer/cfgs/train.yaml')
ex.add_named_config('deformable', '/projectnb/dunlop/ooconnor/object_detection/cell-trackformer/cfgs/train_deformable.yaml')

#### TODO 
# Need to properly update data augmentations so that the multiple frames in a row will get properly flipped
# Also only certain data augmentations for the 16 bit gt

def train(args: Namespace) -> None:

    
    args.output_dir = Path(args.output_dir) / (f'{date.today().strftime("%y%m%d")}_{"group" if args.group_object else "no_group"}_{"dab" if args.use_dab else "no_dab"}_{"mask" if args.masks else "no_mask"}')
    # args.resume = ('/projectnb/dunlop/ooconnor/object_detection/cell-trackformer/results/20221020_mask_weight_target_cell_100_other_cells_20_dataaug_not_hflip_freeze_detr_2/checkpoint.pth')
    args.save_model_interval = False
    # args.resume_optim = False
    # args.freeze_detr = True
    # args.overwrite_lrs = True

    print(args)

    args.output_dir.mkdir(exist_ok=True)
    (args.output_dir / 'eval_outputs').mkdir(exist_ok=True)
    (args.output_dir / 'train_outputs').mkdir(exist_ok=True)

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.debug:
        # args.tracking_eval = False
        args.num_workers = 0

    if not args.deformable:
        assert args.num_feature_levels == 1
        
    if args.tracking:
        if args.tracking_eval:
            assert 'mot' in args.dataset or 'cells' in args.dataset

    output_dir = Path(args.output_dir)
    if args.output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        yaml.dump(
            vars(args),
            open(output_dir / 'config.yaml', 'w'), allow_unicode=True)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()

    os.environ['PYTHONHASHSEED'] = str(seed)
    # os.environ['NCCL_DEBUG'] = 'INFO'
    # os.environ["NCCL_TREE_THRESHOLD"] = "0"

    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    model, criterion, postprocessors = build_model(args)
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

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_drop, gamma=0.5)

    dataset_train = build_dataset(split='train', args=args)
    dataset_val = build_dataset(split='val', args=args)

    if args.distributed:
        sampler_train = utils.DistributedWeightedSampler(dataset_train)
        # sampler_train = DistributedSampler(dataset_train)
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

    best_val_stats = None
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
                #     if checkpoint_value.shape[1] * 2 == v.shape[1]:
                #         # from hidden size 256 to 512
                #         resume_value = checkpoint_value.repeat(1, 2)
                #     elif checkpoint_value.shape[0] * 5 == v.shape[0]:
                #         # from 100 to 500 object queries
                #         resume_value = checkpoint_value.repeat(5, 1)
                #     elif checkpoint_value.shape[0] > v.shape[0]:
                #         resume_value = checkpoint_value[:v.shape[0]]
                #     elif checkpoint_value.shape[0] < v.shape[0]:
                #         resume_value = v
                #     else:
                #         raise NotImplementedError
                elif 'linear2' in k or 'input_proj' in k:
                    resume_value = checkpoint_value.repeat((2,) + (num_dims - 1) * (1, ))
                elif 'class_embed' in k:
                    # person and no-object class
                    # resume_value = checkpoint_value[[1, -1]]
                    # resume_value = checkpoint_value[[0, -1]]
                    # resume_value = checkpoint_value[[1,]]
                    resume_value = checkpoint_value[list(range(0, 20))]
                    # resume_value = v
                    # print(f'Load {k} {tuple(v.shape)} from scratch.')
                else:
                    raise NotImplementedError(f"No rule for {k} with shape {v.shape}.")

                print(f"Load {k} {tuple(v.shape)} from resume model "
                      f"{tuple(checkpoint_value.shape)}.")
            elif args.resume_shift_neuron and 'class_embed' in k:
                checkpoint_value = checkpoint_state_dict[k]
                # no-object class
                resume_value = checkpoint_value.clone()
                # no-object class
                # resume_value[:-2] = checkpoint_value[1:-1].clone()
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

            best_val_stats = checkpoint['best_val_stats']

    if args.eval_only:
        evaluate(
            model, criterion, postprocessors, data_loader_val, device,
            output_dir, args, 0)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs + 1):
        num_plots = 10 if epoch < args.epochs else 30
        # TRAIN
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_one_epoch(
            model, criterion, postprocessors, data_loader_train, optimizer, device, epoch, args, num_plots)

        if args.eval_train:
            random_transforms = data_loader_train.dataset._transforms
            data_loader_train.dataset._transforms = data_loader_val.dataset._transforms
            evaluate(
                model, criterion, postprocessors, data_loader_train, device,
                output_dir, args, epoch, train=True)
            data_loader_train.dataset._transforms = random_transforms

        lr_scheduler.step()

        checkpoint_paths = [output_dir / 'checkpoint.pth']

        evaluate(
            model, criterion, data_loader_val, device,
            output_dir, args, epoch, train=True)

        # MODEL SAVING
        if args.output_dir:
            if args.save_model_interval and not epoch % args.save_model_interval:
                checkpoint_paths.append(output_dir / f"checkpoint_epoch_{epoch}.pth")

            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


@ex.main
def load_config(_config, _run):
    """ We use sacred only for config loading from YAML files. """
    sacred.commands.print_config(_run)


if __name__ == '__main__':
    config = ex.run_commandline().config
    args = utils.nested_dict_to_namespace(config)
    train(args)
