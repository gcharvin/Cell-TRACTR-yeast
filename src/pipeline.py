# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import datetime
import os
import random
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import sacred
import torch
import yaml

import trackformer.util.misc as utils
from trackformer.engine import run_pipeline
from trackformer.models import build_model
from trackformer.util.misc import nested_dict_to_namespace


ex = sacred.Experiment('train')
# ex.add_config('../cfgs/train.yaml')
ex.add_config('/projectnb/dunlop/ooconnor/object_detection/cell-trackformer/cfgs/train.yaml')
ex.add_named_config('deformable', '/projectnb/dunlop/ooconnor/object_detection/cell-trackformer/cfgs/train_deformable.yaml')

def train(args: Namespace) -> None:

    modelname = '221208_dn_track_dab_no_mask'

    args.dn_track = False
    args.dn_object = False
    args.group_object = False
   
    args.output_dir = Path(args.output_dir) / modelname
    print(args.output_dir)
    args.save_model_interval = False
    args.eval_only = True
    args.resume = Path('/projectnb/dunlop/ooconnor/object_detection/cell-trackformer/results') / modelname / 'checkpoint.pth'
    
    datapath = Path('/projectnb/dunlop/ooconnor/object_detection/cell-trackformer/data/cells/predictions/2022-04-24_TrainingSet8/img')
    fps = sorted(list((datapath).glob('*.png')))

    print(args)

    args.output_dir.mkdir(exist_ok=True)
    (args.output_dir / 'eval_outputs').mkdir(exist_ok=True)
    (args.output_dir / 'train_outputs').mkdir(exist_ok=True)

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

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

    if str(args.resume).startswith('https'):
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

    run_pipeline(model, fps, device, output_dir, args)



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
