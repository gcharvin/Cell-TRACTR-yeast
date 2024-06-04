# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import random
import time
from argparse import Namespace
from pathlib import Path
import numpy as np
import sacred
import torch
import re
import shutil 

import trackformer.util.misc as utils
from trackformer.engine import pipeline
from trackformer.models import build_model
from trackformer.util.misc import nested_dict_to_namespace

dataset = 'moma'
dataset = 'DynamicNuclearNet-tracking-v1_0'

modelpath = Path('/projectnb/dunlop/ooconnor/MOT/models/cell-trackformer/results') / dataset
datapath = Path('/projectnb/dunlop/ooconnor/MOT/data') / dataset / 'CTC' / 'test'

ex = sacred.Experiment('pipeline')
ex.add_config(modelpath.as_posix() + '/config.yaml')

def train(args: Namespace, modelpath, datapath) -> None:

    args.output_dir = Path(args.output_dir)
    print(args.output_dir)
    args.resume = modelpath / 'checkpoint.pth'

    dataset_name = args.dataset

    if not datapath.exists():
        datapath = datapath.parent / 'val'

    args.output_dir = args.output_dir / datapath.name

    args.hooks = False
    args.avg_attn_weight_maps = False

    if dataset_name != 'moma':
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
    model.train_model = False
    model.eval()
    args.eval_only = True
    criterion.eval_only = True

    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('NUM TRAINABLE MODEL PARAMS:', n_parameters)

    model_without_ddp = utils.load_model(model_without_ddp,args) # If eval_only, optimizer will not be loaded (only relevant for training)


    folderpaths = [folderpath for folderpath in sorted(datapath.iterdir()) if re.findall('\d\d$',folderpath.name)]
    folderpaths = [fp for fp in folderpaths if int(fp.stem) in [10]]

    if not args.tracking or not args.masks:
        raise NotImplementedError

    (args.output_dir.parent).mkdir(exist_ok=True)
    (args.output_dir).mkdir(exist_ok=True)
    args.output_dir = args.output_dir / 'CTC'
    (args.output_dir).mkdir(exist_ok=True)

    start_time = time.time()
    total_frames = 0

    for f,folderpath in enumerate(folderpaths):

        fps = sorted(list(folderpath.glob("*.tif")))
        total_frames += len(fps)

        Pipeline = pipeline(model, fps, args)
        Pipeline.forward()

        if f == len(folderpaths) - 1 and Pipeline.all_videos_same_size:
            Pipeline.display_enc_map(save=False,last=True)
        elif not Pipeline.all_videos_same_size:
            shutil.rmtree(args.output_dir / 'two_stage')

    total_time = time.time() - start_time
    fps = total_frames / total_time

    # with open(str(args.output_dir.parent / 'FPS.txt'), 'w') as file:
    #     file.write(f"Frames per second (FPS): {fps:2f}\n")

@ex.main
def load_config(_config, _run):
    """ We use sacred only for config loading from YAML files. """
    sacred.commands.print_config(_run)

if __name__ == '__main__':
    config = ex.run_commandline().config
    args = nested_dict_to_namespace(config)
    train(args,modelpath,datapath)
