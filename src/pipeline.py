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

filepath = Path(__file__)

ex = sacred.Experiment('pipeline')

def train(args: Namespace, path_to_dataset) -> None:

    print(args.output_dir)
    args.resume = args.output_dir / 'checkpoint.pth'
    args.output_dir = args.output_dir / path_to_dataset.name

    if (path_to_dataset / 'CTC' / 'test').exists():
        path_to_dataset = path_to_dataset / 'CTC' / 'test'
    elif (path_to_dataset / 'CTC' / 'val').exists():
        path_to_dataset = path_to_dataset / 'CTC' / 'train'
    else:
        raise NotImplementedError

    args.hooks = False
    args.avg_attn_weight_maps = False
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

    folderpaths = [folderpath for folderpath in sorted(path_to_dataset.iterdir()) if re.findall('\d\d$',folderpath.name)]

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

    with open(str(args.output_dir.parent / 'FPS.txt'), 'w') as file:
        file.write(f"Frames per second (FPS): {fps:2f}\n")

@ex.config
def my_config():
    path_to_dataset = '/path/to/CTC-dataset' 
    modelname = 'pretrain-OD-freeze-backbone-train-tracking-segmentation-moma'

@ex.main
def load_config(_config, _run):
    """ We use sacred only for config loading from YAML files. """
    sacred.commands.print_config(_run)

if __name__ == '__main__':
    args = ex.run_commandline().config
    path_to_dataset = Path(args['path_to_dataset'])
    modelname = args['modelname']


    path_to_dataset = Path('/projectnb/dunlop/ooconnor/MOT/data/CTC_datasets/moma')
    respath = filepath.parents[1] / 'results' / modelname

    ex.add_config(str(respath / 'config.yaml'))
    args = ex.run_commandline().config
    args = utils.nested_dict_to_namespace(args)

    args.output_dir = Path(args.output_dir)

    train(args,path_to_dataset)
