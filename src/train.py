# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import datetime
import os
import random
import time
from pathlib import Path
import numpy as np
import sacred
import torch
import yaml
from torch.utils.data import DataLoader
import argparse, sys

import trackformer.util.misc as utils
from trackformer.datasets import build_dataset
from trackformer.engine import evaluate, train_one_epoch
from trackformer.models import build_model
from trackformer.util.misc import collate_fn as tr_collate_fn

from trackformer.util.mitosis_sampler import scan_mitosis, TripletBatchSampler, _guess_mantrack_path
from trackformer.data.triplet_seq import TripletWrapperDataset, collate_triplet_batch


DEFAULT_USE_MITOSIS_SAMPLER = False  # <-- Mets True si tu veux réactiver plus tard

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

filepath = Path(__file__)
yaml_file_paths = (filepath.parents[1] / 'cfgs').glob("*.yaml")
yaml_files = [yaml_file.stem.split('train_')[1] for yaml_file in yaml_file_paths]



ex = sacred.Experiment('train', save_git_info=False)

@ex.config
def _defaults():
    use_mitosis_sampler = DEFAULT_USE_MITOSIS_SAMPLER  # <= clé connue par Sacred

def train(respath, dataset,args) -> None:
    # La config a déjà été ajoutée dans __main__ via ex.add_config
   # config = ex.run_commandline().config
   # args = utils.nested_dict_to_namespace(config)

    Path(args.output_dir).mkdir(exist_ok=True)

    if args.dn_track or args.dn_object:
        assert args.use_dab, f'DAB-DETR is needed to use denoised boxes for tracking / object detection. args.use_dab is currently set to {args.use_dab}'

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if getattr(args, "CoMOT", False):
        assert not args.share_bbox_layers, 'Are you sure you want to share layers when using CoMOT'

    if args.num_OD_layers:
        assert args.num_OD_layers == 1

    if args.debug:
        args.num_workers = 0

    if args.output_dir:
        args.output_dir = str(args.output_dir)
        yaml.dump(
            vars(args),
            open(Path(args.output_dir) / 'config.yaml', 'w'),
            allow_unicode=True
        )

    args.output_dir = Path(args.output_dir)
    print(args)

    val_output_folder = 'val_outputs'
    train_output_folder = 'train_outputs'
    utils.create_folders(train_output_folder, val_output_folder, args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    seed = 248
    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)

    model, criterion = build_model(args)
    model.to(device)

    model_without_ddp = model

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
    
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    
    args_drop_num = args.epochs // args.lr_drop
    lr_drop = [args.lr_drop * i for i in range(1, args_drop_num + 1)] 

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_drop, gamma=0.1)

    dataset_train = build_dataset(split='train', args=args)
    dataset_val = build_dataset(split='val', args=args)
    
    use_mitosis_sampler = getattr(args, "use_mitosis_sampler", DEFAULT_USE_MITOSIS_SAMPLER)
    
    print(f"[CFG] use_mitosis_sampler = {use_mitosis_sampler}")


    if use_mitosis_sampler:
        # ====== CHEMIN ACTUEL (custom sampler) ======
        if not hasattr(dataset_train, "split_name"): setattr(dataset_train, "split_name", "train")
        if not hasattr(dataset_val,   "split_name"): setattr(dataset_val,   "split_name", "val")

        coco_root = Path(args.data_dir) / "moma" / "COCO"
        if not coco_root.exists():
            cand = Path(args.output_dir).parents[1] / "trainingdataset" / "COCO"
            if cand.exists(): coco_root = cand
            else: raise FileNotFoundError(f"[COCO] introuvable: {coco_root} (fallback: {cand})")
        print(f"[COCO] root = {coco_root}")

        # scan + samplers comme avant...
        pos_idx, neg_idx, seq2frames = scan_mitosis(dataset_train, coco_root=coco_root, split_dir="train")

        centers_per_batch = args.batch_size // 3
        pos_per_batch = max(1, int(0.5 * centers_per_batch))

        batch_sampler = TripletBatchSampler(
            dataset=dataset_train,
            pos_centers=pos_idx,
            neg_centers=neg_idx,
            batch_size=args.batch_size,     # multiple de 3 requis
            pos_per_batch=pos_per_batch,
            seq2frames=seq2frames,
            shuffle=True,
            seed=42,
            require_triplets=True
        )

        data_loader_train = DataLoader(
            dataset_train,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=tr_collate_fn,
        )

        pos_va, neg_va, seq2frames_val = scan_mitosis(dataset_val, coco_root=coco_root, split_dir="val")
        sampler_val = TripletBatchSampler(
            dataset=dataset_val,
            pos_centers=pos_va,
            neg_centers=neg_va,
            batch_size=3,  # 1 triplet
            pos_per_batch=1,
            seq2frames=seq2frames_val,
            shuffle=False,
            seed=0,
            require_triplets=True
        )
        data_loader_val = DataLoader(
            dataset_val,
            batch_sampler=sampler_val,
            collate_fn=tr_collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    else:
        # ====== SEQUENTIAL TRIPLETS (no mitosis balancing) ======
        # Wrap base datasets so each item yields a (prev, cur, next) triplet.
        train_wrapper = TripletWrapperDataset(
            dataset_train,
            seq_len=3,
            pad_mode="edge",      # duplicate edge frames at boundaries
            jitter=0,             # no temporal jitter
            center_mode="middle", # (t-1, t, t+1) around the index
        )
        val_wrapper = TripletWrapperDataset(
            dataset_val,
            seq_len=3,
            pad_mode="edge",
            jitter=0,
            center_mode="middle",
        )

        data_loader_train = DataLoader(
            train_wrapper,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_triplet_batch,   # <<< important
            worker_init_fn=seed_worker,
            persistent_workers=args.num_workers > 0,
        )
        data_loader_val = DataLoader(
            val_wrapper,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_triplet_batch,   # <<< important
            worker_init_fn=seed_worker,
            persistent_workers=args.num_workers > 0,
        )


   
    print("[DEBUG] train collate_fn =", data_loader_train.collate_fn.__name__)
    print("[DEBUG] val   collate_fn =", data_loader_val.collate_fn.__name__)

    # batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
        
    # data_loader_train = DataLoader(
    #     dataset_train,
    #     batch_sampler=batch_sampler_train,
    #     collate_fn=utils.collate_fn,
    #     num_workers=args.num_workers,
    #     worker_init_fn=seed_worker)

    # data_loader_val = DataLoader(
    #     dataset_val, args.batch_size,
    #     sampler=sampler_val,
    #     drop_last=False,
    #     collate_fn=utils.collate_fn,
    #     num_workers=args.num_workers,
    #     worker_init_fn=seed_worker)

    if args.resume:
        model_without_ddp = utils.load_model(model_without_ddp, args, param_dicts, optimizer, lr_scheduler)

    if args.eval_only:
        raise NotImplementedError
        evaluate(model, criterion, data_loader_val, device, args.output_dir, args, 0)
        return
    
    assert args.start_epoch < args.epochs + 1
    
    print("Start training")
    model.train_model = True  # detr_tracking script will process multiple sequential frames

    for epoch in range(args.start_epoch, args.epochs + 1):

        start_epoch = time.time()  # Measure time for each epoch
        
        # set epoch for reproducibility
        dataset_train.set_epoch(epoch)
        dataset_val.set_epoch(epoch) 

        # TRAIN
        train_metrics = train_one_epoch(model, criterion, data_loader_train, optimizer, epoch, args)

        lr_scheduler.step()

        # VAL
        val_metrics = evaluate(model, criterion, data_loader_val, args, epoch)

        # Save loss and metrics in a pickle file
        utils.save_metrics_pkl(train_metrics, args.output_dir, 'train', epoch=epoch)  
        utils.save_metrics_pkl(val_metrics, args.output_dir, 'val', epoch=epoch)  

        # plot loss
        utils.plot_loss_and_metrics(args.output_dir)

        checkpoint_paths = [args.output_dir / 'checkpoint.pth']

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

        # Display and save time epoch took
        total_epoch_time = time.time() - start_epoch
        print(f'Epoch took {str(datetime.timedelta(seconds=int(total_epoch_time)))}')
        with open(str(args.output_dir / "training_time.txt"), "a") as f:
            f.write(f"Epoch {epoch}: {str(datetime.timedelta(seconds=int(total_epoch_time)))}\n")

    # Display and save total time training took
    total_time = utils.get_total_time(args)
    print('Training time {}'.format(total_time))


@ex.main
def load_config(_config, _run):
    """ We use sacred only for config loading from YAML files. """
    sacred.commands.print_config(_run)


if __name__ == '__main__':
        # 1) Parser uniquement notre arg custom et retirer le reste
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataset', type=str)
    cli_args, remaining = parser.parse_known_args()

    # 2) Choisir le YAML
    dataset = cli_args.dataset if cli_args.dataset else yaml_files[0]
    yaml_path = filepath.parents[1] / 'cfgs' / f'train_{dataset}.yaml'
    print("[CFG] YAML chargé :", yaml_path)
    ex.add_config(str(yaml_path))

    # 3) IMPORTANT : enlever --dataset des args avant Sacred
    sys.argv = [sys.argv[0]] + remaining

    # 4) Laisser Sacred parser les overrides "with ..."
    run = ex.run_commandline()
    config = run.config
    args = utils.nested_dict_to_namespace(config)

    respath = filepath.parents[1] / 'results' / dataset
    respath.mkdir(exist_ok=True)
    train(respath, dataset, args)           # <— on passe args i
    
    
    # import argparse
    # parser = argparse.ArgumentParser(add_help=False)
    # parser.add_argument("--dataset", type=str, help="suffixe après train_, ex: moma -> train_moma.yaml")
    # cli_args, _ = parser.parse_known_args()

    # if cli_args.dataset:
    #     dataset = cli_args.dataset
    # else:
    #     dataset = yaml_files[0]

    # yaml_path = filepath.parents[1] / 'cfgs' / f"train_{dataset}.yaml"
    # print("[CFG] YAML chargé :", yaml_path)  # pour vérifier
    # ex.add_config(str(yaml_path))

    # args = ex.run_commandline().config
    # respath = filepath.parents[1] / 'results' / dataset
    # respath.mkdir(exist_ok=True)
    # train(respath, dataset)
