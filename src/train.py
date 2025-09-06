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

import trackformer.util.misc as utils
from trackformer.datasets import build_dataset
from trackformer.engine import evaluate, train_one_epoch
from trackformer.models import build_model
from trackformer.util.misc import collate_fn as tr_collate_fn

from trackformer.util.mitosis_sampler import scan_mitosis, TripletBatchSampler, _guess_mantrack_path

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

filepath = Path(__file__)
yaml_file_paths = (filepath.parents[1] / 'cfgs').glob("*.yaml")
yaml_files = [yaml_file.stem.split('train_')[1] for yaml_file in yaml_file_paths]


ex = sacred.Experiment('train', save_git_info=False)

def train(respath, dataset) -> None:
    # La config a déjà été ajoutée dans __main__ via ex.add_config
    config = ex.run_commandline().config
    args = utils.nested_dict_to_namespace(config)

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

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

   
  # règle simple : ~50% POS si dispo, sinon 0 (fallback auto)
    # batch_sampler = BalancedMitosisBatchSampler.from_dataset(
    #     dataset_train,
    #     batch_size=args.batch_size,
    #     pos_per_batch=None,    # None -> heuristique (moitié si pos>0)
    #     nscan=None,            # ou limite ex. 10000 si dataset énorme
    #     shuffle=True,
    #     seed=42,
    #     drop_last=True,
    # )

    # data_loader_train = DataLoader(
    #     dataset_train,
    #     batch_sampler=batch_sampler,   # (défini ci-dessous)
    #     num_workers=args.num_workers,
    #     pin_memory=True,
    #     collate_fn=tr_collate_fn,
    # )

# S'assure que le dataset expose un split_name (utile pour lire man_track/train|val)
    if not hasattr(dataset_train, "split_name"):
        setattr(dataset_train, "split_name", "train")
    if not hasattr(dataset_val, "split_name"):
        setattr(dataset_val, "split_name", "val")

    # === COCO root depuis le YAML ===
    coco_root = Path(args.data_dir) / "moma" / "COCO" 
    
    if not coco_root.exists():
        # Fallback simple si jamais data_dir pointe ailleurs
        cand = Path(args.output_dir).parents[1] / "trainingdataset" / "COCO"
        if cand.exists():
            coco_root = cand
        else:
            raise FileNotFoundError(f"[COCO] introuvable: {coco_root} (fallback: {cand})")

    print(f"[COCO] root = {coco_root}")

   # 1) Scanner les centres POS/NEG une fois
    pos_idx, neg_idx, seq2frames = scan_mitosis(dataset_train, coco_root=coco_root, split_dir="train")

    # DEBUG : vérifie une séquence
    if hasattr(dataset_train, "coco_root"):
        print("[COCO] root =", dataset_train.coco_root)
    # Affiche les 3 premières clés de seq détectées
    from itertools import islice
    print("[MITOSIS][DEBUG] nb seq =", len(seq2frames))
    print("[MITOSIS][DEBUG] sample seqs =", list(islice(sorted(seq2frames.keys()), 5)))
    if len(seq2frames):
        s = next(iter(seq2frames.keys()))
        print("[MITOSIS][DEBUG] man_track path =", _guess_mantrack_path(coco_root, "train", s))


    # 2) Définir un ratio par *centres* (ex. 50% de centres POS par batch)
    centers_per_batch = args.batch_size // 3
    
    pos_per_batch = max(1, int(0.5 * centers_per_batch))

    # 3) Sampler qui émet des indices [t-1, t, t+1,  t-1, t, t+1, ...]
    batch_sampler = TripletBatchSampler(
        dataset=dataset_train,
        pos_centers=pos_idx,
        neg_centers=neg_idx,
        batch_size=args.batch_size,         # doit être multiple de 3
        pos_per_batch=pos_per_batch,        # sur les *centres*
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
        collate_fn=tr_collate_fn
    )

    pos_va, neg_va, seq2frames_val = scan_mitosis(dataset_val, coco_root=coco_root, split_dir="val")

    # batch de 1 triplet (t-1, t, t+1) en validation
    sampler_val = TripletBatchSampler(
        dataset=dataset_val,
        pos_centers=pos_va,
        neg_centers=neg_va,
        batch_size=3,           # 1 “centre” => 3 images
        pos_per_batch=1,        # au moins 1 centre POS si dispo
        seq2frames=seq2frames_val,
        shuffle=False,
        seed=0,
        require_triplets=True
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_sampler=sampler_val,     # <== PAS de batch_size ici
        collate_fn=tr_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

        
    # OPTION A — le plus simple : pas de batch_sampler, juste batch_size + shuffle
    # data_loader_train = DataLoader(
    #     dataset_train,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=args.num_workers,
    #     pin_memory=True,
    #     collate_fn=tr_collate_fn
    # )
    
    # data_loader_val= DataLoader(
    #     dataset_val,
    #     num_workers=args.num_workers,
    #     pin_memory=True,
    #     collate_fn=tr_collate_fn
    # )


  
    
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
    # Parse the dataset from the command line
    dataset = yaml_files[0]  # ou override via --with dataset=...
    yaml_path = filepath.parents[1] / 'cfgs' / ('train_' + dataset + '.yaml')
    ex.add_config(str(yaml_path))

    args = ex.run_commandline().config
    respath = filepath.parents[1] / 'results' / dataset
    respath.mkdir(exist_ok=True)
    train(respath, dataset)
