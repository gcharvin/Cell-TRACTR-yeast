# Copyright (c) Facebook, Inc.
# All Rights Reserved

import os
import re
import sys
import time
import random
import shutil
import argparse
from argparse import Namespace
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import sacred
import sacred.commands as sacred_commands
import torch

import trackformer.util.misc as utils
from trackformer.engine import pipeline
from trackformer.models import build_model


# =========================
# CLI (résumé ; le reste via Sacred)
# =========================
parser = argparse.ArgumentParser()
parser.add_argument(
    "--results_path",
    type=str,
    default=None,
    help="Dossier PARENT des runs (ex: .../results). Le run est choisi via 'with res_name=...'"
)
parser.add_argument(
    "--only",
    type=str,
    default=None,
    help="Ne traiter qu'une séquence (ex: 'seq17393940381' ou '00')"
)
args_cmd, unknown = parser.parse_known_args()

# Laisser Sacred consommer ses propres args (les 'with ...')
sys.argv = [sys.argv[0]] + unknown

filepath = Path(__file__)
ex = sacred.Experiment('pipeline')


# =========================
# Utilitaires
# =========================
def _pick_checkpoint(respath: Path) -> Path:
    """Prend checkpoint_best.pth si présent, sinon checkpoint.pth, sinon erreur claire."""
    ckpt_best = respath / "checkpoint_best.pth"
    ckpt_last = respath / "checkpoint.pth"
    if ckpt_best.exists():
        return ckpt_best
    if ckpt_last.exists():
        return ckpt_last
    raise FileNotFoundError(
        f"[CKPT] Aucun checkpoint trouvé dans {respath}\n"
        f"  - attendu: {ckpt_best} ou {ckpt_last}"
    )


def _list_sequences(datapath: Path, only: Optional[str]) -> List[Path]:
    """Liste les dossiers séquences (regex 'seq\\d+' ou '^[0-9]{2}$')."""
    if not datapath.exists():
        raise FileNotFoundError(
            f"[DATA] {datapath} n'existe pas. Attendu: <data_dir>/CTC/<dataset>/"
        )
    cand = sorted([p for p in datapath.iterdir() if p.is_dir()])
    folderpaths = [p for p in cand if re.search(r'^seq\d+$', p.name)]
    if not folderpaths:
        folderpaths = [p for p in cand if re.search(r'^\d{2}$', p.name)]

    if only is not None:
        folderpaths = [p for p in folderpaths if p.name == only]
        if not folderpaths:
            raise FileNotFoundError(f"[ERROR] Aucun dossier '{only}' sous {datapath}")

    return folderpaths


def _parse_with_overrides(argv: List[str]) -> Dict[str, str]:
    """
    Extrait les paires key=value passées après le mot-clé 'with' (syntaxe Sacred).
    Exemple argv: ['.../pipeline.py', 'with', 'res_name=foo', 'dataset=bar']
    """
    d: Dict[str, str] = {}
    if "with" not in argv:
        return d
    i = argv.index("with") + 1
    while i < len(argv):
        tok = argv[i]
        if tok == "with":  # peu probable, mais on sort si on retombe dessus
            break
        if "=" in tok:
            k, v = tok.split("=", 1)
            d[k.strip()] = v.strip()
        i += 1
    return d


def _coerce_path(x) -> Path:
    return x if isinstance(x, Path) else Path(x)


def _fix_paths_for_inference(args: Namespace, respath: Path) -> Namespace:
    """
    - args.resume : checkpoint du RUN (results/<res_name>/checkpoint*.pth)
    - args.output_dir : base des sorties d’inférence = results/<dataset>/test
    - args.data_dir : doit contenir CTC/<dataset>
    - Désactive crop/shift pour éviter le décalage des masques en inférence.
    """
    # Checkpoint du RUN
    ckpt_path = _pick_checkpoint(respath)
    args.resume = str(ckpt_path)

    # Sûreté inférence (pas d'aug de training)
    args.crop = False
    args.shift = False

    # Sorties d'inférence = results/<dataset>/test (le code créera /CTC/<seq>)
    results_root = respath.parent  # .../results
    out_base = results_root / args.dataset / "test"
    args.output_dir = _coerce_path(out_base)  # Path (engine.py s'attend à Path)

    # --- Ajustement data_dir si besoin ---
    # On attend : <data_dir>/CTC/<dataset>/<seq>/img_***.tif
    data_dir = _coerce_path(args.data_dir)
    expected_ctc = data_dir / "CTC" / args.dataset
    if not expected_ctc.exists():
        # Cas fréquent : config de training -> ".../trainingdataset"
        if data_dir.name.lower() == "trainingdataset":
            data_dir = data_dir.parent
            expected_ctc = data_dir / "CTC" / args.dataset
            print(f"[DATA-PATH] Ajustement data_dir -> {data_dir}")
        else:
            # Dernière tentative : si le chemin contient ".../trainingdataset/...",
            # on coupe à ce niveau pour remonter à la racine 'classification/<dataset>'
            parts = [p.lower() for p in data_dir.parts]
            if "trainingdataset" in parts:
                idx = parts.index("trainingdataset")
                data_dir = Path(*data_dir.parts[:idx])
                expected_ctc = data_dir / "CTC" / args.dataset
                print(f"[DATA-PATH] Ajustement data_dir -> {data_dir}")

    # Vérification finale
    if not expected_ctc.exists():
        raise FileNotFoundError(
            "[DATA] data_dir incohérent pour l'inférence.\n"
            f"  - data_dir actuel : {data_dir}\n"
            f"  - attendu         : {expected_ctc} (doit contenir les images exportées en TIFF)\n"
            "Régle 'data_dir' via YAML/CLI : with data_dir='.../classification/<dataset>'"
        )

    args.data_dir = data_dir
    return args


# =========================
# Entraînement/Inférence
# =========================
def train(args: Namespace, datapath: Path) -> None:
    """
    Inférence :
    - lit les TIFF sous <data_dir>/CTC/<dataset>/<seqName>/
    - écrit sous results/<dataset>/test/CTC/<seqName>/
    """
    # Init distrib
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    device = torch.device(args.device)

    # Seeds
    seed = args.seed + utils.get_rank()
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    # Modèle
    model, criterion = build_model(args)
    model.to(device)
    model.train_model = False
    model.eval()
    args.eval_only = True
    criterion.eval_only = True

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('NUM TRAINABLE MODEL PARAMS:', n_parameters)

    # Charger le checkpoint
    print(f"[DEBUG] Using checkpoint = {args.resume} | exists={Path(args.resume).exists()}")
    model_without_ddp = utils.load_model(model_without_ddp, args)

    # Séquences à traiter (limitables avec --only)
    folderpaths = _list_sequences(datapath, args_cmd.only)

    print("[INFO] dataset   =", args.dataset)
    print("[INFO] data_dir  =", args.data_dir)
    print("[INFO] datapath  =", str(datapath.resolve()))
    print("[INFO] out_base  =", str(args.output_dir))
    print("[DEBUG] Séquences retenues :", [p.name for p in folderpaths])
    for p in folderpaths:
        n = len(list(p.glob("*.tif")))
        print(f"[DEBUG] {p.name} -> {n} tif | {p.resolve()}")

    if not getattr(args, "masks", True):
        print("[WARN] masks=False -> inférence sans GT; évaluation/alignement GT désactivés.")

    # Base de sortie : results/<dataset>/test/CTC/
    base_out = _coerce_path(args.output_dir)  # results/<dataset>/test
    base_out.mkdir(parents=True, exist_ok=True)
    base_out = base_out / "CTC"
    base_out.mkdir(parents=True, exist_ok=True)

    # Boucle inférence — on passe la BASE 'CTC' à l'engine (il créera /<seqName>)
    total_frames = 0
    start_time = time.time()
    for f_idx, folderpath in enumerate(folderpaths):
        fps = sorted(list(folderpath.glob("*.tif")))
        total_frames += len(fps)

        # IMPORTANT : output_dir = base CTC (pas /<seqName>)
        args_seq = argparse.Namespace(**vars(args))
        args_seq.output_dir = base_out  # Path requis par engine.py

        Pipeline = pipeline(model, fps, args_seq)
        Pipeline.forward()

        # Nettoyage local au vrai dossier de la séquence créé par l'engine
        seq_dir = base_out / folderpath.name
        if f_idx == len(folderpaths) - 1 and Pipeline.all_videos_same_size:
            Pipeline.display_enc_map(save=False, last=True)
        elif not Pipeline.all_videos_same_size:
            ts_dir = seq_dir / 'two_stage'
            if ts_dir.exists():
                shutil.rmtree(ts_dir)

    # Stats globales (au niveau de CTC/)
    total_time = time.time() - start_time
    fps_v = total_frames / total_time if total_time > 0 else 0.0
    with open(str(base_out / 'FPS.txt'), 'w') as file:
        file.write(f"Frames per second (FPS): {fps_v:2f}\n")


# =========================
# Sacred : config minimale
# =========================
@ex.config
def my_config():
    # Valeurs par défaut; remplacées par le YAML du RUN + CLI 'with'
    res_name = 'default'
    dataset = 'moma'


@ex.main
def load_config(_config, _run):
    """Sacred sert uniquement à charger YAML/CLI et afficher la config."""
    sacred_commands.print_config(_run)


# =========================
# Main
# =========================
def _show_cli_overrides(label: str):
    print(f"[CFG-TRACE] {label} | sys.argv = {' '.join(sys.argv)}")


if __name__ == '__main__':
    # Montrer ce que reçoit le script
    _show_cli_overrides("raw argv (pre-parse)")

    # Récupérer res_name depuis la ligne de commande (sans ex.run_commandline()).
    overrides = _parse_with_overrides(sys.argv)
    res_name = overrides.get('res_name', 'default')

    # Localiser le dossier du RUN d'entraînement
    if args_cmd.results_path is not None:
        respath = Path(args_cmd.results_path) / res_name
    else:
        respath = filepath.parents[1] / 'results' / res_name

    # Ajouter UNIQUEMENT le YAML du RUN (archi/hparams)
    cfg_run = respath / 'config.yaml'
    if not cfg_run.exists():
        raise FileNotFoundError(
            f"[CFG] Introuvable : {cfg_run}\n"
            f"       (attendu: dossier du run '{res_name}' sous {respath.parent})"
        )
    print(f"[CFG-TRACE] using run config.yaml : {cfg_run}")
    ex.add_config(str(cfg_run))

    # Passe unique : RUN + CLI 'with ...'
    args_cfg = ex.run_commandline().config
    args = utils.nested_dict_to_namespace(args_cfg)

    # Résumé des hyperparamètres critiques AVANT fix des chemins
    print("[CFG-DEBUG] Hyperparamètres critiques (avant _fix_paths_for_inference) :")
    print(f"  backbone     = {getattr(args, 'backbone', None)}")
    print(f"  dataset      = {getattr(args, 'dataset', None)}")
    print(f"  target_size  = {getattr(args, 'target_size', None)}")
    print(f"  crop         = {getattr(args, 'crop', None)}")
    print(f"  num_workers  = {getattr(args, 'num_workers', None)}")
    print(f"  data_dir     = {getattr(args, 'data_dir', None)}")
    print(f"  output_dir   = {getattr(args, 'output_dir', None)}")
    print(f"  resume       = {getattr(args, 'resume', None)}")
    print("------------------------------------------------------------------")

    # Forcer les chemins d'INFÉRENCE (sans toucher aux YAML d’entraînement)
    args = _fix_paths_for_inference(args, respath)

    # Résumé après fix des chemins
    print("[CFG-DEBUG] Chemins d'inférence (après _fix_paths_for_inference) :")
    print(f"  data_dir     = {args.data_dir}")
    print(f"  output_dir   = {args.output_dir}")
    print(f"  resume       = {args.resume}")
    print("------------------------------------------------------------------")

    # Chemin des données exportées : <data_dir>/CTC/<dataset>
    args.data_dir = _coerce_path(args.data_dir)
    args.output_dir = _coerce_path(args.output_dir)
    datapath = args.data_dir / 'CTC' / args.dataset

    # Go !
    train(args, datapath)
