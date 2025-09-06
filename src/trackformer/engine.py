# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Tuple, Any, Dict, List
import numpy as np
import PIL
import torch
import cv2
from pathlib import Path
from tqdm import tqdm
from ast import literal_eval
import ffmpeg
import time
from skimage.measure import label
import matplotlib.pyplot as plt

from .util import data_viz
from .util import misc as utils
from .util import box_ops
from .datasets.transforms import Normalize, ToTensor, Compose

import os, csv, json
from datetime import datetime

# ============================================================
# Triplet diagnostics (non-destructif, pas de reorder/skip)
# ============================================================

class TripletStats:
    def __init__(self):
        self.batches = 0
        self.constructed_triplets = 0
        self.reordered_triplets = 0     # on ne réordonne jamais ici
        self.skipped_batches = 0        # on ne skippe jamais ici
        self.bad_ctc = 0
        self.bad_nonconsecutive = 0
        self.missing_keys = 0
        self.bad_shape = 0
        self.examples: List[Dict[str, Any]] = []

def _triplet_log_paths(args):
    out_dir = getattr(args, "output_dir", ".")
    return (os.path.join(out_dir, "triplet_stats.jsonl"),
            os.path.join(out_dir, "triplet_issues.csv"))


def _to_int(x, default=None) -> int:
    try:
        if isinstance(x, torch.Tensor):
            return int(x.item())
        return int(x)
    except Exception:
        return default

# alias (certain vieux bouts de code appellent _tensor_to_int)
_tensor_to_int = _to_int

def _meta_from_slot(slot_dict: Dict[str, Any], fallback_root: Dict[str, Any] = None) -> Tuple[int, int]:
    """Récupère ctc_id/frame_id dans un sous-cible (prev/cur/fut).
       Fallback possible depuis la cible racine si le slot ne les porte pas."""
    ctc = _to_int(slot_dict.get("ctc_id", None))
    fr  = _to_int(slot_dict.get("frame_id", None))
    if (ctc is None or fr is None) and isinstance(fallback_root, dict):
        if ctc is None: ctc = _to_int(fallback_root.get("ctc_id", None))
        if fr  is None: fr  = _to_int(fallback_root.get("frame_id", None))
    return ctc, fr

def validate_triplet_item(item: Dict[str, Any], stats: TripletStats,
                          keep_examples: bool=False, max_examples: int=10) -> bool:
    """Valide un SEUL item (il contient déjà prev/cur/fut). Ne modifie rien."""
    try:
        main = item["main"]
        prev = main["prev_target"]; cur = main["cur_target"]; fut = main["fut_target"]
    except Exception:
        stats.missing_keys += 1
        if keep_examples and len(stats.examples) < max_examples:
            stats.examples.append({"reason": "missing_main_or_slots"})
        return False

    ctc_prev, fr_prev = _meta_from_slot(prev, item)
    ctc_cur,  fr_cur  = _meta_from_slot(cur,  item)
    ctc_fut,  fr_fut  = _meta_from_slot(fut,  item)

    if None in (ctc_prev, ctc_cur, ctc_fut, fr_prev, fr_cur, fr_fut):
        stats.missing_keys += 1
        if keep_examples and len(stats.examples) < max_examples:
            stats.examples.append({
                "reason": "missing_ctc_or_frame",
                "ctc": [ctc_prev, ctc_cur, ctc_fut],
                "fr":  [fr_prev,  fr_cur,  fr_fut],
            })
        return False

    if not (ctc_prev == ctc_cur == ctc_fut):
        stats.bad_ctc += 1
        if keep_examples and len(stats.examples) < max_examples:
            stats.examples.append({
                "reason": "ctc_mismatch",
                "ctc": [ctc_prev, ctc_cur, ctc_fut],
                "fr":  [fr_prev,  fr_cur,  fr_fut],
            })
        return False

    # ordre strict croissant ET consécutif
    if not (fr_prev < fr_cur < fr_fut) or not (fr_prev + 1 == fr_cur and fr_cur + 1 == fr_fut):
        stats.bad_nonconsecutive += 1
        if keep_examples and len(stats.examples) < max_examples:
            stats.examples.append({
                "reason": "non_consecutive_or_bad_order",
                "ctc": [ctc_prev, ctc_cur, ctc_fut],
                "fr":  [fr_prev,  fr_cur,  fr_fut],
            })
        return False

    return True

def validate_triplet_batch(targets: List[Dict[str, Any]], stats: TripletStats,
                           keep_examples: bool=False) -> float:
    """Vérifie une liste d’items (taille = nb de centres). Ne modifie rien."""
    stats.batches += 1
    stats.constructed_triplets += len(targets)
    oks = []
    for it in targets:
        oks.append(validate_triplet_item(it, stats, keep_examples=keep_examples))
    return float(sum(oks)) / max(1, len(oks))

# ---------- helpers pour extraire ctc_id / frame_id ----------
def _find_key_recursive(d, key):
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            out = _find_key_recursive(v, key)
            if out is not None:
                return out
    elif isinstance(d, (list, tuple)):
        for v in d:
            out = _find_key_recursive(v, key)
            if out is not None:
                return out
    return None

def _to_int(x, default=None):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return int(x.item())
        return int(x)
    except Exception:
        return default

def _extract_ids_anywhere(t):
    # Cherche ctc_id / frame_id n'importe où dans la structure
    ctc = t.get("ctc_id", None) if isinstance(t, dict) else None
    fr  = t.get("frame_id", None) if isinstance(t, dict) else None
    if ctc is None: ctc = _find_key_recursive(t, "ctc_id")
    if fr  is None: fr  = _find_key_recursive(t, "frame_id")
    # fallback possible (selon ton dataset)
    if fr is None: fr = _find_key_recursive(t, "framenb")
    return _to_int(ctc), _to_int(fr)

# ---------- vérifie & réordonne un batch (groupes de 3) ----------
def _check_and_fix_triplet_batch(samples, targets, stats, keep_examples=False, max_examples=20):
    """
    Pour chaque triplet (t-1,t,t+1), on impose:
      - même ctc_id
      - frames strictement croissants ET consécutifs
    On réordonne si possible (tri par frame + check consécutif), sinon on laisse tel quel
    ou on rejette le batch si TRIPLET_STRICT=1 et que l'erreur est bloquante.
    """
    import os, numpy as _np, torch as _torch
    strict = os.environ.get("TRIPLET_STRICT", "0") == "1"

    B = len(targets) if targets is not None else 0
    if B == 0 or (B % 3) != 0:
        stats.bad_shape += 1
        stats.skipped_batches += 1
        return samples, targets, False, {"reason": "bad_shape", "B": B}

    stats.batches += 1
    triplets = B // 3
    stats.constructed_triplets += triplets

    perm = _np.arange(B, dtype=_np.int64)
    reordered_local = 0

    for g in range(0, B, 3):
        ctcs, frms, miss = [], [], False
        for t in targets[g:g+3]:
            ctc, fr = _extract_ids_anywhere(t)
            if ctc is None or fr is None:
                miss = True
            ctcs.append(ctc); frms.append(fr)

        if miss:
            stats.missing_keys += 1
            if keep_examples and len(stats.examples) < max_examples:
                stats.examples.append({"group": g//3, "reason": "missing_keys", "ctc": ctcs, "frames": frms})
            if strict:
                stats.skipped_batches += 1
                return samples, targets, False, {"reason": "missing_keys", "ctc": ctcs, "frames": frms}
            continue

        if not (ctcs[0] == ctcs[1] == ctcs[2]):
            stats.bad_ctc += 1
            if keep_examples and len(stats.examples) < max_examples:
                stats.examples.append({"group": g//3, "reason": "ctc_mismatch", "ctc": ctcs, "frames": frms})
            if strict:
                stats.skipped_batches += 1
                return samples, targets, False, {"reason": "ctc_mismatch", "ctc": ctcs, "frames": frms}
            continue

        # Ordre par frame si nécessaire
        if not (frms[0] < frms[1] < frms[2]):
            order = _np.argsort(frms)
            sorted_fr = [frms[i] for i in order]
            if (sorted_fr[0] + 1 == sorted_fr[1]) and (sorted_fr[1] + 1 == sorted_fr[2]):
                perm[g:g+3] = perm[g:g+3][order]
                reordered_local += 1
            else:
                stats.bad_nonconsecutive += 1
                if keep_examples and len(stats.examples) < max_examples:
                    stats.examples.append({"group": g//3, "reason": "non_consecutive", "frames": frms})
                if strict:
                    stats.skipped_batches += 1
                    return samples, targets, False, {"reason": "non_consecutive", "frames": frms}
                continue
        else:
            # déjà croissants: vérifier "consécutifs"
            if not (frms[0] + 1 == frms[1] and frms[1] + 1 == frms[2]):
                stats.bad_nonconsecutive += 1
                if keep_examples and len(stats.examples) < max_examples:
                    stats.examples.append({"group": g//3, "reason": "non_consecutive", "frames": frms})
                if strict:
                    stats.skipped_batches += 1
                    return samples, targets, False, {"reason": "non_consecutive", "frames": frms}

    stats.reordered_triplets += reordered_local

    # appliquer permutation si nécessaire
    if not (perm == _np.arange(B)).all():
        idx = _torch.as_tensor(perm, device=samples.tensors.device, dtype=_torch.long)
        tensors = samples.tensors.index_select(0, idx)
        mask = samples.mask.index_select(0, idx) if getattr(samples, "mask", None) is not None else None
        samples = type(samples)(tensors, mask)
        targets = [targets[i] for i in perm.tolist()]

    return samples, targets, True, {"reordered_groups": int(reordered_local)}


# ============================================================
# Schéma de target (souple)
# ============================================================

def _assert_slot(slot: Dict[str, Any], slot_name: str):
    for key in ['boxes', 'labels', 'track_ids', 'flexible_divisions']:
        assert key in slot, f"missing {slot_name}.{key}"

    B = slot['boxes'].shape[0]
    # boxes: (N, 8) dans ce projet (xyxy + 2*div boxes) — on autorise >=4
    assert slot['boxes'].ndim == 2 and slot['boxes'].shape[0] == B and slot['boxes'].shape[1] >= 4, \
        f"bad shape for {slot_name}.boxes: {tuple(slot['boxes'].shape)}"

    # labels: 1D (N,) ou 2D (N,K) suivant le dataset (ex: moma => Nx2)
    assert slot['labels'].ndim in (1, 2), f"{slot_name}.labels must be 1D or 2D"
    assert slot['labels'].shape[0] == B, f"{slot_name}.labels 1st dim must match boxes"

    assert slot['track_ids'].shape == (B,), f"{slot_name}.track_ids must be (N,)"
    assert slot['flexible_divisions'].shape == (B,), f"{slot_name}.flexible_divisions must be (N,)"

def assert_target_schema(t: Dict[str, Any]):
    assert 'main' in t and all(k in t['main'] for k in ['prev_target','cur_target','fut_target']), "missing main.{prev,cur,fut}_target"
    _assert_slot(t['main']['prev_target'], 'prev_target')
    _assert_slot(t['main']['cur_target'],  'cur_target')
    _assert_slot(t['main']['fut_target'],  'fut_target')

    # ctc_id/frame_id peuvent être au niveau racine ou au niveau des slots : on ne force pas ici
    # mais s’ils sont présents ils doivent être scalaires
    for key in ('ctc_id', 'frame_id'):
        if key in t:
            v = t[key]
            if isinstance(v, torch.Tensor):
                assert v.ndim == 0, f"{key} must be scalar tensor"

# ============================================================
# Perte et boucle train/val
# ============================================================

def calc_loss_for_training_methods(outputs, targets, criterion):
    outputs_split = {}
    losses = {}
    training_methods = outputs['training_methods']  # p.ex. ['main', 'OD', ...]
    outputs_split = {}

    for training_method in training_methods:

        target_TM = targets[0][training_method]
        outputs_TM = utils.split_outputs(outputs, target_TM)

        if training_method == 'main':
            if 'two_stage' in outputs:
                outputs_TM['two_stage'] = outputs['two_stage']
                outputs_split['two_stage'] = outputs['two_stage']
            if 'OD' in outputs:
                outputs_TM['OD'] = outputs['OD']
                outputs_split['OD'] = outputs['OD']

        outputs_split[training_method] = outputs_TM
        losses = criterion(outputs_TM, targets, losses, training_method)

    outputs_split['prev_outputs'] = outputs['prev_outputs']
    outputs_split['prev_prev_outputs'] = outputs['prev_prev_outputs']
    return outputs_split, losses

# ------------------------------------------------------------

def _ensure_output_dir(out_dir: Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def _triplet_log_paths(args):
    out_dir = _ensure_output_dir(getattr(args, "output_dir", "."))
    return (out_dir / "triplet_stats.jsonl"), (out_dir / "triplet_issues.csv")

def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args,
    interval: int = 50
):
    dataset = 'train'
    model.train()
    criterion.train()

    ids = np.concatenate(([0], np.random.randint(0, len(data_loader), args.num_plots)))
    metrics_dict: Dict[str, Any] = {}

    # triplet diag
    triplet_stats = TripletStats()
    stats_path, issues_path = _triplet_log_paths(args)
    csv_initialized = False
    log_triplet_examples = True

    for i, (samples, targets) in enumerate(data_loader):
        samples = samples.to(args.device)
        targets = [utils.nested_dict_to_device(t, args.device) for t in targets]

        # --- check triplets (non destructif) ---
        ok_ratio = validate_triplet_batch(targets, triplet_stats, keep_examples=log_triplet_examples)
        if (i % 20 == 0) or (i == 0):
            print(f"[TRIPLET-CHECK][train] batch={i}  ok_items={ok_ratio*100:.1f}%  "
                  f"bad_ctc={triplet_stats.bad_ctc}  bad_noncons={triplet_stats.bad_nonconsecutive}  "
                  f"missing={triplet_stats.missing_keys}", flush=True)

        # dump d’exemples problématiques (optionnel, pas bloquant)
        if log_triplet_examples and triplet_stats.examples:
            try:
                if not csv_initialized:
                    with open(issues_path, "w", newline="", encoding="utf-8") as f:
                        w = csv.writer(f)
                        w.writerow(["epoch", "batch", "reason", "detail"])
                    csv_initialized = True
                # on flush puis on clear pour ne pas dupliquer
                with open(issues_path, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    for ex in triplet_stats.examples:
                        w.writerow([epoch, i, ex.get("reason", "?"), json.dumps(ex)])
                triplet_stats.examples.clear()
            except Exception as e:
                print(f"[TRIPLET-WARN] write CSV failed: {e}", flush=True)

        # schéma (souple)
        for k in range(len(targets)):
            try:
                assert_target_schema(targets[k])
            except AssertionError as e:
                # on remonte l’erreur pour qu’elle soit visible – ici on ne “répare” rien
                raise

        # forward
        outputs, targets, _, _, _ = model(samples, targets)

        # (sanity) debug divisions 1× au début
        if (i == 0) and getattr(args, "div_sanity_check", True):
            utils.validate_divisions_in_targets(
                targets,
                training_method=getattr(args, "training_method", "main"),
                input_target_name="prev_target",
                output_target_name="cur_target",
                div_event_key="gt_div"
            )

        import os as _os
        if getattr(args, "debug_division", False) or _os.environ.get("DIVDBG") == "1":
            for k in range(min(2, len(targets))):
                utils.debug_div_event_alignment(
                    targets[k],
                    training_method=getattr(args, "training_method", "main"),
                    input_target_name="prev_target",
                    output_target_name="cur_target",
                    gt_div_key="gt_div",
                    pred_div_key="pred_div",
                    max_print=5
                )

        torch.cuda.empty_cache()

        # pertes
        outputs, loss_dict = calc_loss_for_training_methods(outputs, targets, criterion)

        # neutralisation optionnelle de la CE CoMOT
        weight_dict = criterion.weight_dict.copy()
        if hasattr(args, "CoMOT_loss_ce") and not args.CoMOT_loss_ce:
            for k in list(loss_dict.keys()):
                if k.startswith("CoMOT_loss_ce"):
                    loss_dict[k] = torch.nan_to_num(loss_dict[k], nan=0.0, posinf=0.0, neginf=0.0).detach() * 0.0
            for k in list(weight_dict.keys()):
                if k.startswith("CoMOT_loss_ce"):
                    weight_dict[k] = 0.0

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

        if i == 0:
            lr = np.zeros((1, len(optimizer.param_groups)))
            for p, param_group in enumerate(optimizer.param_groups):
                lr[0, p] = param_group['lr']

        main_targets = [target['main']['cur_target'] for target in targets]

        acc_dict = {}
        if targets[0]['track']:
            acc_dict = utils.calc_track_acc(acc_dict, outputs['main'], main_targets, args)

            import os as _os2
            if (getattr(args, "debug_division", False) or _os2.environ.get("DIVDBG") == "1") and 'dbg_div' in acc_dict:
                d = np.array(acc_dict['dbg_div'])
                if d.ndim == 2 and d.shape[1] == 8:
                    sums = d[:, :6].sum(axis=0)
                    means = np.nanmean(d[:, 6:], axis=0)
                    g0, g1, g2, g3, g4, g5 = sums
                    m6, m7 = means
                elif d.ndim == 1 and d.shape[0] == 8:
                    g0, g1, g2, g3, g4, g5, m6, m7 = d
                else:
                    flat = d.reshape(-1)
                    g0, g1, g2, g3, g4, g5, m6, m7 = flat[:8]

                print(f"[DIV-DBG] gt_div={int(g0)} pred_div>thr(noGT)={int(g1)} "
                      f"pred_div>thr&GT={int(g2)} pred_div<=thr&GT={int(g3)} "
                      f"iou_div_hits/trials={int(g4)}/{int(g5)} "
                      f"mean_pred_div(gt)={float(m6):.3f} mean_pred_div(noGT)={float(m7):.3f}",
                      flush=True)

        # acc det
        if outputs['prev_prev_outputs'] is not None:
            det_outputs = outputs['prev_prev_outputs']
            det_targets = [target['main']['prev_prev_target'] for target in targets]
        elif outputs['prev_outputs'] is not None:
            det_outputs = outputs['prev_outputs']
            det_targets = [target['main']['prev_target'] for target in targets]
        else:
            det_outputs = outputs['main']
            det_targets = main_targets

        acc_dict = utils.calc_bbox_acc(acc_dict, det_outputs, det_targets, args)

        if 'OD' in outputs:
            OD_targets = [target['OD']['cur_target'] for target in targets]
            acc_dict = utils.calc_bbox_acc(acc_dict, outputs['OD'], OD_targets, args, text='OD_L1_')

        metrics_dict = utils.update_metrics_dict(metrics_dict, acc_dict, loss_dict, weight_dict, i, lr)

        if (i in ids and (epoch % 5 == 0 or epoch == 1)) and args.data_viz:
            data_viz.plot_results(
                outputs, targets, samples.tensors, args.output_dir,
                folder=dataset + '_outputs',
                filename=f'Epoch{epoch:03d}_Step{i:06d}.png', args=args
            )

        if i > 0 and (i % interval == 0 or i == len(data_loader) - 1):
            utils.display_loss(metrics_dict, i, len(data_loader), epoch=epoch, dataset=dataset)

        # impression finale DIV-DBG (robuste)
        if 'dbg_div' in acc_dict:
            d = np.array(acc_dict['dbg_div'])
            if d.ndim == 2 and d.shape[1] == 8:
                sums = d[:, :6].sum(axis=0)
                means = np.nanmean(d[:, 6:], axis=0)
                g0, g1, g2, g3, g4, g5 = sums
                m6, m7 = means
            elif d.ndim == 1 and d.shape[0] == 8:
                g0, g1, g2, g3, g4, g5, m6, m7 = d
            else:
                flat = d.reshape(-1)
                g0, g1, g2, g3, g4, g5, m6, m7 = flat[:8]

            print(f"[DIV-DBG] gt_div={int(g0)}  pred_div>thr(noGT)={int(g1)}  "
                  f"pred_div>thr&GT={int(g2)}  pred_div<=thr&GT={int(g3)}  "
                  f"iou_div_hits/trials={int(g4)}/{int(g5)}  "
                  f"mean_pred_div(gt)={float(m6):.3f}  mean_pred_div(noGT)={float(m7):.3f}",
                  flush=True)

    # --- Résumé / dump JSONL des stats triplets (train) ---
    summary = {
        "phase": "train",
        "epoch": int(epoch),
        "batches": triplet_stats.batches,
        "constructed_triplets": triplet_stats.constructed_triplets,
        "reordered_triplets": triplet_stats.reordered_triplets,
        "skipped_batches": triplet_stats.skipped_batches,
        "bad_ctc": triplet_stats.bad_ctc,
        "bad_nonconsecutive": triplet_stats.bad_nonconsecutive,
        "missing_keys": triplet_stats.missing_keys,
        "bad_shape": triplet_stats.bad_shape,
    }
    print("[TRIPLET-STATS][train]", summary, flush=True)
    try:
        with open(_triplet_log_paths(args)[0], "a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")
    except Exception as e:
        print(f"[TRIPLET-WARN] write stats failed: {e}", flush=True)

    return metrics_dict

# ------------------------------------------------------------

@torch.no_grad()
def evaluate(model, criterion, data_loader, args, epoch: int = None, interval=50):

    model.eval()
    criterion.eval()
    dataset = 'val'
    ids = np.concatenate(([0], np.random.randint(0, len(data_loader), args.num_plots)))

    metrics_dict: Dict[str, Any] = {}

    bs = getattr(data_loader, "batch_sampler", None)
    name = type(bs).__name__ if bs is not None else "None"
    print(f"[VAL] batch_sampler: {name}", flush=True)

    # Info seulement, pas d'assert fort ici : la validité des triplets
    # est déjà garantie par _check_and_fix_triplet_batch plus bas.
    if bs is None:
        print("[VAL][WARN] DataLoader sans batch_sampler explicite (probable batch_size passé).", flush=True)
    else:
        if hasattr(bs, "last_batch_indices") and hasattr(bs, "build_triplet_info"):
            try:
                _ = bs.build_triplet_info(getattr(bs, "last_batch_indices", None))
            except Exception as e:
                print(f"[VAL][WARN] build_triplet_info indisponible/échec: {e}", flush=True)
        else:
            print("[VAL][INFO] Sampler sans build_triplet_info (OK) — validation des triplets au runtime.", flush=True)
            
    triplet_stats_eval = TripletStats()

    for i, (samples, targets) in enumerate(data_loader):
        samples = samples.to(args.device)
        targets = [utils.nested_dict_to_device(t, args.device) for t in targets]
        
        samples, targets, ok, info = _check_and_fix_triplet_batch(
            samples, targets, triplet_stats_eval, keep_examples=False
        )
        if not ok:
            # batch invalide en mode strict (ou shape ≠ 3k) -> on log et on passe au suivant
            print(f"[VAL][TRIPLET-FAIL] batch={i} reason={info.get('reason','?')} details={info}", flush=True)
            continue


        ok_ratio = validate_triplet_batch(targets, triplet_stats_eval, keep_examples=False)
        if (i % 20 == 0) or (i == 0):
            print(f"[TRIPLET-CHECK][val] batch={i}  ok_items={ok_ratio*100:.1f}%  "
                  f"bad_ctc={triplet_stats_eval.bad_ctc}  bad_noncons={triplet_stats_eval.bad_nonconsecutive}  "
                  f"missing={triplet_stats_eval.missing_keys}", flush=True)

        if i < 3:
            print("[VAL] BATCH LEN =", len(targets))

        outputs, targets, _, _, _ = model(samples, targets)
        outputs, loss_dict = calc_loss_for_training_methods(outputs, targets, criterion)

        if (i == 0) and getattr(args, "div_sanity_check", True):
            utils.validate_divisions_in_targets(
                targets,
                training_method=getattr(args, "training_method", "main"),
                input_target_name="prev_target",
                output_target_name="cur_target",
                div_event_key="gt_div"
            )

        import os as _os
        if getattr(args, "debug_division", False) or _os.environ.get("DIVDBG") == "1":
            for k in range(min(2, len(targets))):
                utils.debug_div_event_alignment(
                    targets[k],
                    training_method=getattr(args, "training_method", "main"),
                    input_target_name="prev_target",
                    output_target_name="cur_target",
                    gt_div_key="gt_div",
                    pred_div_key="pred_div",
                    max_print=5
                )

        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        loss_dict['loss'] = losses

        main_targets = [target['main']['cur_target'] for target in targets]

        acc_dict = {}
        if targets[0]['track']:
            acc_dict = utils.calc_track_acc(acc_dict, outputs['main'], main_targets, args)

        if outputs['prev_prev_outputs'] is not None:
            det_outputs = outputs['prev_prev_outputs']
            det_targets = [target['main']['prev_prev_target'] for target in targets]
        elif outputs['prev_outputs'] is not None:
            det_outputs = outputs['prev_outputs']
            det_targets = [target['main']['prev_target'] for target in targets]
        else:
            det_outputs = outputs['main']
            det_targets = main_targets

        acc_dict = utils.calc_bbox_acc(acc_dict, det_outputs, det_targets, args)

        if 'OD' in outputs:
            OD_targets = [target['OD']['cur_target'] for target in targets]
            acc_dict = utils.calc_bbox_acc(acc_dict, outputs['OD'], OD_targets, args, text='OD_L1_')

        metrics_dict = utils.update_metrics_dict(metrics_dict, acc_dict, loss_dict, weight_dict, i)

        if i in ids and (epoch % 5 == 0 or epoch == 1) and args.data_viz:
            data_viz.plot_results(
                outputs, targets, samples.tensors, args.output_dir,
                folder=dataset + '_outputs',
                filename=f'Epoch{epoch:03d}_Step{i:06d}.png', args=args
            )

        if i > 0 and (i % interval == 0 or i == len(data_loader) - 1):
            utils.display_loss(metrics_dict, i, len(data_loader), epoch=epoch, dataset=dataset)

    summary = {
        "phase": "val",
        "epoch": int(epoch) if epoch is not None else -1,
        "batches": triplet_stats_eval.batches,
        "constructed_triplets": triplet_stats_eval.constructed_triplets,
        "reordered_triplets": triplet_stats_eval.reordered_triplets,
        "skipped_batches": triplet_stats_eval.skipped_batches,
        "bad_ctc": triplet_stats_eval.bad_ctc,
        "bad_nonconsecutive": triplet_stats_eval.bad_nonconsecutive,
        "missing_keys": triplet_stats_eval.missing_keys,
        "bad_shape": triplet_stats_eval.bad_shape,
    }
    print("[TRIPLET-STATS][val]", summary, flush=True)
    try:
        with open(_triplet_log_paths(args)[0], "a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")
    except Exception as e:
        print(f"[TRIPLET-WARN] write stats failed: {e}", flush=True)

    return metrics_dict

# ============================================================
# Pipeline vidéo (inchangé sauf menues sécurités d’E/S)
# ============================================================

@torch.no_grad()
class pipeline():
    def __init__(self, model, fps, args, display_all_aux_outputs=False):

        self.model = model
        self.display_all_aux_outputs = display_all_aux_outputs

        # Make a new folder with CTC folder number
        self.output_dir = (Path(args.output_dir) / fps[0].parts[-2])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.write_video = True

        # Convert args into class attributes
        self.args = args
        self.target_size = args.target_size
        self.num_queries = args.num_queries
        self.device = args.device
        self.masks = args.masks
        self.use_dab = args.use_dab
        self.two_stage = args.two_stage
        self.return_intermediate_masks = args.return_intermediate_masks
        self.track = args.tracking
        self.all_videos_same_size = True

        self.data_viz_folder = 'data_viz'
        (self.output_dir / self.data_viz_folder).mkdir(exist_ok=True)

        self.fps = fps
        self.threshold = 0.5
        self.mask_threshold = 0.5
        self.alpha = 0.25

        self.method = 'object_detection' if not self.track else 'track'

        # Image normalization (comme ResNet)
        self.normalize = Compose([ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        # hooks (optionnels)
        self.use_hooks = False
        if self.use_hooks:
            (self.output_dir / self.data_viz_folder / 'attn_weight_maps').mkdir(exist_ok=True)

        if isinstance(self.target_size[0], str):
            self.target_size = literal_eval(self.target_size)

        np.random.seed(1)

        # couleurs
        self.all_colors = np.array([tuple((255*np.random.random(3))) for _ in range(10000)])
        self.all_colors[:6] = np.array([[0.,0.,255.],[0.,255.,0.],[255.,0.,0.],[255.,0.,255.],[0.,255.,255.],[255.,255.,0.]])
        self.all_colors = np.concatenate((self.all_colors, np.array([[127.5,127.5,0.],[0.,127.5,0.],[0.,0.,127.5],[0.,127.5,127.5],[127.5,0.,0.],[255.,127.5,255.],[255.,127.5,0.],[127.5,255.,127.5],[255.,255.,127.5],[127.5,127.5,255.],[50.,200,200],[255.,127.5,127.5],[75,75,150.],[127.5,255.,255.],[255.,255.,255.],[127.5,127.5,127.5]])))

        if args.two_stage:
            (self.output_dir.parent / 'two_stage').mkdir(exist_ok=True)
            (self.output_dir / self.data_viz_folder / 'two_stage').mkdir(exist_ok=True)
            self.enc_map = None
        else:
            self.query_box_locations = [np.zeros((0,4)) for _ in range(args.num_queries)]

        self.query_box_locations = [np.zeros((0,4)) for _ in range(args.num_queries)]

        self.display_decoder_aux = True if args.dataset == 'moma' else False
        if self.display_decoder_aux:
            (self.output_dir / self.data_viz_folder / 'decoder_bbox_outputs').mkdir(exist_ok=True)
            self.num_decoder_frames = 1

        img = PIL.Image.open(fps[0], mode='r')
        self.color_stack = np.zeros((len(fps), img.size[1], img.size[0], 3))

        if np.max(img) < 2**8:
            self.max_val = 255
        elif np.max(img) < 2**12:
            self.max_val = 2**12-1
        else:
            self.max_val = 2**16-1

    def update_query_box_locations(self, pred_boxes):
        oq_boxes = pred_boxes[self.object_indices, :4].cpu().numpy()
        oq_boxes[:,1::2] = np.clip(oq_boxes[:,1::2] * self.target_size[0], 0, self.target_size[0])
        oq_boxes[:,0::2] = np.clip(oq_boxes[:,::2] * self.target_size[1], 0, self.target_size[1])

        oq_indices_norm = (self.object_indices - self.num_TQs)
        for oq_ind, oq_box in zip(oq_indices_norm, oq_boxes):
            self.query_box_locations[oq_ind] = np.append(self.query_box_locations[oq_ind], oq_box[None], axis=0)

    def split_up_divided_cells(self):
        self.div_track = -1 * np.ones((len(self.all_indices) + len(self.div_indices)), dtype=np.uint16)
        nb_divs = 0
        for div_ind in self.div_indices:
            ind = np.where(self.all_indices == div_ind)[0][0]
            self.max_cellnb += 1
            self.cells = np.concatenate((self.cells[:ind+1], [self.max_cellnb], self.cells[ind+1:]))
            self.all_indices = np.concatenate((self.all_indices[:ind], self.all_indices[ind:ind+1], self.all_indices[ind:]))
            self.track_indices = np.concatenate((self.track_indices[:ind], self.track_indices[ind:ind+1], self.track_indices[ind:]))
            self.div_track[ind:ind+2] = div_ind
            nb_divs += 1
        self.object_indices += nb_divs

        if self.i > 0:
            self.new_cells = self.cells == 0
        if 0 in self.cells:
            self.max_cellnb += 1
            self.cells[self.cells == 0] = np.arange(self.max_cellnb, self.max_cellnb + sum(self.cells == 0), dtype=np.uint16)
            self.max_cellnb = np.max(self.cells)

    def update_div_boxes(self, pred_boxes, pred_masks=None):
        unique_divs = np.unique(self.div_track[self.div_track != -1])
        for unique_div in unique_divs:
            div_ids = (self.div_track == unique_div).nonzero()[0]
            pred_boxes[div_ids[1], :4] = pred_boxes[div_ids[0], 4:]
            if pred_masks is not None:
                pred_masks[div_ids[1], :1] = pred_masks[div_ids[0], 1:]
        return pred_boxes[:, :4], pred_masks[:, 0]

    def preprocess_img(self, fp):
        self.img = cv2.imread(str(fp), cv2.IMREAD_ANYDEPTH)
        self.img = self.img / np.max(self.img)
        self.img = (255 * self.img).astype(np.uint8)
        img = PIL.Image.fromarray(self.img)
        self.img_size = img.size
        img_resized = img.resize((self.target_size[1], self.target_size[0])).convert('RGB')
        samples = self.normalize(img_resized)[0][None]
        samples = samples.to(self.device)
        return samples

    def get_track_object_div_indices(self, pred_logits):
        N,_ = pred_logits.shape
        self.num_TQs = N - self.num_queries
        keep = (pred_logits[:,0] > self.threshold)
        keep_div = (pred_logits[:,1] > self.threshold)
        keep_div[-self.num_queries:] = False
        keep_div[~keep] = False
        all_indices = keep.nonzero()[0]
        track_indices = np.array([ind for ind in all_indices if ind < self.num_TQs], dtype=int)
        object_indices = np.array([ind for ind in all_indices if ind >= self.num_TQs], dtype=int)
        div_indices = keep_div.nonzero()[0]
        return all_indices, track_indices, object_indices, div_indices

    def post_process_masks(self, masks, boxes):
        masks = cv2.resize(np.transpose(masks.cpu().numpy(), (1,2,0)), self.img_size)
        masks = masks[:,:,None] if masks.ndim == 2 else masks
        masks = np.transpose(masks, (-1,0,1))

        masks_filt = np.zeros((masks.shape))
        argmax = np.argmax(masks, axis=0)
        for m in range(masks.shape[0]):
            masks_filt[m, argmax==m] = masks[m, argmax==m]
        masks = masks_filt > self.mask_threshold

        keep_all_cells = np.ones(len(self.all_indices), dtype=bool)
        keep_track_cells = np.ones(len(self.track_indices), dtype=bool)
        keep_object_cells = np.ones(len(self.object_indices), dtype=bool)
        keep_div_cells = np.ones(len(self.div_indices), dtype=bool)

        for m, mask in enumerate(masks):
            if mask.sum() > 0:
                label_mask = label(mask)
                labels = np.unique(label_mask)
                labels = labels[labels != 0]
                largest_ind = np.argmax(np.array([label_mask[label_mask == label].sum() for label in labels]))
                new_mask = np.zeros_like(mask, dtype=bool)
                new_mask[label_mask == labels[largest_ind]] = True
                masks[m] = new_mask
            else:
                keep_all_cells[m] = False
                if self.all_indices[m] < self.num_TQs:
                    keep_track_cells[m] = False
                else:
                    keep_object_cells[m - len(self.track_indices)] = False
                    assert len(self.all_indices) - len(self.object_indices) == len(self.track_indices)
                index = self.all_indices[self.cells == self.cells[m]][0]
                if index in self.div_indices:
                    keep_div_cells[self.div_indices == index] = False
                    self.div_track[self.div_track == index] = -1
                    if index < self.num_TQs and self.prevcells[index] not in self.cells:
                        other_div_cell = self.cells[(self.div_track == index) * (self.cells != self.cells[m])]
                        self.cells[self.cells == other_div_cell] = self.prevcells[index]

        self.cells = self.cells[keep_all_cells]
        self.all_indices = self.all_indices[keep_all_cells]
        self.track_indices = self.track_indices[keep_track_cells]
        self.object_indices = self.object_indices[keep_object_cells]
        self.div_indices = self.div_indices[keep_div_cells]
        self.div_track = self.div_track[keep_all_cells]
        masks = masks[keep_all_cells]
        boxes = boxes[keep_all_cells]
        self.num_TQs -= (~keep_track_cells).sum()
        return masks, boxes

    def reset_vars(self):
        self.all_indices = None
        self.track_indices = None
        self.object_indices = None
        self.div_track = -1 * np.ones(len(self.cells), dtype=np.uint16)
        self.new_cells = None

    def forward(self):
        print(f'video {self.fps[0].parts[-2]}')

        if self.display_decoder_aux:
            random_nbs = np.random.choice(np.arange(1, len(self.fps)), self.num_decoder_frames)
            random_nbs = np.concatenate((random_nbs, random_nbs+1))

        ctc_data = np.zeros((0,4))
        targets = [{'main': {'cur_target': {}}}]
        self.max_cellnb = 0
        self.cells = np.zeros((0))

        for i, fp in enumerate(tqdm(self.fps)):
            self.reset_vars()
            self.fp = fp
            self.i = i

            if self.use_hooks and ((self.display_decoder_aux and i in random_nbs) or (self.display_all_aux_outputs and i > 0)):
                dec_attn_outputs = []
                hooks = [self.model.decoder.layers[layer_index].self_attn.register_forward_hook(lambda self, input, output: dec_attn_outputs.append(output)) for layer_index in range(len(self.model.decoder.layers))]

            samples = self.preprocess_img(fp)
            with torch.no_grad():
                outputs, targets, _, _, _ = self.model(samples, targets=targets)

            pred_logits = outputs['pred_logits'][0].sigmoid().cpu().numpy()
            self.all_indices, self.track_indices, self.object_indices, self.div_indices = self.get_track_object_div_indices(pred_logits)

            self.prevcells = np.copy(self.cells)
            self.cells = np.zeros((len(self.all_indices)), dtype=np.uint16)

            if len(self.all_indices) > 0:
                self.cells[:len(self.track_indices)] = self.prevcells[self.track_indices]
                self.split_up_divided_cells()

                boxes = outputs['pred_boxes'][0][self.all_indices]
                masks = None
                if self.masks:
                    masks = outputs['pred_masks'][0].sigmoid()[self.all_indices]

                boxes, masks = self.update_div_boxes(boxes, masks)
                if self.masks:
                    masks, boxes = self.post_process_masks(masks, boxes)

                if self.track:
                    targets[0]['main']['cur_target']['track_query_hs_embeds'] = outputs['hs_embed'][0, self.all_indices]
                    if self.args.init_boxes_from_masks:
                        boxes_encoded_from_masks = box_ops.masks_to_boxes(torch.tensor(masks), cxcywh=True).to(self.device)
                        targets[0]['main']['cur_target']['track_query_boxes'] = boxes_encoded_from_masks
                    else:
                        targets[0]['main']['cur_target']['track_query_boxes'] = boxes

                boxes = boxes.cpu().numpy()
                assert boxes.shape[0] == len(self.cells)
            else:
                boxes = None
                masks = None

            if True:
                self.update_query_box_locations(outputs['pred_boxes'][0])

            if self.img.ndim == 2:
                self.img = np.repeat(self.img[:,:,None], 3, -1)

            if self.track:
                color_frame = data_viz.plot_tracking_results(self.img, boxes, masks, self.all_colors[self.cells-1], self.div_track, self.new_cells)
            else:
                color_frame = data_viz.plot_tracking_results(self.img, boxes, masks, self.all_colors[:len(self.cells)], self.div_track, None)

            assert np.max(color_frame) <= 255 and np.min(color_frame) >= 0
            self.color_stack[i] = color_frame

            if self.use_hooks and ((self.display_decoder_aux and i in random_nbs) or (self.display_all_aux_outputs and i > 0)):
                for hook in hooks:
                    hook.remove()
                self.display_attn_maps(dec_attn_outputs)

            if masks is not None:
                ctc_data = self.save_ctc(ctc_data, masks)

            if i == 0:
                self.display_two_stage(outputs)

            display_proposal_index_on_img = False
            if self.display_all_aux_outputs and i in random_nbs:
                self.display_aux_preds(outputs)
                display_proposal_index_on_img = True

            if 'two_stage' in outputs:
                self.save_enc_map(outputs, display_proposal_index_on_img)

            torch.cuda.empty_cache()

            if len(self.all_indices) == 0:
                self.prevcells = None

            self.prev_outputs = outputs.copy()
            self.previmg = self.img.copy()

        start_time = time.time()
        print(f'{time.time() - start_time} seconds')

        if 'two_stage' in outputs:
            self.display_enc_map()

        if ctc_data is not None and self.masks:
            np.savetxt(self.output_dir / 'res_track.txt', ctc_data, fmt='%d')

        if self.write_video:
            self.save_video()

        if not self.two_stage:
            self.save_object_query_box_locations(outputs)

    def save_ctc(self,ctc_data,masks):

        if len(self.all_indices) > 0:

            label_mask = np.zeros(masks.shape[-2:],dtype=np.uint16)
            
            for m, cell in enumerate(self.cells):
                assert  masks[m].sum() > 0
                label_mask[masks[m] > 0] = cell

        max_cellnb = ctc_data.shape[0]

        ctc_cells_new = np.copy(self.cells)
        mask_copy = np.copy(label_mask)

        if len(self.prevcells) > 0:
            for c,cell in enumerate(self.prevcells):
                if cell in self.cells:
                    ctc_cell = self.ctc_cells[c]

                    # Cell Divided from previous frame
                    if self.div_track[self.cells == cell] != -1:
                        div_ind = self.div_track[self.cells == cell]
                        div_cells = self.cells[self.div_track == div_ind]
                        max_cellnb += 1
                        new_cell_1 = np.array([max_cellnb,self.i,self.i,ctc_cell])[None]
                        ctc_cells_new[self.cells == div_cells[0]] = max_cellnb 
                        label_mask[mask_copy == div_cells[0]] = max_cellnb
                        max_cellnb += 1
                        new_cell_2 = np.array([max_cellnb,self.i,self.i,ctc_cell])[None]
                        ctc_data = np.concatenate((ctc_data,new_cell_1,new_cell_2),axis=0)
                        ctc_cells_new[self.cells == div_cells[1]] = max_cellnb
                        label_mask[mask_copy == div_cells[1]] = max_cellnb

                    # Cell was tracked from previous frame
                    else:
                        ctc_data[ctc_cell-1,2] = self.i
                        ctc_cells_new[self.cells == cell] = self.ctc_cells[self.prevcells == cell]
                        label_mask[mask_copy == cell] = ctc_cell

        for c,cell in enumerate(self.cells):
            if len(self.prevcells) == 0 or cell not in self.prevcells and self.div_track[c] == -1:
                max_cellnb += 1
                new_cell = np.array([max_cellnb,self.i,self.i,0])
                ctc_data = np.concatenate((ctc_data,new_cell[None]),axis=0)

                ctc_cells_new[self.cells == cell] = max_cellnb     
                label_mask[mask_copy == cell] = max_cellnb

        self.ctc_cells = ctc_cells_new

        cv2.imwrite(str(self.output_dir / f'mask{self.i:03d}.tif'),label_mask)

        return ctc_data
        
    def save_object_query_box_locations(self,outputs):

        scale = 8 if self.args.dataset == 'moma' else 2
        wspacer = 5 * scale
        hspacer = 20 * scale

        max_area = [np.max(boxes[:,2] * boxes[:,3]) for boxes in self.query_box_locations]
        num_boxes_used = np.sum(np.array(max_area) > 0)
        query_frames = np.ones((self.target_size[0]*scale + hspacer, (self.target_size[1]*scale + wspacer) * num_boxes_used,3),dtype=np.uint8) * 255
        where_boxes = np.where(np.array(max_area) > 0)[0]

        for j,ind in enumerate(where_boxes):

            if self.args.dataset == 'moma':
                img_empty = cv2.imread(str(self.output_dir.parents[1] / 'examples' / 'empty_chamber.png'))
                img_empty = cv2.resize(img_empty,(self.target_size[1]*scale,self.target_size[0]*scale))
            else:
                img_empty = np.ones((self.target_size[1]*scale,self.target_size[0]*scale,3),dtype=np.uint8) * 255

            for box in self.query_box_locations[ind][1:]:
                img_empty = cv2.circle(img_empty, (int(box[0]*scale),int(box[1]*scale)), radius=1*scale, color=(255,0,0), thickness=-1)

            img_empty = np.concatenate((np.ones((hspacer,self.target_size[1]*scale,3),dtype=np.uint8)*255,img_empty),axis=0)
            shift = 5 if ind + 1 >= 10 else 12
            img_empty = cv2.putText(img_empty,f'{ind+1}',org=(shift*scale,15*scale),fontFace=cv2.FONT_HERSHEY_COMPLEX,fontScale=4,color=(0,0,0),thickness=4)
            query_frames[:,j*(self.target_size[1]*scale+wspacer): j*(self.target_size[1]*scale+wspacer) + self.target_size[1]*scale] = img_empty

        cv2.imwrite(str(self.output_dir / self.data_viz_folder / (f'{self.method}_object_query_box_locations.png')),query_frames)

        if self.use_dab:
            height,width = self.target_size[0] * scale, self.target_size[1] * scale
            boxes = outputs['aux_outputs'][0]['pred_boxes'][0,:,:4].cpu().numpy()

            boxes[:,1::2] = boxes[:,1::2] * height
            boxes[:,::2] = boxes[:,::2] * width

            boxes[:,0] = boxes[:,0] - boxes[:,2] // 2
            boxes[:,1] = boxes[:,1] - boxes[:,3] // 2

            for j,ind in enumerate(where_boxes):

                bounding_box = boxes[ind]

                query_frames = cv2.rectangle(
                query_frames,
                (int(np.clip(bounding_box[0],0,width)) + j * (width + wspacer), int(np.clip(bounding_box[1],0,height))+hspacer),
                (int(np.clip(bounding_box[0] + bounding_box[2],0,width)) + j * (width + wspacer), int(np.clip(bounding_box[1] + bounding_box[3],0,height))+hspacer),
                color=tuple(np.array([50.,50.,50.])),
                thickness = 5)

            cv2.imwrite(str(self.output_dir  / self.data_viz_folder / (f'{self.method}_object_query_box_locations_with_boxes.png')),query_frames)

    def display_two_stage(self, outputs):

        logits_topk = outputs['two_stage']['pred_logits'][0,:,0].sigmoid().cpu().numpy()
        boxes_topk = outputs['two_stage']['pred_boxes'][0].cpu().numpy()

        enc_colors = np.array([(np.array([0.,0.,0.])) for _ in range(self.num_queries)])

        # If proposed enc object query is used, we want to use the same color that the final prediction
        if self.track:
            for o,index in enumerate(self.object_indices):
                    enc_colors[index] = self.all_colors[self.cells[len(self.track_indices)+o]-1]
        else:
            enc_colors[:len(self.cells)] = self.all_colors[self.cells-1]

        thresholds = [1,0.75,0.5,0.25,0]
        boxes_list = []
        for t in range(len(thresholds)-1):
            boxes_list.append(boxes_topk[(logits_topk > thresholds[t+1]) * (logits_topk < thresholds[t])])

        num_per_box = [box.shape[0] for box in boxes_list]
        enc_frames = []
        for b,boxes in enumerate(boxes_list):
            enc_frame  = data_viz.plot_tracking_results(self.img,boxes,None,enc_colors[sum(num_per_box[:b]):sum(num_per_box[:b+1])],None,None)
            enc_frame  = cv2.putText(enc_frame, text = f'{thresholds[b+1]}-', org=(0,10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.4, color = (255,255,255), thickness=1)
            enc_frame  = cv2.putText(enc_frame, text = f'{thresholds[b]}', org=(0,20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.4, color = (255,255,255), thickness=1)
            enc_frames.append(enc_frame)

        enc_frames = np.concatenate((enc_frames),axis=1)

        cv2.imwrite(str(self.output_dir / self.data_viz_folder / 'two_stage' / (f'encoder_frame_{self.fp.stem}.png')),enc_frames)

        self.enc_colors = enc_colors

    
    def save_enc_map(self, outputs, display_proposal_index_on_img=False):

        enc_outputs = outputs['two_stage']

        proposals_img = self.img.copy()
        final_proposal_pred = self.img.copy()

        spatial_shapes = enc_outputs['spatial_shapes']
        fmaps_cum_size = torch.tensor([spatial_shape[0] * spatial_shape[1] for spatial_shape in spatial_shapes]).cumsum(0)

        if self.enc_map is None:
            self.enc_map = [np.zeros((spatial_shape[0].item(),spatial_shape[1].item())) for spatial_shape in spatial_shapes]

        topk_proposals = enc_outputs['topk_proposals'][0].cpu()
        proposals = topk_proposals[enc_outputs['pred_logits'][0,:,0].sigmoid().cpu() > 0.5].clone().cpu()

        for proposal in proposals:

            proposal_clone = proposal.clone()

            # Get the index of the feature map (multi-scale features maps --> multiple feature maps)
            f = [i for i, fmap_size in enumerate(fmaps_cum_size) if proposal < fmap_size][0]

            # Scale the index back to the correct feature map size
            if f > 0:
                proposal_clone -= fmaps_cum_size[f-1]

            # Get x,y coordinates of the proposal in the feature map; the proposal index correspond to a flattened feature map
            y = torch.floor_divide(proposal_clone,spatial_shapes[f,1])
            x = torch.remainder(proposal_clone,spatial_shapes[f,1])

            # Add the coordinate to the enc map which will be displayed at the end of the video
            self.enc_map[f][y,x] += 1

            if display_proposal_index_on_img:
            
                # Overall proposal index onto the image
                small_mask = np.zeros(spatial_shapes[f].cpu().numpy(),dtype=np.uint8)
                small_mask[y,x] += 1

                resize_mask = cv2.resize(small_mask,self.target_size[::-1])

                y, x = np.where(resize_mask == 1)

                X = np.min(x) 
                Y = np.min(y)
                width = (np.max(x) - np.min(x)) 
                height = (np.max(y) - np.min(y)) 
                
                proposal_ind = int(torch.where(topk_proposals == proposal)[0][0])

                proposals_img = cv2.rectangle(proposals_img,(X,Y),(int(X+width),int(Y+height)),color=self.enc_colors[proposal_ind],thickness=1)

                pred_box = outputs['pred_boxes'][:,proposal_ind+self.num_TQs,:4].cpu().numpy()

                final_proposal_pred = data_viz.plot_tracking_results(final_proposal_pred,pred_box,None,colors=self.enc_colors[proposal_ind][None], new_cells = [proposal_ind+self.num_TQs in self.object_indices and self.track and self.i > 0], use_new_img=False)

        if display_proposal_index_on_img:

            enc_frames = cv2.imread(str(self.output_dir / self.data_viz_folder / 'two_stage' / (f'encoder_frame_{self.fp.stem}.png')))
            enc_frames = np.concatenate((enc_frames, proposals_img,final_proposal_pred),axis=1)

            cv2.imwrite(str(self.output_dir / self.data_viz_folder / 'two_stage' / (f'encoder_frame_{self.fp.stem}.png')),enc_frames)


    def display_enc_map(self,save=True,last=False):

        spacer = 1
        enc_maps = []
        max_value = 0

        if last:
            last_enc_maps = []

        if self.all_videos_same_size:
            for e,enc_map in enumerate(self.enc_map):
                if save:
                    cum_enc_map = enc_map
                    if (self.output_dir.parent / 'two_stage' / f'enc_queries_{e:02d}.npy').exists():
                        cum_enc_map = np.load(self.output_dir.parent / 'two_stage' / f'enc_queries_{e:02d}.npy')
                        if cum_enc_map.shape != enc_map.shape:
                            self.all_videos_same_size = False
                            continue
                        cum_enc_map += enc_map

                    np.save(self.output_dir.parent / 'two_stage' / f'enc_queries_{e:02d}.npy',cum_enc_map)

                if last:
                    last_enc_maps += [np.load(self.output_dir.parent / 'two_stage' / f'enc_queries_{e:02d}.npy')]

            if last:
                self.enc_map = last_enc_maps

        for enc_map in self.enc_map:

            enc_map = cv2.resize(enc_map,(self.target_size[1],self.target_size[0]),interpolation=cv2.INTER_NEAREST)
            enc_map = np.repeat(enc_map[:,:,None],3,-1)
            max_value = max(max_value,np.max(enc_map))
            enc_maps.append(enc_map)
            border = np.zeros((self.target_size[0],spacer,3),dtype=np.uint8)
            border[:,:,0] = -1
            enc_maps.append(border)

        enc_maps = np.concatenate((enc_maps),axis=1)

        enc_maps[enc_maps!=-1] = (enc_maps[enc_maps!=-1] / max_value) * 255
        enc_maps[enc_maps==-1] = 255
        enc_maps = enc_maps[:,:-spacer]

        enc_maps = np.concatenate((enc_maps[:,self.target_size[1]:self.target_size[1]+spacer],enc_maps),axis=1)

        if not last:
            cv2.imwrite(str(self.output_dir / self.data_viz_folder / 'enc_queries_picked.png'),enc_maps.astype(np.uint8))
        else:
            cv2.imwrite(str(self.output_dir.parent / 'two_stage' / 'all_enc_queries_picked.png'),enc_maps.astype(np.uint8))

            nb_plots = len(self.enc_map)

            fig, axs = plt.subplots(1, nb_plots, figsize=(18, 6))

            max_value = max(map.max() for map in self.enc_map)

            for pidx, map in enumerate(self.enc_map):

                # Plotting the multi-scale features
                cbar0 = axs[pidx].imshow(map, cmap='viridis', vmin=0, vmax=max_value)
                axs[pidx].set_title(f'Multi-Scale Feature {pidx:02d}')

                cbar = axs[pidx].imshow(map, cmap='viridis', vmin=0, vmax=max_value)
                axs[pidx].set_title('Multi-Scale Feature 00')
                axs[pidx].grid(True, which='both', color='black', linewidth=0.5)
                axs[pidx].xaxis.set_major_locator(plt.MultipleLocator(1))
                axs[pidx].yaxis.set_major_locator(plt.MultipleLocator(1))
                axs[pidx].set_xlim(0, map.shape[1] - 1)
                axs[pidx].set_ylim(0, map.shape[0] - 1)
                axs[pidx].xaxis.set_major_locator(plt.MaxNLocator(10))
                axs[pidx].yaxis.set_major_locator(plt.MaxNLocator(10))

            # Adjust the spacing between plots
            plt.subplots_adjust(wspace=0.1)

            # Adding a single color bar to the right of the plots
            fig.subplots_adjust(right=0.85)
            cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
            fig.colorbar(cbar0, cax=cbar_ax)

            fig.savefig(self.output_dir.parent / 'two_stage' / 'heat_map.pdf')


    def display_attn_maps(self,dec_attn_outputs):

        for layer_index, dec_attn_output in enumerate(dec_attn_outputs):
            dec_attn_weight_maps = dec_attn_output[1].cpu().numpy() # [output, attention_map]
            num_cols, num_rows = dec_attn_weight_maps.shape[-2:]
            num_heads = 1 if dec_attn_weight_maps.ndim == 3 else dec_attn_weight_maps.shape[1]
            dec_attn_weight_maps = np.repeat(dec_attn_weight_maps[...,None],3,axis=-1)
            for dec_attn_weight_map in dec_attn_weight_maps: # per batch
                for h in range(num_heads):
                    if num_heads > 1:
                        dec_attn_weight_map_h = dec_attn_weight_map[h]
                    else:
                        h = 'averaged'
                        dec_attn_weight_map_h = dec_attn_weight_map
                    dec_attn_weight_map_color = np.zeros((num_cols+1,num_rows+1,3))
                    dec_attn_weight_map_color[-num_cols:,-num_rows:] = dec_attn_weight_map_h
                    # dec_attn_weight_map_color = ((dec_attn_weight_map_color / np.max(dec_attn_weight_map_color)) * 255).astype(np.uint8)
                    dec_attn_weight_map_color = (dec_attn_weight_map_color * 255).astype(np.uint8)

                    # Go through track queries
                    for tidx in range(self.num_TQs):
                        dec_attn_weight_map_color[tidx+1,0] = self.all_colors[self.prevcells[tidx]-1]
                        dec_attn_weight_map_color[0,tidx+1] = self.all_colors[self.prevcells[tidx]-1]                                

                    # color Track queries only
                    if False:
                        cv2.imwrite(str(self.output_dir / 'attn_weight_maps' / (f'self_attn_weight_map_{self.fp.stem}_layer_{layer_index}_head_{h}.png')),dec_attn_weight_map_color_resize)

                    # Go through object queries
                    for q in range(self.num_TQs, self.num_TQs + self.num_queries):
                        dec_attn_weight_map_color[q+1,0] = self.all_colors[-q]
                        dec_attn_weight_map_color[0,q+1] = self.all_colors[-q]
                    
                    cv2.imwrite(str(self.output_dir / 'attn_weight_maps' / (f'self_attn_map_{self.fp.stem}_layer_{layer_index}_head_{h}.png')),dec_attn_weight_map_color)

                if num_heads > 1:
                    dec_attn_weight_map_avg = (dec_attn_weight_map.mean(0) * 255).astype(np.uint8)

                    dec_attn_weight_map_avg_color = dec_attn_weight_map_color
                    dec_attn_weight_map_avg_color[-num_cols:,-num_rows:] = dec_attn_weight_map_avg

                    cv2.imwrite(str(self.output_dir  / self.data_viz_folder / 'attn_weight_maps' / (f'self_attn_map_{self.fp.stem}_layer_{layer_index}_head_averaged.png')),dec_attn_weight_map_avg_color)


    def display_aux_preds(self,outputs):

        references = outputs['references']

        if references.shape[-1] == 2:
            references = torch.cat((references, torch.zeros(references.shape[:-1] + (6,),device=references.device)),axis=-1)

        # the initial anchors / reference points + output from the intermedaite layers of decoder
        ref_boxes = references[0]
        ref_logits = outputs['two_stage']['pred_logits']

        if len(self.track_indices) > 0:
            self.decoder_frame = np.concatenate((self.previmg,self.img),axis=1)
            self.get_track_preds(ref_boxes,None,ref_boxes=True)
            self.get_object_preds(ref_logits,ref_boxes,text='QS',use_logits=True)
        else:
            self.decoder_frame = self.img
            self.get_object_preds(ref_logits, ref_boxes,first_decoder_layer=True, text='QS')

        if self.args.num_OD_layers:
            OD_boxes = outputs['OD']['pred_boxes']
            OD_logits = outputs['OD']['pred_logits']
            OD_masks = None
            if 'pred_masks' in outputs['OD']:
                OD_masks = outputs['OD']['pred_masks']
            self.get_object_preds(OD_logits,OD_boxes,OD_masks, text='OD 1',use_logits=True)

        # add the last layer of the decoder which is the final prediction
        aux_outputs = outputs['aux_outputs']
        aux_outputs.append({'pred_boxes':outputs['pred_boxes'],'pred_logits':outputs['pred_logits']}) 
        
        # Add segmentation masks 
        if 'pred_masks' in outputs:
            aux_outputs[-1]['pred_masks'] = outputs['pred_masks'] # add mask

        for a,aux_output in enumerate(aux_outputs):
            aux_boxes = aux_output['pred_boxes']
            aux_logits = aux_output['pred_logits']

            aux_masks = None
            if self.masks and self.return_intermediate_masks:
                aux_masks = aux_output['pred_masks']

            last_decoder_layer = a == len(aux_outputs) - 1

            if len(self.track_indices) > 0:
                self.get_track_preds(aux_boxes,aux_masks,last_decoder_layer=last_decoder_layer, text=f'Tr {a+1+self.args.num_OD_layers}')

                if self.args.CoMOT or last_decoder_layer:
                    use_logits = self.args.CoMOT_loss_ce
                    self.get_object_preds(aux_logits,aux_boxes,aux_masks, CoMOT=self.args.CoMOT,text=f'OD {a+1+self.args.num_OD_layers}',use_logits=use_logits)                    

            else:
                self.get_object_preds(aux_logits,aux_boxes,aux_masks,last_decoder_layer=last_decoder_layer, text=f'OD {a+1+self.args.num_OD_layers}')

        cv2.imwrite(str(self.output_dir / self.data_viz_folder / 'decoder_bbox_outputs' / (f'{self.method}_decoder_frame_{self.fp.stem}.png')),self.decoder_frame)

    def get_object_preds(self,logits,boxes,masks=None,first_decoder_layer=False,last_decoder_layer=False, CoMOT=False, text='', use_logits=False):

        object_boxes = boxes[0,-self.num_queries:].cpu().numpy()
        if use_logits:
            object_indices = np.where(logits[0,-self.num_queries:,0].cpu().numpy() > self.threshold)[0]
        else:
            if CoMOT:
                object_indices = np.arange(len(self.cells)) # we assume the cell detected from QS are continually used to detect the cells #np.where(object_logits > self.threshold)[0]
            else:
                object_indices = self.object_indices
        pred_object_boxes = object_boxes[object_indices]
        object_colors = np.zeros((len(pred_object_boxes),3))

        pred_object_masks = None
        if masks is not None and (self.return_intermediate_masks or last_decoder_layer) and len(object_indices) > 0:
            object_masks = masks[0,-self.num_queries:,0]
            pred_object_masks = object_masks[object_indices].cpu().sigmoid().numpy()
            pred_object_masks = cv2.resize(np.transpose(pred_object_masks,(1,2,0)),self.img_size) # regardless of cropping / resize, segmentation is 2x smaller than the original image                                
            pred_object_masks = pred_object_masks[:,:,None] if pred_object_masks.ndim == 2 else pred_object_masks # cv2.resize will drop last dim if it is 1
            pred_object_masks = np.transpose(pred_object_masks,(-1,0,1))
            pred_object_masks = np.where(pred_object_masks > self.mask_threshold,True,False)

        for i,object_index in enumerate(object_indices):
            if object_index + self.num_TQs in self.object_indices:
                cell_index = np.where(self.all_indices == object_index + self.num_TQs)[0]
                cell = self.cells[cell_index]
                object_colors[i] = self.all_colors[cell-1]
            else:
                ind = object_index + self.num_TQs
                object_colors[i] = self.all_colors[-ind]

        if first_decoder_layer:
            all_object_colors = np.concatenate((np.array([[0,0,0] for _ in range(self.num_queries - len(object_indices))]),object_colors))
            non_object_indices = np.array([i for i in range(self.num_queries) if i not in object_indices])
            object_all_boxes = np.concatenate((object_boxes[non_object_indices],pred_object_boxes))
            all_objects_no_label_colors = np.array([[0,0,0] for _ in range(self.num_queries)],dtype=float)
            all_objects_no_label_frame = data_viz.plot_tracking_results(self.img,object_all_boxes,None,all_objects_no_label_colors,None,None)
            all_objects_no_label_frame = cv2.putText(all_objects_no_label_frame,text = text, org=(0,10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.4, color = (255,255,255), thickness=1)
            all_objects_frame = data_viz.plot_tracking_results(self.img,object_all_boxes,None,all_object_colors,None,None)
            all_objects_frame = cv2.putText(all_objects_frame,text = text, org=(0,10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.4, color = (255,255,255), thickness=1)
            self.decoder_frame = np.concatenate((self.decoder_frame,all_objects_no_label_frame,all_objects_frame),axis=1)

        pred_objects_frame = data_viz.plot_tracking_results(self.img,pred_object_boxes,pred_object_masks,object_colors,None,None)
        pred_objects_frame = cv2.putText(pred_objects_frame,text = text, org=(0,10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.4, color = (255,255,255), thickness=1)
        self.decoder_frame = np.concatenate((self.decoder_frame,pred_objects_frame),axis=1)

        if last_decoder_layer:
            img_box = data_viz.plot_tracking_results(self.img,pred_object_boxes,None,object_colors)
            img_box = cv2.putText(img_box,text = text, org=(0,10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.4, color = (255,255,255), thickness=1)
            self.decoder_frame = np.concatenate((self.decoder_frame,img_box),axis=1)
            if self.masks:
                img_mask = data_viz.plot_tracking_results(self.img,None,pred_object_masks,object_colors)
                img_mask = cv2.putText(img_mask,text = text, org=(0,10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.4, color = (255,255,255), thickness=1)
                self.decoder_frame = np.concatenate((self.decoder_frame,img_mask),axis=1)

    def get_track_preds(self,boxes,masks=None,ref_boxes=False,last_decoder_layer=False, text=''):
            
        pred_all_masks = None
        new_cells = None
        div_track = None

        if ref_boxes:
            pred_all_boxes = boxes.cpu().numpy()[0,np.arange(self.num_TQs)]
            all_colors = self.all_colors[self.prevcells-1]
            img = self.previmg
        else:
            pred_all_boxes = boxes[0,self.all_indices].cpu().numpy()
            all_colors = self.all_colors[self.cells-1]
            img = self.img

            if masks is not None and (self.return_intermediate_masks or last_decoder_layer) and len(self.all_indices) > 0:
                pred_all_masks = masks[0,self.all_indices].cpu().sigmoid().numpy()

                unique_divs = np.unique(self.div_track[self.div_track != -1])
                for unique_div in unique_divs:
                    div_ids = (self.div_track == unique_div).nonzero()[0]
                    pred_all_boxes[div_ids[1],:4] = pred_all_boxes[div_ids[0],4:]
                    if self.masks and (self.return_intermediate_masks or last_decoder_layer) and len(self.all_indices) > 0:
                        pred_all_masks[div_ids[1],:1] = pred_all_masks[div_ids[0],1:]

                pred_all_boxes = pred_all_boxes[:,:4]

                if self.masks and (self.return_intermediate_masks or self.last_decoder_layer) and len(self.all_indices) > 0:
                    pred_all_masks = cv2.resize(np.transpose(pred_all_masks[:,0],(1,2,0)),self.img_size) # regardless of cropping / resize, segmentation is 2x smaller than the original image                                
                    pred_all_masks = pred_all_masks[:,:,None] if pred_all_masks.ndim == 2 else pred_all_masks # cv2.resize will drop last dim if it is 1
                    pred_all_masks = np.transpose(pred_all_masks,(-1,0,1))
                    pred_all_masks = np.where(pred_all_masks > self.mask_threshold,True,False)

                div_track = self.div_track
                new_cells = self.new_cells

        pred_objects_frame = data_viz.plot_tracking_results(img,pred_all_boxes,pred_all_masks,all_colors,div_track,new_cells)
        pred_objects_frame = cv2.putText(pred_objects_frame,text = text, org=(0,10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.4, color = (255,255,255), thickness=1)
        self.decoder_frame = np.concatenate((self.decoder_frame,pred_objects_frame),axis=1)

        if last_decoder_layer:
            img_box = data_viz.plot_tracking_results(self.img,pred_all_boxes,None,all_colors,div_track)
            img_box = cv2.putText(img_box,text = text, org=(0,10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.4, color = (255,255,255), thickness=1)
            self.decoder_frame = np.concatenate((self.decoder_frame,img_box),axis=1)
            if self.masks:
                img_mask = data_viz.plot_tracking_results(self.img,None,pred_all_masks,all_colors,div_track)
                img_mask = cv2.putText(img_mask,text = text, org=(0,10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.4, color = (255,255,255), thickness=1)
                self.decoder_frame = np.concatenate((self.decoder_frame,img_mask),axis=1)

            all_boxes = boxes.cpu().numpy()[0,:,:4]
        
            # color_queries = np.array([(np.array([0.,0.,0.])) for _ in range(pred_boxes.shape[0])])
            # color_queries[self.all_indices] = self.all_colors[self.cells-1]
            color_queries = self.all_colors[self.cells-1]
            # color_queries = color_queries[::-1]

            div_track = self.div_track

            # Get cells that exit the frame --> they get highlighted so it's easier to spot errors
            cell_exit_ids = [[cidx,c] for cidx,c in enumerate(self.prevcells.astype(np.int64)) if c not in self.cells]

            if len(cell_exit_ids) > 0:

                for cell_exit_ind, cell_exit_id in cell_exit_ids:
                    # color_queries[cell_exit_ind] = self.all_colors[cell_exit_id-1]

                    color_queries = np.concatenate((self.all_colors[cell_exit_id-1][None],color_queries))
                    div_track = np.concatenate((np.array([-1]),div_track))
                    pred_all_boxes = np.concatenate((all_boxes[cell_exit_ind,:4][None],pred_all_boxes))

                # pred_boxes = pred_boxes[color_queries.sum(1) > 0]
                # div_track = div_track[color_queries.sum(1) > 0]
                # color_queries = color_queries[color_queries.sum(1) > 0]

                pred_all_boxes_and_exit_cells_frame = data_viz.plot_tracking_results(self.img,pred_all_boxes,None,color_queries,div_track)
                pred_all_boxes_and_exit_cells_frame = cv2.putText(pred_all_boxes_and_exit_cells_frame,text = 'Exit', org=(0,10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.4, color = (255,255,255), thickness=1)
                
                self.decoder_frame = np.concatenate((self.decoder_frame,pred_all_boxes_and_exit_cells_frame),axis=1)


    def save_video(self):

        crf = 20
        verbose = 1

        filename = self.output_dir / (f'movie_full.mp4') 

        assert self.color_stack.max() <= 255.0 and self.color_stack.max() >= 0.0               

        print(filename)
        height, width, _ = self.color_stack[0].shape
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
        for frame in self.color_stack:
            process.stdin.write(frame[:height, :width].astype(np.uint8).tobytes())

        # Close file stream:
        process.stdin.close()

        # Wait for processing + close to complete:
        process.wait()