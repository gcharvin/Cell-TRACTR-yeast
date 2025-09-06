#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
import traceback
from pathlib import Path

import cv2
import numpy as np
import tifffile as tiff
from tqdm import tqdm
from skimage.measure import label  # utilisé par create_anno (multi-composantes)
import utils_coco as utils


# --------- I/O util ---------
def read_u16(path):
    """Lit un TIFF et retourne un np.uint16 (clip/convert au besoin)."""
    try:
        arr = tiff.imread(str(path))
    except Exception as e:
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise RuntimeError(
                f"Impossible de lire {path} (compression/codec ?). "
                f"Installe imagecodecs ou utilise un autre lecteur. Origine: {e}"
            )

    if arr.dtype == np.uint16:
        return arr

    old_dtype = arr.dtype
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.nan_to_num(arr, nan=0.0, posinf=65535.0, neginf=0.0)
    arr = np.clip(arr, 0, 65535).astype(np.uint16, copy=False)
    print(f"[FIX] {path}: {old_dtype} -> uint16")
    return arr


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True,
                   help="Nom du dataset (ex: moma)")
    p.add_argument("--datapath", type=str, required=True,
                   help="Chemin vers le dossier parent qui contient <dataset>/(CTC|COCO)")
    p.add_argument("--check_only", action="store_true",
                   help="Ne fait que checker CTC (pas d'export COCO).")
    p.add_argument("--debug_seq", type=str, default=None,
                   help="Ex: 01 — n'afficher les debugs que pour cette séquence")
    p.add_argument("--debug_frame", type=int, default=None,
                   help="Ex: 125 — n'afficher les debugs que pour ce frame")
    p.add_argument("--echo_gt", action="store_true",
                   help="Log si un GT (SEG) est vide (max==0).")
    p.add_argument("--strict_mismatch", action="store_true",
                   help="Si activé, on lève une erreur en cas de mismatch IDs masque <-> man_track.")
    return p.parse_args()


def main():
    args = parse_args()

    dataset = args.dataset
    root = Path(args.datapath) / dataset
    ctc_dir = root / "CTC"
    coco_dir = root / "COCO"

    if dataset == "moma":
        min_area = 0
        target_size = (256, 32)
        resize = True
    else:
        min_area = 0
        target_size = None
        resize = False

    if min_area > 0:
        raise NotImplementedError("min_area>0 non géré ici.")

    licenses = [{'id': 1, 'name': 'MIT license'}]
    categories = [{'id': 1, 'name': 'cell'}]
    info = utils.get_info(dataset)

    coco_dir.mkdir(exist_ok=True, parents=True)
    folders = ["train", "val"]
    utils.create_folders(coco_dir, folders)

    img_reader = utils.reader(dataset=dataset, target_size=target_size, resize=resize, min_area=min_area)

    train_sets = sorted([x for x in (ctc_dir / "train").iterdir()
                         if x.is_dir() and re.findall(r"\d\d$", x.name)])
    val_sets = sorted([x for x in (ctc_dir / "val").iterdir()
                       if x.is_dir() and re.findall(r"\d\d$", x.name)])

    for split_name, dataset_paths in zip(folders, [train_sets, val_sets]):
        image_id = 0
        annotation_id = 0
        images = []
        annotations = []
        skipped = 0
        max_num_of_cells = 0

        for dataset_path in dataset_paths:
            try:
                fps = sorted(dataset_path.glob("*.tif"))
                if len(fps) < 4:
                    skipped += 1
                    continue

                dataset_name = dataset_path.name
                gt_dir = dataset_path.parent / f"{dataset_name}_GT"
                man_track_txt = gt_dir / "TRA" / "man_track.txt"

                if not man_track_txt.exists():
                    print(f"[WARN] man_track.txt manquant pour {dataset_name} -> skip")
                    skipped += 1
                    continue

                # Lecture du man_track.txt
                track_file = []
                with open(man_track_txt, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        parts = line.split()
                        try:
                            row = [int(i) for i in parts]
                        except Exception:
                            continue
                        if len(row) >= 4:
                            track_file.append(row[:4])
                track_file = np.array(track_file, dtype=int) if len(track_file) else np.zeros((0, 4), dtype=int)

                if len(fps) > 0 and track_file.size > 0:
                    first_nb = int(re.findall(r"\d+", fps[0].name)[-1])
                    last_nb = int(re.findall(r"\d+", fps[-1].name)[-1])
                    print(f"[CHECK] {dataset_name}: file_frames=[{first_nb}..{last_nb}] "
                          f"man_track=[{track_file[:,1].min()}..{track_file[:,2].max()}]")

                img_reader.load_track_file(track_file)
                try:
                    img_reader.read_gts(fps)
                except Exception as e:
                    print(f"[WARN] read_gts a échoué pour {dataset_name}: {e}")

                for counter, fp in enumerate(tqdm(fps, desc=f"{split_name}:{dataset_name}")):
                    try:
                        framenb = int(re.findall(r"\d+", fp.name)[-1])
                        img = img_reader.read_image(fp)
                        H, W = img.shape[:2]

                        seg_gt = None
                        try:
                            seg_gt = img_reader.read_gt(fp, counter)
                        except Exception as e:
                            print(f"[WARN] read_gt échec {dataset_name}/frame {framenb}: {e}")

                        if args.echo_gt and (seg_gt is None or (hasattr(seg_gt, "max") and seg_gt.max() == 0)):
                            print(f"[ECHO] EMPTY SEG GT: {fp}")

                        # fallback sur TRA si seg_gt manquant
                        if seg_gt is None:
                            tra_path = gt_dir / "TRA" / f"man_track{framenb:03d}.tif"
                            if tra_path.exists():
                                seg_gt = read_u16(tra_path)
                            else:
                                print(f"[WARN] GT introuvable -> skip frame: {tra_path}")
                                continue

                        tra = seg_gt
                        if (tra.shape[0], tra.shape[1]) != (H, W):
                            tra = cv2.resize(tra, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.uint16)

                        cellnbs = np.unique(tra)
                        cellnbs = cellnbs[cellnbs != 0].astype(int)
                        if len(cellnbs) > 0:
                            max_num_of_cells = max(max_num_of_cells, int(len(cellnbs)))

                        if len(cellnbs) > 0:
                            for cellnb in cellnbs:
                                try:
                                    _ = label(tra == cellnb)
                                    annotation = utils.create_anno(tra, int(cellnb), image_id, annotation_id, dataset_name)
                                    annotation["ctc_id"]  = int(ctc_id_int) if ctc_id_int is not None else None
                                    annotation["frame_id"] = int(framenb)
                                    annotations.append(annotation)
                                    annotation_id += 1
                                except Exception as e:
                                    print(f"[WARN] {dataset_name}/frame {framenb}: anno ratée pour cell {int(cellnb)} ({e})")
                                    continue

                        fn = f"CTC_{dataset_name}_frame_{framenb:03d}.tif"
                       # seq comme entier (01 -> 1)
                        try:
                            ctc_id_int = int(re.findall(r"\d+", dataset_name)[-1])
                        except Exception:
                            ctc_id_int = None  # on garde une roue de secours si jamais

                        images.append({
                            "license": 1,
                            "man_track_id": dataset_name,
                            "file_name": fn,
                            "height": int(H),
                            "width": int(W),
                            "id": int(image_id),
                            "ctc_id": int(ctc_id_int) if ctc_id_int is not None else None,  # <<< INT, pas str
                            "frame_id": int(framenb),                                       # <<< INT
                            "seq_length": int(len(fps)),
                        })
                        image_id += 1

                        if not args.check_only:
                            (coco_dir / split_name / "img").mkdir(parents=True, exist_ok=True)
                            (coco_dir / split_name / "gt").mkdir(parents=True, exist_ok=True)
                            if not cv2.imwrite(str(coco_dir / split_name / "img" / fn), img):
                                print(f"[ERR] write failed (img): {fn}")
                            if not cv2.imwrite(str(coco_dir / split_name / "gt" / fn), tra):
                                print(f"[ERR] write failed (gt):  {fn}")

                    except Exception as e:
                        frame_str = "??"
                        try:
                            frame_str = re.findall(r"\d+", fp.name)[-1] if fp else "??"
                        except Exception:
                            pass
                        print(f"[ERR] {dataset_name}/frame {frame_str}: {e}")
                        continue

                if not args.check_only:
                    try:
                        dest = (coco_dir / "man_track" / split_name)
                        dest.mkdir(parents=True, exist_ok=True)
                        np.savetxt(dest / f"{dataset_name}.txt", track_file, fmt="%d")
                    except Exception as e:
                        print(f"[WARN] sauvegarde man_track échouée pour {dataset_name}: {e}")

            except Exception as e:
                print(f"[SEQ-ERR] séquence {dataset_path.name}: {e}")
                print(traceback.format_exc())
                continue

        metadata = {
            "annotations": annotations,
            "images": images,
            "categories": categories,
            "licenses": licenses,
            "info": info,
            "sequences": "cells",
            "max_num_of_cells": int(max_num_of_cells),
        }

        if args.check_only:
            print(f"[CHECK-ONLY] Pas d'écriture COCO pour split={split_name}.")
        else:
            out_dir = coco_dir / "annotations" / split_name
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "anno.json", "w") as f:
                json.dump(metadata, f, cls=utils.NpEncoder)
            print(f"[OK] {split_name}: Max number of cells = {metadata['max_num_of_cells']}")

        print(f"[DONE] {split_name}: {skipped:03d} sequences skipped.")
        
    # après la boucle split, en plus du anno.json
    with open(out_dir / "index_map.json", "w") as f:
        json.dump({img["id"]: {"ctc_id": img["ctc_id"], "frame_id": img["frame_id"], "file_name": img["file_name"]}
                for img in images}, f)

    print("[FINISHED] Conversion CTC -> COCO terminée.")


if __name__ == "__main__":
    main()
