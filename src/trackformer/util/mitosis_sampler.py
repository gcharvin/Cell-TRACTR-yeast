# -*- coding: utf-8 -*-
from __future__ import annotations
import os, re
from collections import defaultdict
from typing import Sequence, Optional, Tuple, Dict, List, Set

import numpy as np
from torch.utils.data import BatchSampler

# -------------------- helpers (inchangés / améliorés) --------------------
# Motif EXACT de ton export: "CTC_<seq>_frame_<fr>.tif"
_RX_CTC = re.compile(r"CTC_(?P<seq>\d+)_frame_(?P<fr>\d+)\.(?:tif|tiff|png|jpg|jpeg)$", re.IGNORECASE)

def _parse_seq_frame_from_filename(file_name: str):
    m = _RX_CTC.search(file_name)
    if m:
        return m.group("seq"), int(m.group("fr"))
    return None, None

def _build_seq_index(dataset) -> Tuple[Dict[str, Dict[int, int]], Dict[int, Tuple[str, int]]]:
    """
    Construit:
      - seq2frames: {"01": {0: idx, 1: idx, ...}}
      - idx2meta : {idx: ("01", frame)}
    en lisant **file_name** via dataset.coco/dataset.ids.
    """
    seq2frames: Dict[str, Dict[int, int]] = defaultdict(dict)
    idx2meta: Dict[int, Tuple[str, int]] = {}

    N = len(dataset)

    # Chemin COCO standard : dataset.coco + dataset.ids
    has_coco = hasattr(dataset, "coco") and hasattr(dataset, "ids") and len(getattr(dataset, "ids")) == N
    if not has_coco:
        # Dernier recours: essayer dataset.images[i]["file_name"] / dataset.imgs[i]["file_name"]
        for i in range(N):
            name = None
            for attr in ("images", "imgs", "file_paths"):
                if hasattr(dataset, attr):
                    src = getattr(dataset, attr)[i]
                    name = src["file_name"] if isinstance(src, dict) and "file_name" in src else (src if isinstance(src, str) else None)
                    if name: break
            if not name: 
                # essaie via target
                try:
                    tgt = dataset.get_target(i) if hasattr(dataset, "get_target") else dataset[i][1]
                    if isinstance(tgt, dict) and "file_name" in tgt:
                        name = tgt["file_name"]
                except Exception:
                    pass
            if not name:
                continue
            seq, fr = _parse_seq_frame_from_filename(os.path.basename(name))
            if seq is None or fr is None:
                continue
            seq2frames[seq][fr] = i
            idx2meta[i] = (seq, fr)
        return seq2frames, idx2meta

    # Chemin principal (pycocotools-like)
    coco = dataset.coco
    ids = list(dataset.ids)  # ids = image ids dans l'annotation COCO
    for i, img_id in enumerate(ids):
        # Selon l’implémentation, l’accès peut être via coco.imgs[...] ou coco.loadImgs(...)
        info = coco.imgs.get(img_id) if hasattr(coco, "imgs") else None
        if info is None:
            loaded = coco.loadImgs([img_id])
            info = loaded[0] if loaded else None
        if not info:
            continue

        fn = info.get("file_name") if isinstance(info, dict) else None
        if not fn:
            continue

        seq, fr = _parse_seq_frame_from_filename(fn)
        if seq is None or fr is None:
            continue

        seq2frames[seq][fr] = i
        idx2meta[i] = (seq, fr)

    return seq2frames, idx2meta

def _derive_ctc_root(coco_root: Optional[str]) -> Optional[str]:
    if not coco_root: return None
    path = os.path.normpath(coco_root)
    parts = path.split(os.sep)
    if parts and parts[-1].lower() == "coco":
        parts[-1] = "CTC"
        return os.sep.join(parts)
    return os.path.join(os.path.dirname(path), "CTC")


def _seq_variants(seq: str) -> List[str]:
    """Retourne des variantes plausibles du code séquence (avec/ sans zéros)."""
    out = {seq}
    try:
        n = int(seq)
        out.add(str(n))           # "01" -> "1"
        out.add(f"{n:02d}")       # "1"  -> "01"
        out.add(f"{n:03d}")       # "1"  -> "001"
    except Exception:
        pass
    return list(out)

def _find_mantrack_in_dir(dirpath: str, seq: str) -> Optional[str]:
    """
    Dans dirpath (= .../COCO/man_track/train), cherche un fichier *.txt dont
    le nom sans extension représente le *même entier* que `seq`.
    """
    if not os.path.isdir(dirpath):
        return None
    try:
        seq_num = int(seq)
    except Exception:
        seq_num = None

    # 1) tentatives directes avec variantes "1","01","001"
    for cand in _seq_variants(seq):
        p = os.path.join(dirpath, f"{cand}.txt")
        if os.path.exists(p):
            return p

    # 2) sinon, scan complet et match numérique
    if seq_num is not None:
        for fn in os.listdir(dirpath):
            if not fn.lower().endswith(".txt"):
                continue
            stem = os.path.splitext(fn)[0]
            try:
                if int(stem) == seq_num:
                    p = os.path.join(dirpath, fn)
                    if os.path.exists(p):
                        return p
            except Exception:
                continue
    return None

def _guess_mantrack_path(coco_root: Optional[str], split_dir: str, seq: str) -> Optional[str]:
    """
    Priorité: .../COCO/man_track/<split>/<seq>.txt (ta structure)
    Fallbacks historiques pour compat :
      - .../COCO/<split>/<seq>[_GT]/TRA/man_track.txt
      - .../CTC/<split>/<seq>[_GT]/TRA/man_track.txt
    """
    if not coco_root:
        coco_root = ""
    # 1) Ta structure export COCO
    p = os.path.join(coco_root, "man_track", split_dir, f"{int(seq):02d}.txt")
    if os.path.exists(p):
        return p
    # tenter variantes sans/pourcent de zéros
    for cand in (f"{int(seq)}.txt", f"{int(seq):03d}.txt"):
        p2 = os.path.join(coco_root, "man_track", split_dir, cand)
        if os.path.exists(p2):
            return p2

    # 2) Fallbacks dans COCO
    for cand in (
        os.path.join(coco_root, split_dir, f"{int(seq):02d}_GT", "TRA", "man_track.txt"),
        os.path.join(coco_root, split_dir, f"{int(seq):02d}",     "TRA", "man_track.txt"),
    ):
        if os.path.exists(cand):
            return cand

    # 3) Fallbacks dans CTC à côté de COCO
    ctc_root = os.path.join(os.path.dirname(coco_root), "CTC")
    for cand in (
        os.path.join(ctc_root, split_dir, f"{int(seq):02d}_GT", "TRA", "man_track.txt"),
        os.path.join(ctc_root, split_dir, f"{int(seq):02d}",     "TRA", "man_track.txt"),
    ):
        if os.path.exists(cand):
            return cand

    return None



def _compute_div_frames_from_mantrack(man_track_txt: str) -> Tuple[Set[int], int]:
    """
    Division = >=2 starts le même frame (par 'starts' ou par (parent,start)).
    Centre fixé à t = start-1 (filles démarrent à t+1).
    """
    if not (man_track_txt and os.path.exists(man_track_txt)):
        return set(), 0
    starts_count = defaultdict(int)
    parent_start_count = defaultdict(int)
    max_frame_seen = -1
    with open(man_track_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = line.split()
            if len(parts) < 4: continue
            start = int(parts[1]); end = int(parts[2]); parent = int(parts[3])
            starts_count[start] += 1
            parent_start_count[(parent, start)] += 1
            if end   > max_frame_seen:   max_frame_seen = end
            if start > max_frame_seen:   max_frame_seen = start
    pos_from_starts = {s-1 for s,c in starts_count.items() if c>=2 and s-1>=0}
    pos_from_parent = {s-1 for (p,s),c in parent_start_count.items() if c>=2 and s-1>=0}
    pos_frames = pos_from_parent if len(pos_from_parent) >= len(pos_from_starts) else pos_from_starts
    return pos_frames, (max_frame_seen+1)

# -------------------- scan mitosis → POS/NEG centres --------------------

def scan_mitosis(dataset, coco_root: Optional[str] = None, split_dir: str = "train"
) -> Tuple[List[int], List[int], Dict[str, Dict[int,int]]]:
    """
    Renvoie des indices *centres* POS/NEG (un centre = un frame t),
    déterminés par les man_track quand les flags n'existent pas dans les targets.
    """
    N = len(dataset)
    seq2frames, idx2meta = _build_seq_index(dataset)
    
    # 1) via targets si dispo
    flags = [False] * N
    found_from_targets = False
    for i in range(N):
        try:
            tgt = dataset.get_target(i) if hasattr(dataset, "get_target") else dataset[i][1]
            if isinstance(tgt, dict):
                if "is_div" in tgt:      flags[i] = bool(tgt["is_div"]);   found_from_targets = True
                elif "div_flag" in tgt:  flags[i] = bool(tgt["div_flag"]); found_from_targets = True
        except Exception:
            pass
    pos_idx = [i for i,f in enumerate(flags) if f]
    neg_idx = [i for i,f in enumerate(flags) if not f]
    print(f"[MITOSIS-PREFLIGHT] pos={len(pos_idx)}  neg={len(neg_idx)}  total={N}")

    # 2) fallback man_track
    if (not found_from_targets) or (len(pos_idx) == 0):
        print("[MITOSIS-PREFLIGHT][WARN] Aucune division détectée via extract_div_flag().")
        if coco_root is None:
            coco_root = getattr(dataset, "coco_root", None) or getattr(dataset, "root", None) or ""
        flags2 = [False]*N
        used_any = False
        for seq, fr2idx in seq2frames.items():
            if not fr2idx: continue
            man_p = _guess_mantrack_path(coco_root, split_dir, seq)
            if not man_p:  continue
            used_any = True
            pos_frames, _ = _compute_div_frames_from_mantrack(man_p)
            for fr, idx in fr2idx.items():
                if fr in pos_frames:
                    flags2[idx] = True
        pos_idx = [i for i,f in enumerate(flags2) if f]
        neg_idx = [i for i,f in enumerate(flags2) if not f]
        print(f"[MITOSIS-PREFLIGHT][FALLBACK man_track] pos={len(pos_idx)}  neg={len(neg_idx)}  total={N}")

    return pos_idx, neg_idx, seq2frames

# -------------------- Sampler qui émet des TRIPLETS contigus --------------------

class TripletBatchSampler(BatchSampler):
    """
    Émet des indices *plats* par TRIPLETS contigus: [..., t-1, t, t+1,  t-1, t, t+1, ...]
    -> Compatible avec la collate_fn existante.

    Contraintes:
    - batch_size doit être multiple de 3 (chaque groupe = 3 frames).
    - require_triplets=True => on ne garde que les centres ayant (t-1,t,t+1).
    - pos_per_batch = nb de *centres POS* par batch (et non nb de frames POS).
    """
    def __init__(
        self,
        dataset,
        pos_centers: Sequence[int],
        neg_centers: Sequence[int],
        batch_size: int,
        pos_per_batch: int,
        seq2frames: Optional[Dict[str, Dict[int,int]]] = None,
        shuffle: bool = True,
        seed: int = 42,
        require_triplets: bool = True,
    ):
        if batch_size % 3 != 0:
            raise ValueError("TripletBatchSampler: batch_size doit être multiple de 3 (ex: 6, 12, 24...).")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.centers_per_batch = self.batch_size // 3
        self.pos_per_batch = min(int(pos_per_batch), self.centers_per_batch)
        self.neg_per_batch = self.centers_per_batch - self.pos_per_batch

        self.shuffle = shuffle
        self.rng = np.random.RandomState(int(seed))

        self.seq2frames, self.idx2meta = _build_seq_index(dataset) if seq2frames is None else (seq2frames, None)
        valid_centers = self._valid_triplet_centers() if require_triplets else set(range(len(dataset)))

        # Filtrage: ne garder que les centres valides
        pos_centers = [i for i in pos_centers if i in valid_centers]
        neg_centers = [i for i in neg_centers if i in valid_centers]

        self.pos_centers = np.asarray(pos_centers, dtype=np.int64)
        self.neg_centers = np.asarray(neg_centers, dtype=np.int64)

        total_centers = len(self.pos_centers) + len(self.neg_centers)
        self.batches_per_epoch = int(np.ceil(total_centers / self.centers_per_batch)) if total_centers>0 else 0

        print(f"[MITOSIS-TRIPLET] pos_centers={len(self.pos_centers)}  neg_centers={len(self.neg_centers)}  "
              f"centers/batch={self.centers_per_batch} (=> batch={self.batch_size})  "
              f"pos/batch={self.pos_per_batch}  batches/epoch={self.batches_per_epoch}")

        if len(self.pos_centers) == 0:
            print("[MITOSIS-TRIPLET][WARN] 0 centres POS valides — entraînement quasi full-NEG.")

    def _valid_triplet_centers(self) -> set:
        valid = set()
        for seq, fr2idx in self.seq2frames.items():
            if not fr2idx: continue
            frames = sorted(fr2idx.keys())
            s = set(frames)
            for t in frames:
                if (t-1) in s and (t+1) in s:
                    valid.add(fr2idx[t])
        return valid

    def _expand_center_to_triplet(self, center_idx: int) -> List[int]:
        """Retourne [idx(t-1), idx(t), idx(t+1)] pour un centre idx(t)."""
        # retrouver (seq, frame_t)
        if self.idx2meta is None:
            # reconstruit à la volée
            _, self.idx2meta = _build_seq_index(self.dataset)
        seq, t = self.idx2meta[center_idx]
        fr2idx = self.seq2frames[seq]
        return [fr2idx[t-1], fr2idx[t], fr2idx[t+1]]

    def __iter__(self):
        if self.batches_per_epoch <= 0:
            return
        pos = self.pos_centers.copy()
        neg = self.neg_centers.copy()
        if self.shuffle:
            self.rng.shuffle(pos)
            self.rng.shuffle(neg)
        p_ptr = n_ptr = 0
        p_len, n_len = len(pos), len(neg)

        for _ in range(self.batches_per_epoch):
            centers_batch: List[int] = []

            # POS centres
            take_p = min(self.pos_per_batch, p_len - p_ptr) if p_len>0 else 0
            if take_p < self.pos_per_batch and p_len>0:
                # reshuffle & wrap
                self.rng.shuffle(pos); p_ptr = 0
                take_p = min(self.pos_per_batch, p_len)
            if take_p > 0:
                centers_batch.extend(pos[p_ptr:p_ptr+take_p].tolist())
                p_ptr += take_p

            # NEG centres
            take_n = self.neg_per_batch
            avail_n = n_len - n_ptr
            if take_n > avail_n and n_len>0:
                self.rng.shuffle(neg); n_ptr = 0
                avail_n = n_len
            take_n = min(take_n, avail_n) if n_len>0 else 0
            if take_n > 0:
                centers_batch.extend(neg[n_ptr:n_ptr+take_n].tolist())
                n_ptr += take_n

            # Complète si insuffisant
            while len(centers_batch) < self.centers_per_batch:
                pool = pos if (p_len>0) else neg
                if len(pool)==0: break
                centers_batch.append(self.rng.choice(pool))

            # Mélange des centres, puis expansion en indices (t-1,t,t+1)
            if self.shuffle and len(centers_batch)>1:
                self.rng.shuffle(centers_batch)

            batch_indices: List[int] = []
            for c in centers_batch:
                trip = self._expand_center_to_triplet(c)
                batch_indices.extend(trip)  # <<< ordre contigu par triplet

            self.last_batch_indices = batch_indices
            self.last_triplet_centers = centers_batch
            yield batch_indices

    def __len__(self):
        return self.batches_per_epoch
