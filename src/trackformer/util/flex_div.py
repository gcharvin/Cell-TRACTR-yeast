import os
import torch

from .misc import man_track_ids
from .box_ops import (
    combine_div_boxes,
    calc_iou,
    combine_div_masks,
    divide_mask,
    divide_box,
    generalized_box_iou,
    box_cxcywh_to_xyxy,
)
from trackformer.util.misc import read_thresholds


# ---------------------------------------------------------------------
# Seuils (ENV ou défauts) + debug
# ---------------------------------------------------------------------
_cls, _div, _iou, _dbg = read_thresholds(None)
CLS_THR = _cls
DIV_THR = _div
DIV_BBOX_IOU_THR = _iou  # IoU utilisé pour les divisions

if _dbg:
    print(
        f"[DIVDBG] enabled  cls={CLS_THR:.2f}  div={DIV_THR:.2f}  IoU={DIV_BBOX_IOU_THR:.2f}",
        flush=True,
    )


# ---------------------------------------------------------------------
# Helpers robustes (torch / numpy) pour lever toute ambiguïté booléenne
# ---------------------------------------------------------------------

def _dbg_div_box_choice(box_f1, box_f2, chosen_box, extra=None):
    """
    Log léger pour diagnostiquer la construction de la GT de division.
    - box_f1, box_f2 : cx,cy,w,h des filles (t ou t+1 selon le cas)
    - chosen_box     : boîte retenue (cx,cy,w,h, cx,cy,w,h ou cx,cy,w,h si combinée)
    """
    def _tolist(t):
        import torch
        if isinstance(t, torch.Tensor):
            return [float(x) for x in t.detach().flatten().tolist()]
        return list(t)
    msg = (f"[DIVDBG:GT-CHOICE] d1={_tolist(box_f1[:4])}  d2={_tolist(box_f2[:4])}  "
           f"-> chosen={_tolist(chosen_box)}")
    if extra:
        # extra peut contenir fr, ids, etc.
        try:
            import json
            msg += f"  ctx={json.dumps(extra)}"
        except Exception:
            msg += f"  ctx={extra}"
    print(msg, flush=True)



def _to_int(x):
    """Retourne un int Python à partir d'un tensor 0-D / numpy / int."""
    try:
        if isinstance(x, torch.Tensor):
            return int(x.item()) if x.ndim == 0 else int(x)
        return int(x)
    except Exception:
        return int(x)

def _any(x):
    """True si un élément True dans x (torch/numpy/bool)."""
    try:
        if isinstance(x, torch.Tensor):
            return bool(x.any().item())
        import numpy as _np
        return bool(_np.any(x))
    except Exception:
        return bool(x)

def _any_eq(x, val):
    """True si un élément de x == val (torch/numpy)."""
    try:
        if isinstance(x, torch.Tensor):
            return bool((x == val).any().item())
        import numpy as _np
        return bool(_np.any(x == val))
    except Exception:
        return x == val

def _count_eq(x, val):
    """Nombre d'éléments égaux à val (torch/numpy)."""
    try:
        if isinstance(x, torch.Tensor):
            return int((x == val).sum().item())
        import numpy as _np
        return int((_np.asarray(x) == val).sum())
    except Exception:
        return 0

def _isin_scalar(val, arr):
    """Test 'val in arr' robuste pour des tenseurs 1D."""
    if isinstance(arr, torch.Tensor):
        return _any(arr == val)
    try:
        return val in arr
    except Exception:
        return False


# ---------------------------------------------------------------------
# Early/Late division updates (vectorisé et safe)
# ---------------------------------------------------------------------
def update_early_or_late_track_divisions(
    outputs,
    targets,
    training_method,
    prev_target_name,
    cur_target_name,
    fut_target_name,
):
    device = outputs["pred_logits"].device
    use_masks = "masks" in targets[0][training_method][cur_target_name]
    

    # check for early / late cell division and adjust ground truths as necessary
    for t, target in enumerate(targets):
        man_track = target[training_method]["man_track"]

        prev_target = target[training_method][prev_target_name]
        cur_target = target[training_method][cur_target_name]
        fut_target = target[training_method][fut_target_name]

        if cur_target["empty"]:
            continue

        if "track_query_match_ids" in cur_target:
            # Get all predictions for TP track queries
            pred_boxes_track = outputs["pred_boxes"][t][
                cur_target["track_queries_TP_mask"]
            ].detach()
            pred_logits_track = (
                outputs["pred_logits"][t][cur_target["track_queries_TP_mask"]]
                .sigmoid()
                .detach()
            )

            # ensure prev<->cur remap is up-to-date
            targets = man_track_ids(
                targets, training_method, prev_target_name, cur_target_name
            )

            boxes = cur_target["boxes"].clone()
            track_ids = cur_target["track_ids"].clone()

            for p, pred_box in enumerate(pred_boxes_track):
                cur_idx = cur_target["track_query_match_ids"][p]
                box = boxes[cur_idx].clone()
                track_id = track_ids[cur_idx].clone()

                # si ce track n'existait pas à t-1, essayer d'utiliser la mère
                prev_has = _any(prev_target["track_ids"] == track_id)
                track_id_prev = track_id  # défaut

                if not prev_has:
                    # man_track: [gid, start, end, parent]
                    mt = man_track
                    row = (mt[:, 0] == track_id)
                    if _any(row):
                        start = _to_int(mt[row, 1][0])
                        parent = _to_int(mt[row, 3][0])
                    else:
                        start, parent = -1, -1

                    # nouveau-né à cette frame -> utiliser la mère à t-1
                    if start == _to_int(cur_target["framenb"]) and parent > 0 and _any(
                        prev_target["track_ids"] == parent
                    ):
                        track_id_prev = prev_target["track_ids"].new_tensor(parent)
                    else:
                        # cas object-query de nouveau-né -> ignorer
                        continue

                # si flexible division à t-1, ne force rien
                prev_flex_mask = prev_target["track_ids"] == track_id_prev
                if _any(prev_target["flexible_divisions"][prev_flex_mask]):
                    continue

                # ----- LATE division fix -----
                # GT = division ; pred = single-cell
                cond_late = (
                    (box[-1] > 0).item()
                    and (pred_logits_track[p, 0] > CLS_THR).item()
                    and (pred_logits_track[p, -1] < DIV_THR).item()
                )
                if cond_late:
                    area_box_1 = box[2] * box[3]
                    area_box_2 = box[6] * box[7]

                    prev_box = prev_target["boxes"][prev_flex_mask]
                    area_prev_box = (
                        prev_box[0, 2] * prev_box[0, 3] if prev_box.shape[0] > 0 else 0
                    )

                    # division très asymétrique -> ne pas fusionner
                    if area_box_1 > area_box_2 * 2 or area_box_2 > area_box_1 * 2:
                        continue

                    combined_box = combine_div_boxes(box)
                    iou_div = calc_iou(box, pred_box)

                    pred_box_single = pred_box.clone()
                    pred_box_single[4:] = 0
                    iou_combined = calc_iou(combined_box, pred_box_single)
                    
                    # --- DEBUG (2% des cas) : on log les 2 filles (box[:4], box[4:]) et la boîte combinée choisie
                    if torch.rand(1, device=combined_box.device).item() < 0.5:
                        _dbg_div_box_choice(
                            box[:4], box[4:8], combined_box,
                            extra={
                                "phase": "LATE",
                                "fr": int(_to_int(cur_target.get("framenb", -1))),
                                "iou_div_vs_pred": float(_to_int(iou_div)),
                                "iou_combined_vs_pred": float(_to_int(iou_combined)),
                            }
                        )


                    if (iou_combined - iou_div > 0) and (iou_combined > DIV_BBOX_IOU_THR):
                        # remplacer la division GT par une boîte unique combinée
                        cur_target["boxes"][cur_idx] = combined_box
                        cur_target["labels"][cur_idx] = torch.tensor([0, 1]).to(device)
                        cur_target["flexible_divisions"][cur_idx] = True

                        track_id_now = cur_target["track_ids"][cur_idx].clone()

                        div_bool_1 = cur_target["boxes_orig"][:, :4].eq(
                            box[None, :4]
                        ).all(1)
                        div_bool_2 = cur_target["boxes_orig"][:, :4].eq(
                            box[None, 4:]
                        ).all(1)

                        div_track_id_1 = cur_target["track_ids_orig"][div_bool_1].clone()
                        div_track_id_2 = cur_target["track_ids_orig"][div_bool_2].clone()

                        cur_target["boxes_orig"][div_bool_1] = combined_box.clone()
                        cur_target["boxes_orig"] = cur_target["boxes_orig"][~div_bool_2]

                        cur_target["track_ids_orig"][div_bool_1] = track_id_now
                        cur_target["track_ids_orig"] = cur_target["track_ids_orig"][
                            ~div_bool_2
                        ]

                        cur_target["flexible_divisions_orig"][div_bool_1] = True
                        cur_target["flexible_divisions_orig"] = (
                            cur_target["flexible_divisions_orig"][~div_bool_2]
                        )

                        is_touching_edge = _any(
                            cur_target["is_touching_edge_orig"][div_bool_1]
                        ) or _any(cur_target["is_touching_edge_orig"][div_bool_2])
                        cur_target["is_touching_edge"][cur_idx] = is_touching_edge
                        cur_target["is_touching_edge_orig"][div_bool_1] = (
                            is_touching_edge
                        )
                        cur_target["is_touching_edge_orig"] = cur_target[
                            "is_touching_edge_orig"
                        ][~div_bool_2]

                        assert _any(cur_target["labels_orig"][div_bool_1, 1] == 1)
                        cur_target["labels_orig"] = cur_target["labels_orig"][
                            ~div_bool_2
                        ]

                        if use_masks:
                            mask = cur_target["masks"][cur_idx]
                            prev_mask = prev_target["masks"][prev_flex_mask][0]
                            combined_mask = combine_div_masks(mask, prev_mask)
                            cur_target["masks"][cur_idx] = combined_mask
                            cur_target["masks_orig"][div_bool_1] = combined_mask
                            cur_target["masks_orig"] = cur_target["masks_orig"][
                                ~div_bool_2
                            ]

                        track_id_ind = man_track[:, 0] == track_id_now
                        div_track_id_1_ind = man_track[:, 0] == div_track_id_1
                        div_track_id_2_ind = man_track[:, 0] == div_track_id_2

                        man_track[track_id_ind, 2] += 1
                        man_track[div_track_id_1_ind, 1] += 1
                        man_track[div_track_id_2_ind, 1] += 1

                        # fille qui sort juste après naissance -> la mère remplace
                        daughters_bad = _any(
                            man_track[div_track_id_1_ind, 1]
                            > man_track[div_track_id_1_ind, 2]
                        ) or _any(
                            man_track[div_track_id_2_ind, 1]
                            > man_track[div_track_id_2_ind, 2]
                        )
                        if daughters_bad:
                            man_track[track_id_ind, 2] = torch.max(
                                man_track[div_track_id_1_ind, 2],
                                man_track[div_track_id_2_ind, 2],
                            )
                            man_track[div_track_id_1_ind, 1:] = -1
                            man_track[div_track_id_2_ind, 1:] = -1

                            # update fut ids if needed
                            in_fut_1 = _isin_scalar(div_track_id_1, fut_target["track_ids"])
                            in_fut_2 = _isin_scalar(div_track_id_2, fut_target["track_ids"])

                            if in_fut_1 and in_fut_2:
                                raise NotImplementedError
                            elif in_fut_1:
                                fut_target["track_ids"][
                                    fut_target["track_ids"] == div_track_id_1
                                ] = track_id_now.long().to(device)
                                if fut_target_name != "fut_target" and _isin_scalar(
                                    div_track_id_1, target[training_method]["fut_target"]["track_ids"]
                                ):
                                    target[training_method]["fut_target"]["track_ids"][
                                        target[training_method]["fut_target"]["track_ids"]
                                        == div_track_id_1
                                    ] = track_id_now.long().to(device)
                            elif in_fut_2:
                                fut_target["track_ids"][
                                    fut_target["track_ids"] == div_track_id_2
                                ] = track_id_now.long().to(device)
                                if fut_target_name != "fut_target" and _isin_scalar(
                                    div_track_id_2, target[training_method]["fut_target"]["track_ids"]
                                ):
                                    target[training_method]["fut_target"]["track_ids"][
                                        target[training_method]["fut_target"]["track_ids"]
                                        == div_track_id_2
                                    ] = track_id_now.long().to(device)

                            fut_target["track_ids_orig"] = fut_target["track_ids"].clone()
                            if fut_target_name != "fut_target":
                                target[training_method]["fut_target"]["track_ids_orig"] = target[
                                    training_method
                                ]["fut_target"]["track_ids"].clone()

                            if _isin_scalar(div_track_id_1, man_track[:, -1]) and _isin_scalar(
                                div_track_id_2, man_track[:, -1]
                            ):
                                fut_div_track_id_1, fut_div_track_id_2 = man_track[
                                    (man_track[:, -1] == div_track_id_1), 0
                                ]
                                fut_div_track_id_1_ind = (
                                    man_track[:, 0] == fut_div_track_id_1
                                )
                                fut_div_track_id_2_ind = (
                                    man_track[:, 0] == fut_div_track_id_2
                                )
                                man_track[fut_div_track_id_1_ind, -1] = 0
                                man_track[fut_div_track_id_2_ind, -1] = 0
                                fut_div_track_id_1, fut_div_track_id_2 = man_track[
                                    (man_track[:, -1] == div_track_id_2), 0
                                ]
                                fut_div_track_id_1_ind = (
                                    man_track[:, 0] == fut_div_track_id_1
                                )
                                fut_div_track_id_2_ind = (
                                    man_track[:, 0] == fut_div_track_id_2
                                )
                                man_track[fut_div_track_id_1_ind, -1] = 0
                                man_track[fut_div_track_id_2_ind, -1] = 0
                            elif _isin_scalar(div_track_id_1, man_track[:, -1]):
                                fut_div_track_id_1, fut_div_track_id_2 = man_track[
                                    (man_track[:, -1] == div_track_id_1), 0
                                ]
                                fut_div_track_id_1_ind = (
                                    man_track[:, 0] == fut_div_track_id_1
                                )
                                fut_div_track_id_2_ind = (
                                    man_track[:, 0] == fut_div_track_id_2
                                )
                                man_track[fut_div_track_id_1_ind, -1] = track_id_now
                                man_track[fut_div_track_id_2_ind, -1] = track_id_now
                            elif _isin_scalar(div_track_id_2, man_track[:, -1]):
                                fut_div_track_id_1, fut_div_track_id_2 = man_track[
                                    (man_track[:, -1] == div_track_id_2), 0
                                ]
                                fut_div_track_id_1_ind = (
                                    man_track[:, 0] == fut_div_track_id_1
                                )
                                fut_div_track_id_2_ind = (
                                    man_track[:, 0] == fut_div_track_id_2
                                )
                                man_track[fut_div_track_id_1_ind, -1] = track_id_now
                                man_track[fut_div_track_id_2_ind, -1] = track_id_now

                            assert (
                                torch.arange(
                                    1,
                                    target[training_method]["man_track"].shape[0] + 1,
                                    dtype=target[training_method]["man_track"].dtype,
                                    device=target[training_method]["man_track"].device,
                                )
                                == target[training_method]["man_track"][:, 0]
                            ).all()

            # ensure cur->fut remap before early checks
            targets = man_track_ids(targets, training_method, cur_target_name, fut_target_name)

            # ----- EARLY division fix -----
            for p, pred_box in enumerate(pred_boxes_track):
                cur_idx = cur_target["track_query_match_ids"][p]
                box = boxes[cur_idx].clone()

                # GT single ; pred division
                cond_early = (
                    (box[-1] == 0).item()
                    and (pred_logits_track[p, 0] > CLS_THR).item()
                    and (pred_logits_track[p, -1] > DIV_THR).item()
                )
                if not cond_early:
                    continue

                track_id = track_ids[cur_idx].clone()
                track_id_ind = man_track[:, 0] == track_id

                # si juste née à t-1, remonter à la mère
                if _any_eq(man_track[track_id_ind, 2], prev_target["framenb"]):
                    track_id = man_track[track_id_ind, -1]

                if not _isin_scalar(track_id, fut_target["track_ids"]):
                    continue  # sort du champ

                fut_box_ind = (fut_target["track_ids"] == track_id).nonzero()[0][0]
                fut_box = fut_target["boxes"][fut_box_ind]

                if (fut_box[-1] > 0).item():  # divise à t+1 -> early division possible
                    div_box = divide_box(box, fut_box)

                    iou_div = calc_iou(div_box, pred_box)
                    iou = calc_iou(box[:4], pred_box[:4])

                    if (iou_div - iou > 0) and (iou_div > DIV_BBOX_IOU_THR):
                        cur_target["boxes"][cur_idx] = div_box
                        cur_target["labels"][cur_idx] = torch.tensor([0, 0]).to(device)
                        cur_target["flexible_divisions_orig"][cur_idx] = True

                        fut_track_id_1, fut_track_id_2 = man_track[
                            man_track[:, -1] == track_id, 0
                        ]
                        fut_box_1 = fut_target["boxes_orig"][
                            fut_target["track_ids_orig"] == fut_track_id_1
                        ][0]
                        fut_box_2 = fut_target["boxes_orig"][
                            fut_target["track_ids_orig"] == fut_track_id_2
                        ][0]
                        
                        # --- DEBUG (2% des cas) : on log les 2 filles (t+1) et la boîte de division construite à t
                        if torch.rand(1, device=div_box.device).item() < 0.5:
                            _dbg_div_box_choice(
                                fut_box_1[:4], fut_box_2[:4], div_box,
                                extra={
                                    "phase": "EARLY",
                                    "fr": int(_to_int(cur_target.get("framenb", -1))),
                                    "iou_div_vs_pred": float(_to_int(iou_div)),
                                    "iou_single_vs_pred": float(_to_int(iou)),
                                }
                            )


                        if (
                            (div_box[:2] - fut_box_1[:2]).square().sum()
                            + (div_box[4:6] - fut_box_2[:2]).square().sum()
                            > (div_box[:2] - fut_box_2[:2]).square().sum()
                            + (div_box[4:6] - fut_box_1[:2]).square().sum()
                        ):
                            fut_track_id_1, fut_track_id_2 = fut_track_id_2, fut_track_id_1

                        assert (
                            torch.arange(
                                1,
                                target[training_method]["man_track"].shape[0] + 1,
                                dtype=target[training_method]["man_track"].dtype,
                                device=target[training_method]["man_track"].device,
                            )
                            == target[training_method]["man_track"][:, 0]
                        ).all()

                        ind_tgt_orig = torch.where(
                            cur_target["boxes_orig"].eq(box).all(-1)
                        )[0][0]
                        cur_target["boxes_orig"][ind_tgt_orig, :4] = div_box[:4]
                        cur_target["boxes_orig"] = torch.cat(
                            (
                                cur_target["boxes_orig"],
                                torch.cat((div_box[4:], torch.zeros_like(div_box[4:])))[
                                    None
                                ],
                            ),
                            dim=0,
                        )

                        cur_target["track_ids_orig"][ind_tgt_orig] = fut_track_id_1
                        cur_target["track_ids_orig"] = torch.cat(
                            (
                                cur_target["track_ids_orig"],
                                torch.tensor([fut_track_id_2]).to(device),
                            )
                        )

                        cur_target["labels_orig"] = torch.cat(
                            (cur_target["labels_orig"], cur_target["labels_orig"][:1]),
                            dim=0,
                        )
                        cur_target["flexible_divisions_orig"] = torch.cat(
                            (
                                cur_target["flexible_divisions_orig"],
                                torch.tensor([True]).to(device),
                            )
                        )

                        cur_target["is_touching_edge_orig"] = torch.cat(
                            (
                                cur_target["is_touching_edge_orig"],
                                cur_target["is_touching_edge_orig"][ind_tgt_orig][None],
                            )
                        )

                        if use_masks:
                            mask = cur_target["masks"][cur_idx]
                            fut_mask = fut_target["masks"][fut_box_ind]
                            div_mask = divide_mask(mask, fut_mask)
                            cur_target["masks"][cur_idx] = div_mask
                            cur_target["masks_orig"][ind_tgt_orig, :1] = div_mask[:1]
                            cur_target["masks_orig"] = torch.cat(
                                (
                                    cur_target["masks_orig"],
                                    torch.cat(
                                        (div_mask[1:], torch.zeros_like(div_mask[1:]))
                                    )[None],
                                ),
                                dim=0,
                            )

                        fut_track_id_1_ind = man_track[:, 0] == fut_track_id_1
                        fut_track_id_2_ind = man_track[:, 0] == fut_track_id_2

                        man_track[track_id_ind, 2] -= 1
                        man_track[fut_track_id_1_ind, 1] -= 1
                        man_track[fut_track_id_2_ind, 1] -= 1

                        if _any(
                            man_track[track_id_ind, 1] > man_track[track_id_ind, 2]
                        ):
                            man_track[track_id_ind, 1:] = -1
                            man_track[fut_track_id_1_ind, -1] = 0
                            man_track[fut_track_id_2_ind, -1] = 0

        targets[t][training_method]["man_track"] = man_track

    # Rebuild masks (post-loop refresh; s'aligne sur la logique originale)
    # NB: on réutilise les "prev/cur" de la dernière itération — comportement identique au code d'origine
    if "track_query_match_ids" in cur_target:
        targets = man_track_ids(targets, training_method, prev_target_name, cur_target_name)

        prev_track_ids = prev_target["track_ids"][cur_target["prev_ind"][1]]

        # match track ids entre frames
        target_ind_match_matrix = prev_track_ids.unsqueeze(dim=1).eq(
            cur_target["track_ids"]
        )
        cur_target["target_ind_matching"] = target_ind_match_matrix.any(dim=1)
        cur_target["track_query_match_ids"] = target_ind_match_matrix.nonzero()[:, 1]

        # garde-fou si zéro cellule
        if cur_target["target_ind_matching"].shape[0] == 0:
            cur_target["target_ind_matching"] = torch.tensor([], device=device).bool()

        track_queries_mask = torch.ones_like(cur_target["target_ind_matching"]).bool()
        num_queries = (~cur_target["track_queries_mask"]).sum()

        cur_target["track_queries_mask"] = torch.cat(
            [
                track_queries_mask,
                torch.tensor([True] * cur_target["num_FPs"]).to(device),
                torch.tensor([False] * num_queries).to(device),
            ]
        ).bool()

        cur_target["track_queries_TP_mask"] = torch.cat(
            [
                cur_target["target_ind_matching"],
                torch.tensor([False] * cur_target["num_FPs"]).to(device),
                torch.tensor([False] * num_queries).to(device),
            ]
        ).bool()

        cur_target["track_queries_fal_pos_mask"] = torch.cat(
            [
                ~cur_target["target_ind_matching"],
                torch.tensor([True] * cur_target["num_FPs"]).to(device),
                torch.tensor([False] * num_queries).to(device),
            ]
        ).bool()

        assert (
            cur_target["track_queries_TP_mask"].sum()
            == len(cur_target["track_query_match_ids"])
        )

    return targets


# ---------------------------------------------------------------------
# OD / tracking update (vectorisé et safe)
# ---------------------------------------------------------------------
def update_object_detection(
    outputs,
    targets,
    indices,
    num_queries,
    training_method,
    prev_target_name,
    cur_target_name,
    fut_target_name,
):
    N = outputs["pred_logits"].shape[1]
    use_masks = "masks" in targets[0][training_method][cur_target_name]
    device = outputs["pred_logits"].device

    # Indices are saved in targets for calculating object detection / tracking accuracy
    for t, (target, (ind_out, ind_tgt)) in enumerate(zip(targets, indices)):
        prev_target = target[training_method][prev_target_name]
        cur_target = target[training_method][cur_target_name]
        fut_target = target[training_method][fut_target_name]

        if cur_target["empty"]:
            continue

        if training_method == "dn_object":
            cur_target["track_queries_mask"] = torch.zeros_like(
                cur_target["track_queries_mask"]
            ).bool()

        man_track = target[training_method]["man_track"]
        framenb = _to_int(cur_target["framenb"])

        skip = []  # si une GT est fusionnée, on saute la seconde
        ind_keep = torch.tensor([True for _ in range(len(ind_tgt))]).bool()

        for ind_out_i, ind_tgt_i in zip(ind_out, ind_tgt):
            # Confirm prediction is an object query, not a track query
            if ind_out_i >= (N - num_queries) and ind_tgt_i not in skip:
                if "track_queries_mask" in cur_target:
                    assert not cur_target["track_queries_mask"][ind_out_i]
                track_id = cur_target["track_ids"][ind_tgt_i].clone()

                track_id_ind = man_track[:, 0] == track_id

                # === Cas: la cellule vient de NAÎTRE à cette frame (deux filles présentes)
                rows = man_track[track_id_ind]  # (4,) ou (k,4)

                if rows.ndim == 1:
                    start_here = bool((rows[1] == framenb) and (rows[-1] > 0))
                else:
                    start_here = _any(
                        (rows[:, 1] == framenb) & (rows[:, -1] > 0)
                    )

                if start_here:
                    # id de la mère
                    mother_id = man_track[track_id_ind, -1].clone().long()
                    assert _isin_scalar(mother_id, prev_target["track_ids"])

                    track_id_1 = track_id

                    # indices filles 1 & 2
                    ind_tgt_1 = ind_tgt_i
                    ind_out_1 = ind_out_i
                    ind_1 = torch.where(ind_out == ind_out_1)[0][0]

                    track_id_2 = man_track[
                        (man_track[:, -1] == mother_id) * (man_track[:, 0] != track_id_1),
                        0,
                    ][0].clone()
                    ind_tgt_2 = torch.where(cur_target["track_ids"] == track_id_2)[0][0].cpu()
                    ind_2 = torch.where(ind_tgt == ind_tgt_2)[0][0]
                    ind_out_2 = ind_out[ind_2]

                    # prédictions pour 1 & 2
                    pred_box_1 = outputs["pred_boxes"][t, ind_out_1].detach()
                    pred_box_2 = outputs["pred_boxes"][t, ind_out_2].detach()

                    # GT pour 1 & 2
                    box_1 = cur_target["boxes"][ind_tgt_1]
                    box_2 = cur_target["boxes"][ind_tgt_2]
                    assert (
                        box_1[-1] == 0 and box_2[-1] == 0
                    ), "Cells have just divided. Each box should contain just one cell"

                    # formatage concat
                    boxes_1_2 = torch.cat((box_1[:4], box_2[:4]))
                    pred_boxes_1_2 = torch.cat((pred_box_1[:4], pred_box_2[:4]))

                    # IoU séparés / flip
                    iou_sep, flip = calc_iou(pred_boxes_1_2, boxes_1_2, return_flip=True)

                    if flip:
                        iou_1 = calc_iou(pred_boxes_1_2[:4], boxes_1_2[4:])
                        pred_logits_1 = outputs["pred_logits"][t, ind_out_2].sigmoid()[0]
                        iou_2 = calc_iou(pred_boxes_1_2[4:], boxes_1_2[:4])
                        pred_logits_2 = outputs["pred_logits"][t, ind_out_1].sigmoid()[0]
                    else:
                        iou_1 = calc_iou(pred_boxes_1_2[:4], boxes_1_2[:4])
                        pred_logits_1 = outputs["pred_logits"][t, ind_out_1].sigmoid()[0]
                        iou_2 = calc_iou(pred_boxes_1_2[4:], boxes_1_2[4:])
                        pred_logits_2 = outputs["pred_logits"][t, ind_out_2].sigmoid()[0]

                    # hypothèse: GT1 & GT2 étaient en fait une seule cellule
                    combined_box = combine_div_boxes(boxes_1_2)

                    # tous les object queries non utilisés qui prédisent "cellule"
                    used = set([_to_int(x) for x in ind_out.tolist()])
                    pot_inds = []
                    for ind_out_id in range(N - num_queries, N):
                        is_used = (ind_out_id in used) and (
                            ind_out_id not in [_to_int(ind_out_1), _to_int(ind_out_2)]
                        )
                        cls_ok = (
                            outputs["pred_logits"][t, ind_out_id, 0].sigmoid().item()
                            > CLS_THR
                        )
                        if (not is_used) and cls_ok:
                            pot_inds.append(ind_out_id)

                    if len(pot_inds) == 0:
                        continue

                    potential_pred_boxes = outputs["pred_boxes"][t, pot_inds].detach()

                    iou_combined = generalized_box_iou(
                        box_cxcywh_to_xyxy(potential_pred_boxes[:, :4]),
                        box_cxcywh_to_xyxy(combined_box[None, :4]),
                        return_iou_only=True,
                    )

                    max_ind = torch.argmax(iou_combined)
                    assert (
                        0.0 <= iou_combined[max_ind] <= 1.0 and 0.0 <= iou_sep <= 1.0
                    ), "calc_iou out of range"

                    if (
                        iou_combined[max_ind] - iou_sep > 0
                        and iou_combined[max_ind] > CLS_THR
                        and (iou_combined[max_ind] > iou_1 or pred_logits_1 < 0.5)
                        and (iou_combined[max_ind] > iou_2 or pred_logits_2 < 0.5)
                    ):
                        ind_out_combined = pot_inds[_to_int(max_ind)]

                        # on supprime la seconde fille
                        ind_keep[ind_2] = False
                        skip += [ind_tgt_1, ind_tgt_2]

                        # fusion: box 1 <- combined, 2 supprimée
                        cur_target["boxes"][ind_tgt_1] = combined_box
                        mother_id = man_track[track_id_ind, -1].clone().long()
                        cur_target["track_ids"][ind_tgt_1] = mother_id
                        cur_target["track_ids"][ind_tgt_2] = -1
                        cur_target["flexible_divisions"][ind_tgt_1] = True
                        cur_target["is_touching_edge"][ind_tgt_1] = (
                            cur_target["is_touching_edge"][ind_tgt_1]
                            or cur_target["is_touching_edge"][ind_tgt_2]
                        )

                        assert not _isin_scalar(mother_id, cur_target["track_ids_orig"])
                        ind_tgt_orig_1 = cur_target["track_ids_orig"] == track_id_1
                        cur_target["track_ids_orig"][ind_tgt_orig_1] = mother_id
                        cur_target["boxes_orig"][ind_tgt_orig_1] = combined_box
                        cur_target["flexible_divisions_orig"][ind_tgt_orig_1] = True
                        cur_target["is_touching_edge_orig"][ind_tgt_orig_1] = cur_target[
                            "is_touching_edge"
                        ][ind_tgt_1]

                        ind_orig_keep = cur_target["track_ids_orig"] != track_id_2
                        cur_target["track_ids_orig"] = cur_target["track_ids_orig"][ind_orig_keep]
                        cur_target["boxes_orig"] = cur_target["boxes_orig"][ind_orig_keep]
                        cur_target["labels_orig"] = cur_target["labels_orig"][ind_orig_keep]
                        cur_target["flexible_divisions_orig"] = cur_target[
                            "flexible_divisions_orig"
                        ][ind_orig_keep]
                        cur_target["is_touching_edge_orig"] = cur_target[
                            "is_touching_edge_orig"
                        ][ind_orig_keep]

                        if use_masks:
                            mother_ind = torch.where(prev_target["track_ids"] == mother_id)[0][0]
                            prev_mask = prev_target["masks"][mother_ind][:1]

                            mask_1 = cur_target["masks"][ind_tgt_1].detach()[:1]
                            mask_2 = cur_target["masks"][ind_tgt_2].detach()[:1]
                            sep_mask = torch.cat((mask_1, mask_2), dim=0)

                            combined_mask = combine_div_masks(sep_mask, prev_mask)

                            cur_target["masks"][ind_tgt_1] = combined_mask
                            cur_target["masks_orig"][ind_tgt_orig_1] = combined_mask
                            cur_target["masks_orig"] = cur_target["masks_orig"][ind_orig_keep]

                        # remplacer la prédiction 1 par l'object query combiné
                        ind_out[ind_1] = ind_out_combined

                        track_id_mot_ind = man_track[:, 0] == mother_id
                        track_id_1_ind = man_track[:, 0] == track_id_1
                        track_id_2_ind = man_track[:, 0] == track_id_2

                        man_track[track_id_mot_ind, 2] += 1
                        man_track[track_id_1_ind, 1] += 1
                        man_track[track_id_2_ind, 1] += 1

                        # si une fille sort rapidement, remplacer par la mère
                        cond_out_1 = _any(man_track[track_id_1_ind, 2] < man_track[track_id_1_ind, 1])
                        cond_out_2 = _any(man_track[track_id_2_ind, 2] < man_track[track_id_2_ind, 1])
                        if cond_out_1 or cond_out_2:
                            man_track[track_id_mot_ind, 2] = torch.max(
                                man_track[track_id_1_ind, 2], man_track[track_id_2_ind, 2]
                            )
                            man_track[track_id_1_ind, 1:] = -1
                            man_track[track_id_2_ind, 1:] = -1

                            # mettre à jour les fut ids -> mother_id
                            if _isin_scalar(track_id_1, fut_target["track_ids_orig"]):
                                fut_target["track_ids_orig"][
                                    fut_target["track_ids_orig"] == track_id_1
                                ] = mother_id
                            elif _isin_scalar(track_id_2, fut_target["track_ids_orig"]):
                                fut_target["track_ids_orig"][
                                    fut_target["track_ids_orig"] == track_id_2
                                ] = mother_id

                            if _isin_scalar(track_id_1, man_track[:, -1]) and _isin_scalar(
                                track_id_2, man_track[:, -1]
                            ):
                                div_track_id_1, div_track_id_2 = man_track[
                                    (man_track[:, -1] == track_id_1), 0
                                ]
                                div_track_id_1_ind = man_track[:, 0] == div_track_id_1
                                div_track_id_2_ind = man_track[:, 0] == div_track_id_2
                                man_track[div_track_id_1_ind, -1] = 0
                                man_track[div_track_id_2_ind, -1] = 0
                                div_track_id_1, div_track_id_2 = man_track[
                                    (man_track[:, -1] == track_id_2), 0
                                ]
                                div_track_id_1_ind = man_track[:, 0] == div_track_id_1
                                div_track_id_2_ind = man_track[:, 0] == div_track_id_2
                                man_track[div_track_id_1_ind, -1] = 0
                                man_track[div_track_id_2_ind, -1] = 0
                            elif _isin_scalar(track_id_1, man_track[:, -1]):
                                div_track_id_1, div_track_id_2 = man_track[
                                    (man_track[:, -1] == track_id_1), 0
                                ]
                                div_track_id_1_ind = man_track[:, 0] == div_track_id_1
                                div_track_id_2_ind = man_track[:, 0] == div_track_id_2
                                man_track[div_track_id_1_ind, -1] = mother_id
                                man_track[div_track_id_2_ind, -1] = mother_id
                            elif _isin_scalar(track_id_2, man_track[:, -1]):
                                div_track_id_1, div_track_id_2 = man_track[
                                    (man_track[:, -1] == track_id_2), 0
                                ]
                                div_track_id_1_ind = man_track[:, 0] == div_track_id_1
                                div_track_id_2_ind = man_track[:, 0] == div_track_id_2
                                man_track[div_track_id_1_ind, -1] = mother_id
                                man_track[div_track_id_2_ind, -1] = mother_id

                # === Cas: la cellule est sur le point de DIVISER (fin de piste à framenb + 2 filles)
                else:
                    end_here = man_track[track_id_ind, 2]
                    end_at_frame = _any_eq(end_here, framenb)
                    two_daughters = _count_eq(man_track[:, -1], track_id) == 2
                    if (
                        end_at_frame
                        and two_daughters
                        and training_method != "dn_object"
                    ):
                        # GT de la mère à t
                        box = cur_target["boxes"][ind_tgt_i].clone()

                        fut_track_id_1, fut_track_id_2 = man_track[
                            (man_track[:, -1] == track_id), 0
                        ]

                        # indices filles à t+1
                        fut_ind_tgt_1 = torch.where(
                            fut_target["track_ids_orig"] == fut_track_id_1
                        )[0][0]
                        fut_ind_tgt_2 = torch.where(
                            fut_target["track_ids_orig"] == fut_track_id_2
                        )[0][0]

                        # boîtes filles t+1
                        fut_box_1 = fut_target["boxes_orig"][fut_ind_tgt_1, :4]
                        fut_box_2 = fut_target["boxes_orig"][fut_ind_tgt_2, :4]
                        fut_box = torch.cat((fut_box_1, fut_box_2))

                        # simulate divided cell at t
                        div_box = divide_box(box, fut_box)

                        # candidats object queries (incluant celui déjà matché)
                        pot_inds = []
                        used = set([_to_int(x) for x in ind_out.tolist()])
                        for ind_out_id in range(N - num_queries, N):
                            if ind_out_id == _to_int(ind_out_i):
                                pot_inds.append(ind_out_id)
                            else:
                                cls_ok = (
                                    outputs["pred_logits"][
                                        t, ind_out_id, 0
                                    ].sigmoid().item()
                                    > CLS_THR
                                )
                                if cls_ok and (ind_out_id not in used):
                                    pot_inds.append(ind_out_id)

                        if len(pot_inds) > 1:
                            potential_pred_boxes = outputs["pred_boxes"][t, pot_inds].detach()

                            iou_div_all = generalized_box_iou(
                                box_cxcywh_to_xyxy(potential_pred_boxes[:, :4]),
                                box_cxcywh_to_xyxy(
                                    torch.cat(
                                        (div_box[None, :4], div_box[None, 4:]), dim=0
                                    )
                                ),
                                return_iou_only=True,
                            )

                            match_ind = torch.argmax(iou_div_all, dim=0).cpu()

                            if (
                                pot_inds[_to_int(match_ind[0])] != _to_int(ind_out_i)
                                and pot_inds[_to_int(match_ind[1])] != _to_int(ind_out_i)
                            ):
                                continue

                            if len(torch.unique(match_ind)) == 2:
                                selected_pred_boxes = potential_pred_boxes[
                                    match_ind, :4
                                ]
                                iou_div = calc_iou(
                                    div_box,
                                    torch.cat(
                                        (
                                            selected_pred_boxes[0],
                                            selected_pred_boxes[1],
                                        )
                                    ),
                                )

                                pred_box = outputs["pred_boxes"][t, ind_out_i, :4].detach()
                                iou = calc_iou(
                                    box,
                                    torch.cat((pred_box, torch.zeros_like(pred_box))),
                                )

                                assert 0.0 <= iou_div <= 1.0 and 0.0 <= iou <= 1.0

                                if (iou_div - iou > 0) and (iou_div > 0.5):
                                    if (
                                        calc_iou(div_box[:4], selected_pred_boxes[0])
                                        + calc_iou(div_box[4:], selected_pred_boxes[1])
                                        < calc_iou(div_box[4:], selected_pred_boxes[0])
                                        + calc_iou(div_box[:4], selected_pred_boxes[1])
                                    ):
                                        fut_track_id_1, fut_track_id_2 = (
                                            fut_track_id_2,
                                            fut_track_id_1,
                                        )

                                    cur_target["boxes"][ind_tgt_i] = torch.cat(
                                        (div_box[:4], torch.zeros_like(div_box[:4]))
                                    )
                                    cur_target["boxes"] = torch.cat(
                                        (
                                            cur_target["boxes"],
                                            torch.cat(
                                                (div_box[4:], torch.zeros_like(div_box[:4]))
                                            )[None],
                                        )
                                    )

                                    assert cur_target["labels"][ind_tgt_i, 1] == 1
                                    cur_target["labels"] = torch.cat(
                                        (
                                            cur_target["labels"],
                                            torch.tensor([0, 1])[None, :].to(device),
                                        )
                                    )
                                    cur_target["track_ids"][ind_tgt_i] = fut_track_id_1
                                    cur_target["track_ids"] = torch.cat(
                                        (
                                            cur_target["track_ids"],
                                            torch.tensor([fut_track_id_2]).to(device),
                                        )
                                    )
                                    cur_target["flexible_divisions"][ind_tgt_i] = True
                                    cur_target["flexible_divisions"] = torch.cat(
                                        (
                                            cur_target["flexible_divisions"],
                                            torch.tensor([True]).to(device),
                                        )
                                    )
                                    cur_target["is_touching_edge"] = torch.cat(
                                        (
                                            cur_target["is_touching_edge"],
                                            cur_target["is_touching_edge"][ind_tgt_i][
                                                None
                                            ],
                                        )
                                    )

                                    ind_keep = torch.cat((ind_keep, torch.tensor([True])))

                                    ind_tgt_orig_i = torch.where(
                                        cur_target["boxes_orig"].eq(box).all(-1)
                                    )[0][0]

                                    cur_target["boxes_orig"][ind_tgt_orig_i] = torch.cat(
                                        (div_box[:4], torch.zeros_like(div_box[:4]))
                                    )
                                    cur_target["boxes_orig"] = torch.cat(
                                        (
                                            cur_target["boxes_orig"],
                                            torch.cat(
                                                (div_box[4:], torch.zeros_like(div_box[:4]))
                                            )[None],
                                        )
                                    )

                                    cur_target["labels_orig"] = torch.cat(
                                        (
                                            cur_target["labels_orig"],
                                            torch.tensor([0, 1])[None, :].to(device),
                                        )
                                    )
                                    cur_target["track_ids_orig"][ind_tgt_orig_i] = (
                                        fut_track_id_1
                                    )
                                    cur_target["track_ids_orig"] = torch.cat(
                                        (
                                            cur_target["track_ids_orig"],
                                            torch.tensor([fut_track_id_2]).to(device),
                                        )
                                    )
                                    cur_target["flexible_divisions_orig"][
                                        ind_tgt_orig_i
                                    ] = True
                                    cur_target["flexible_divisions_orig"] = torch.cat(
                                        (
                                            cur_target["flexible_divisions_orig"],
                                            torch.tensor([True]).to(device),
                                        )
                                    )
                                    cur_target["is_touching_edge_orig"] = torch.cat(
                                        (
                                            cur_target["is_touching_edge_orig"],
                                            cur_target["is_touching_edge"][ind_tgt_orig_i][
                                                None
                                            ],
                                        )
                                    )

                                    if use_masks:
                                        mask = cur_target["masks"][ind_tgt_i]
                                        fut_mask_1 = fut_target["masks_orig"][
                                            fut_ind_tgt_1
                                        ][:1]
                                        fut_mask_2 = fut_target["masks_orig"][
                                            fut_ind_tgt_2
                                        ][:1]
                                        fut_mask = torch.cat((fut_mask_1, fut_mask_2))
                                        div_mask = divide_mask(mask, fut_mask)

                                        cur_target["masks"][ind_tgt_i] = torch.cat(
                                            (div_mask[:1], torch.zeros_like(div_mask[:1]))
                                        )
                                        cur_target["masks"] = torch.cat(
                                            (
                                                cur_target["masks"],
                                                torch.cat(
                                                    (
                                                        div_mask[1:],
                                                        torch.zeros_like(div_mask[:1]),
                                                    )
                                                )[None],
                                            )
                                        )

                                        cur_target["masks_orig"][ind_tgt_orig_i] = torch.cat(
                                            (div_mask[:1], torch.zeros_like(div_mask[:1]))
                                        )
                                        cur_target["masks_orig"] = torch.cat(
                                            (
                                                cur_target["masks_orig"],
                                                torch.cat(
                                                    (
                                                        div_mask[1:],
                                                        torch.zeros_like(div_mask[:1]),
                                                    )
                                                )[None],
                                            )
                                        )

                                    # synchroniser ind_out
                                    ind_out_copy = torch.cat((ind_out, torch.tensor([-10])))

                                    # invalider un des deux prédits si nécessaire
                                    pot_1 = pot_inds[_to_int(match_ind[1])]
                                    pot_0 = pot_inds[_to_int(match_ind[0])]
                                    if pot_1 != _to_int(ind_out_i) and _isin_scalar(
                                        pot_1, ind_out
                                    ):
                                        ind_out[ind_out == pot_1] = -1
                                    elif pot_0 != _to_int(ind_out_i) and _isin_scalar(
                                        pot_0, ind_out
                                    ):
                                        ind_out[ind_out == pot_0] = -1

                                    ind_out = torch.cat(
                                        (ind_out, torch.tensor([pot_1]))
                                    )
                                    ind_tgt = torch.cat(
                                        (
                                            ind_tgt,
                                            torch.tensor([cur_target["boxes"].shape[0] - 1]),
                                        )
                                    )

                                    ind_out[ind_out_copy == ind_out_i] = torch.tensor([pot_0])

                                    if _any(ind_out == -1):
                                        unmatched_box = cur_target["boxes"][ind_out == -1]
                                        # candidats restants
                                        pot_inds2 = []
                                        used2 = set([_to_int(x) for x in ind_out.tolist()])
                                        for ind_out_id in range(N - num_queries, N):
                                            if (ind_out_id not in used2) and (
                                                outputs["pred_logits"][
                                                    t, ind_out_id, 0
                                                ].sigmoid().item()
                                                > 0.5
                                            ):
                                                pot_inds2.append(ind_out_id)

                                        if len(pot_inds2) == 0:
                                            for ind_out_id in range(N - num_queries, N):
                                                if ind_out_id not in used2:
                                                    pot_inds2.append(ind_out_id)

                                        if len(pot_inds2) == 0:
                                            continue

                                        potential_pred_boxes2 = outputs["pred_boxes"][
                                            t, pot_inds2
                                        ].detach()

                                        iou_div_all2 = generalized_box_iou(
                                            box_cxcywh_to_xyxy(potential_pred_boxes2[:, :4]),
                                            box_cxcywh_to_xyxy(unmatched_box[:, :4]),
                                            return_iou_only=True,
                                        )

                                        if iou_div_all2.sum() == 0:
                                            match_ind2 = torch.randint(
                                                low=0,
                                                high=len(pot_inds2),
                                                size=(1,),
                                                dtype=torch.int64,
                                            )
                                        else:
                                            match_ind2 = torch.argmax(iou_div_all2, dim=0).cpu()

                                        chosen = pot_inds2[_to_int(match_ind2)]
                                        assert not _isin_scalar(chosen, ind_out)
                                        ind_out[ind_out == -1] = chosen

                                    assert not _any(ind_out == -1)
                                    assert len(ind_out) == len(ind_tgt)
                                    assert len(cur_target["boxes"]) == len(
                                        cur_target["labels"]
                                    )

                                    fut_track_id_1_ind = man_track[:, 0] == fut_track_id_1
                                    fut_track_id_2_ind = man_track[:, 0] == fut_track_id_2

                                    man_track[track_id_ind, 2] -= 1
                                    man_track[fut_track_id_1_ind, 1] -= 1
                                    man_track[fut_track_id_2_ind, 1] -= 1

                                    if _any(
                                        man_track[track_id_ind, 1]
                                        > man_track[track_id_ind, 2]
                                    ):
                                        man_track[track_id_ind, 1:] = -1
                                        man_track[fut_track_id_1_ind, -1] = 0
                                        man_track[fut_track_id_2_ind, -1] = 0

        if training_method == "dn_object":
            if cur_target["num_FPs"] > 0:
                cur_target["track_queries_fal_pos_mask"][
                    :-cur_target["num_FPs"]
                ][cur_target["track_ids"] == -1] = True
            else:
                cur_target["track_queries_fal_pos_mask"][
                    cur_target["track_ids"] == -1
                ] = True

        # appliquer ind_keep
        order = ind_tgt[ind_keep].sort()[0]
        cur_target["boxes"] = cur_target["boxes"][order]
        cur_target["labels"] = cur_target["labels"][order]
        cur_target["track_ids"] = cur_target["track_ids"][order]
        cur_target["flexible_divisions"] = cur_target["flexible_divisions"][order]
        cur_target["is_touching_edge"] = cur_target["is_touching_edge"][order]
        if use_masks:
            cur_target["masks"] = cur_target["masks"][order]

        if "track_query_match_ids" in cur_target and training_method != "dn_object":
            prev_track_ids = prev_target["track_ids"][cur_target["prev_ind"][1]]

            target_ind_match_matrix = prev_track_ids.unsqueeze(dim=1).eq(
                cur_target["track_ids"]
            )
            cur_target["target_ind_matching"] = target_ind_match_matrix.any(dim=1)
            cur_target["track_query_match_ids"] = target_ind_match_matrix.nonzero()[:, 1]

            if cur_target["target_ind_matching"].shape[0] == 0:
                cur_target["target_ind_matching"] = torch.tensor([], device=device).bool()

            track_queries_mask = torch.ones_like(
                cur_target["target_ind_matching"]
            ).bool()

            cur_target["track_queries_mask"] = torch.cat(
                [
                    track_queries_mask,
                    torch.tensor([True] * cur_target["num_FPs"]).to(device),
                    torch.tensor([False] * num_queries).to(device),
                ]
            ).bool()

            cur_target["track_queries_TP_mask"] = torch.cat(
                [
                    cur_target["target_ind_matching"],
                    torch.tensor([False] * cur_target["num_FPs"]).to(device),
                    torch.tensor([False] * num_queries).to(device),
                ]
            ).bool()

            cur_target["track_queries_fal_pos_mask"] = torch.cat(
                [
                    ~cur_target["target_ind_matching"],
                    torch.tensor([True] * cur_target["num_FPs"]).to(device),
                    torch.tensor([False] * num_queries).to(device),
                ]
            ).bool()

            assert (
                cur_target["track_queries_TP_mask"].sum()
                == len(cur_target["track_query_match_ids"])
            )

        ind_out = ind_out[ind_keep]
        ind_tgt = ind_tgt[ind_keep]

        # compacter ind_tgt (sans trous)
        while not torch.arange(len(ind_tgt))[:, None].eq(ind_tgt[None]).any(0).all():
            for i in range(len(ind_tgt)):
                if i not in ind_tgt:
                    ind_tgt[ind_tgt > i] = ind_tgt[ind_tgt > i] - 1

        indices[t] = (ind_out, ind_tgt)
        targets[t][training_method]["man_track"] = man_track

    return targets, indices
