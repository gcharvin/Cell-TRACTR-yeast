import torch
from .misc import man_track_ids
from .box_ops import combine_div_boxes, calc_iou, combine_div_masks, divide_mask, divide_box, generalized_box_iou, box_cxcywh_to_xyxy
import os

# Active le debug si FLEX_DIV_DEBUG=1 dans l’environnement
DEBUG_FLEX_DIV = os.environ.get("FLEX_DIV_DEBUG", "0") == "1"

def dbg(*args, **kwargs):
    if DEBUG_FLEX_DIV:
        print(*args, **kwargs, flush=True)



def update_early_or_late_track_divisions(
    outputs,
    targets,
    training_method,
    prev_target_name,
    cur_target_name,
    fut_target_name,
):
    """
    - Sécurise les invariants *_orig avant chaque remapping man_track_ids
    - Corrige "late division" (GT=division mais la prédiction préfère 1 cellule)
    - Corrige "early division" (GT=1 cellule mais la prédiction préfère division)
    - Ajoute des logs détaillés si l'env FLEX_DIV_DEBUG=1
    """
    import os
    DEBUG_FLEX_DIV = os.environ.get("FLEX_DIV_DEBUG", "0") == "1"
    DIV_PROB_THR  = float(os.getenv("FLEX_DIV_EARLY_PROB", "0.30"))  # proba 'division' mini
    IOU_THR_EARLY = float(os.getenv("FLEX_DIV_EARLY_IOU",  "0.40"))  # IoU mini de la boîte divisée
    MAX_EARLY_PER_FRAME = int(os.getenv("FLEX_DIV_EARLY_MAX_PER_FRAME", "2"))


    def dbg(*args, **kwargs):
        if DEBUG_FLEX_DIV:
            print(*args, **kwargs, flush=True)

    device = outputs['pred_logits'].device
    use_masks = 'masks' in targets[0][training_method][cur_target_name]

    # ---------- helpers pour fiabiliser les appels à man_track_ids ----------
    def _align_orig_1to1(tgt):
        """Met *_orig en correspondance 1:1 avec l'état courant (aucune extension)."""
        tgt['boxes_orig']              = tgt['boxes'].clone()
        tgt['labels_orig']             = tgt['labels'].clone()
        tgt['track_ids_orig']          = tgt['track_ids'].clone()
        tgt['flexible_divisions_orig'] = tgt['flexible_divisions'].clone()
        tgt['is_touching_edge_orig']   = tgt['is_touching_edge'].clone()
        if use_masks:
            tgt['masks_orig'] = tgt['masks'].clone()

    def _expand_orig_for_divisions(tgt):
        """
        Étend *_orig en mettant chaque division sur 2 lignes "single":
        - ligne i : uniquement la 1ʳᵉ moitié (b[:4], b[4:]=0)
        - ligne ajoutée : uniquement la 2ᵉ moitié (b[4:], b[:4]=0)
        Ainsi, après reset depuis *_orig, on a bien (boxes[:,-1] > 0).sum() == 0.
        """
        nb = tgt['boxes'].shape[0]

        boxes_o, labels_o, tids_o = [], [], []
        flex_o, edge_o = [], []
        if use_masks:
            masks_o = []

        for i in range(nb):
            b = tgt['boxes'][i]

            # 1) ligne "first half only"
            first_half = torch.cat((b[:4], torch.zeros_like(b[:4])))
            boxes_o.append(first_half)
            labels_o.append(tgt['labels'][i].clone())
            tids_o.append(tgt['track_ids'][i].clone())
            # on garde le flag flexible tel quel pour la 1ʳᵉ ligne
            flex_o.append(tgt['flexible_divisions'][i].clone())
            edge_o.append(tgt['is_touching_edge'][i].clone())
            if use_masks:
                m = tgt['masks'][i]
                if m.ndim == 3 and m.size(0) >= 1:
                    first_mask = torch.cat((m[:1], torch.zeros_like(m[:1])), dim=0)
                else:
                    first_mask = m.clone()
                masks_o.append(first_mask)

            # 2) si division, ajouter la ligne "second half only"
            if (b[-4:] != 0).any():
                second_half = torch.cat((b[4:], torch.zeros_like(b[:4])))
                boxes_o.append(second_half)
                labels_o.append(tgt['labels'][i].clone())
                tids_o.append(tgt['track_ids'][i].clone())
                flex_o.append(torch.tensor(True, device=b.device))  # la ligne ajoutée est issue d’une division
                edge_o.append(tgt['is_touching_edge'][i].clone())
                if use_masks:
                    if m.ndim == 3 and m.size(0) >= 2:
                        second_mask = torch.cat((m[1:2], torch.zeros_like(m[:1])), dim=0)
                    else:
                        second_mask = m.clone()
                    masks_o.append(second_mask)

        tgt['boxes_orig']              = torch.stack(boxes_o,  dim=0) if boxes_o else tgt['boxes'].clone()
        tgt['labels_orig']             = torch.stack(labels_o, dim=0) if labels_o else tgt['labels'].clone()
        tgt['track_ids_orig']          = torch.stack(tids_o,   dim=0) if tids_o else tgt['track_ids'].clone()
        tgt['flexible_divisions_orig'] = torch.stack(flex_o,   dim=0) if flex_o else tgt['flexible_divisions'].clone()
        tgt['is_touching_edge_orig']   = torch.stack(edge_o,   dim=0) if edge_o else tgt['is_touching_edge'].clone()
        if use_masks:
            tgt['masks_orig'] = torch.stack(masks_o, dim=0) if masks_o else tgt['masks'].clone()

        
    def _estimate_removed(prev_tgt, cur_tgt):
        """
        Estime le nb de cellules 'supprimées' de prev->cur par différence de track_ids.
        Robuste même si N=0 d'un côté.
        """
        prev_ids = set(prev_tgt['track_ids'].detach().cpu().tolist()) if prev_tgt['track_ids'].numel() else set()
        cur_ids  = set(cur_tgt['track_ids'].detach().cpu().tolist())  if cur_tgt['track_ids'].numel()  else set()
        return sum(1 for tid in prev_ids if tid not in cur_ids)


    def _pad_missing_removed(tgt, missing, use_masks, template_from=None, extra_track_ids=None):
        """
        Add `missing` dummy rows to *_orig so that
        len(boxes_orig) == len(boxes) + #divs + (#new_cell_ids if present).
        If `extra_track_ids` is provided (e.g. new_cell_ids), use them for track_ids_orig.
        """
        if missing <= 0:
            return

        src = template_from if template_from is not None else tgt
        dev = tgt['boxes'].device

        # --- BOXES ---
        dim_box = (tgt['boxes'].shape[1] if tgt['boxes'].ndim == 2
                else (src['boxes_orig'].shape[1] if 'boxes_orig' in src and src['boxes_orig'].ndim == 2 else 8))
        boxes_pad = torch.zeros((missing, dim_box), dtype=tgt['boxes'].dtype, device=dev)

        # --- LABELS ---
        dim_lab = (tgt['labels'].shape[1] if tgt['labels'].ndim == 2
                else (src['labels_orig'].shape[1] if 'labels_orig' in src and src['labels_orig'].ndim == 2 else 2))
        labels_pad = torch.zeros((missing, dim_lab), dtype=tgt['labels'].dtype, device=dev)

        # --- TRACK IDS ---
        if extra_track_ids is not None and len(extra_track_ids) > 0:
            fill = list(extra_track_ids)[:missing]
            if len(fill) < missing:
                fill += [-1] * (missing - len(fill))
            tids_pad = torch.tensor(fill, dtype=tgt['track_ids'].dtype, device=dev)
        else:
            tids_pad = torch.full((missing,), -1, dtype=tgt['track_ids'].dtype, device=dev)

        # --- FLEX / EDGE ---
        flex_pad = torch.zeros((missing,), dtype=tgt['flexible_divisions'].dtype, device=dev)
        edge_pad = torch.zeros((missing,), dtype=tgt['is_touching_edge'].dtype, device=dev)

        # --- MASKS (optional) ---
        if use_masks:
            if tgt['masks'].numel() > 0 and tgt['masks'].ndim == 4:
                C, H, W = tgt['masks'].shape[1:]
                mdtype = tgt['masks'].dtype
            elif 'masks_orig' in src and src['masks_orig'].numel() > 0 and src['masks_orig'].ndim == 4:
                C, H, W = src['masks_orig'].shape[1:]
                mdtype = src['masks_orig'].dtype
            else:
                C, H, W, mdtype = 1, 1, 1, torch.float32
            masks_pad = torch.zeros((missing, C, H, W), dtype=mdtype, device=dev)

        # --- CONCAT ---
        tgt['boxes_orig']              = torch.cat([tgt['boxes_orig'], boxes_pad],   dim=0)
        tgt['labels_orig']             = torch.cat([tgt['labels_orig'], labels_pad], dim=0)
        tgt['track_ids_orig']          = torch.cat([tgt['track_ids_orig'], tids_pad], dim=0)
        tgt['flexible_divisions_orig'] = torch.cat([tgt['flexible_divisions_orig'], flex_pad], dim=0)
        tgt['is_touching_edge_orig']   = torch.cat([tgt['is_touching_edge_orig'], edge_pad], dim=0)
        if use_masks:
            tgt['masks_orig'] = torch.cat([tgt['masks_orig'], masks_pad], dim=0)


    def _safe_remap(ts, name_a, name_b):
        """
        Appelle man_track_ids en garantissant l'invariant attendu :
        len(X.boxes_orig) == len(X.boxes) + #divisions(X) + len(X.new_cell_ids)
        pour X ∈ {côté input A=name_a, côté output B=name_b}.

        Stratégie :
        1) on remet *_orig en 1:1 puis on "expand" les divisions (A et B) ;
        2) on pad A/B en tenant compte des divisions et de new_cell_ids (dn_object) ;
        3) on réessaie man_track_ids ;
        4) en dernier recours, boucle de padding (A & B) 1 par 1 (borne 32 essais).

        Remarque :
        - Cette fonction s'appuie sur _align_orig_1to1, _expand_orig_for_divisions et
            _pad_missing_removed définies dans le même module.
        - 'training_method', 'use_masks' et 'dbg' sont disponibles dans le scope parent.
        """
        from .misc import man_track_ids

        # 1) Aligner *_orig et développer les divisions pour chaque item du batch
        for _t in ts:
            _align_orig_1to1(_t[training_method][name_a])
            _align_orig_1to1(_t[training_method][name_b])
            _expand_orig_for_divisions(_t[training_method][name_a])
            _expand_orig_for_divisions(_t[training_method][name_b])

        A0 = ts[0][training_method][name_a]
        B0 = ts[0][training_method][name_b]

        # Première tentative directe
        try:
            return man_track_ids(ts, training_method, name_a, name_b)
        except AssertionError:
            pass  # on va corriger via padding, puis retenter

        # 2) Prise en compte explicite des divisions + new_cell_ids (si dn_object)
        for _t in ts:
            A = _t[training_method][name_a]
            B = _t[training_method][name_b]

            # --- B (output side) ---
            n_boxes_B = B['boxes'].shape[0]
            n_divs_B  = int((B['boxes'][:, -1] > 0).sum().item()) if B['boxes'].numel() else 0
            new_out   = len(B['new_cell_ids']) if 'new_cell_ids' in B else 0
            need_B    = n_boxes_B + n_divs_B + new_out
            have_B    = B['boxes_orig'].shape[0]

            if have_B < need_B:
                missing = need_B - have_B
                dbg(f"[flex_div][_safe_remap] pad B: have={have_B} need={need_B} (+{missing})")
                _pad_missing_removed(
                    B, missing, use_masks, template_from=A,
                    extra_track_ids=(B['new_cell_ids'] if 'new_cell_ids' in B else None)
                )

            # --- A (input side) ---
            n_boxes_A = A['boxes'].shape[0]
            n_divs_A  = int((A['boxes'][:, -1] > 0).sum().item()) if A['boxes'].numel() else 0
            new_in    = len(A['new_cell_ids']) if 'new_cell_ids' in A else 0
            need_A    = n_boxes_A + n_divs_A + new_in
            have_A    = A['boxes_orig'].shape[0]

            if have_A < need_A:
                missing = need_A - have_A
                dbg(f"[flex_div][_safe_remap] pad A: have={have_A} need={need_A} (+{missing})")
                _pad_missing_removed(
                    A, missing, use_masks, template_from=B,
                    extra_track_ids=(A['new_cell_ids'] if 'new_cell_ids' in A else None)
                )

        # 3) Deuxième tentative après padding ciblé
        try:
            return man_track_ids(ts, training_method, name_a, name_b)
        except AssertionError:
            pass

        # 4) Dernier recours : on ajoute 1 dummy à A & B à chaque essai (max 32)
        for _try in range(32):
            for _t in ts:
                A = _t[training_method][name_a]
                B = _t[training_method][name_b]
                _pad_missing_removed(
                    B, 1, use_masks, template_from=A,
                    extra_track_ids=(B['new_cell_ids'] if 'new_cell_ids' in B else None)
                )
                _pad_missing_removed(
                    A, 1, use_masks, template_from=B,
                    extra_track_ids=(A['new_cell_ids'] if 'new_cell_ids' in A else None)
                )
            try:
                return man_track_ids(ts, training_method, name_a, name_b)
            except AssertionError:
                continue

        # Toujours KO : on donne du contexte pour débug
        raise AssertionError(
            "[safe_remap] invariant still failing after padding. "
            f"A: boxes={A0['boxes'].shape} orig={A0['boxes_orig'].shape} "
            f"B: boxes={B0['boxes'].shape} orig={B0['boxes_orig'].shape}"
        )



    # check for early / late cell division and adjust ground truths as necessary
    for t, target in enumerate(targets):

        man_track  = target[training_method]['man_track']
        prev_target = target[training_method][prev_target_name]
        cur_target  = target[training_method][cur_target_name]
        fut_target  = target[training_method][fut_target_name]

        if cur_target['empty']:
            continue

        # logs header
        img_id = cur_target.get("image_id", -1)
        frame  = cur_target.get("framenb",  -1)
        n_late_fix = 0
        n_early_fix = 0
        n_skip_nomother = 0

        dbg(f"[flex_div][start] img={img_id} frame={frame} "
            f"prev={prev_target['boxes'].shape[0]}/{prev_target['boxes_orig'].shape[0]} "
            f"cur={cur_target['boxes'].shape[0]}/{cur_target['boxes_orig'].shape[0]}")

        # Avant toute correspondance prev->cur, on synchronise *_orig et on remap
      #  _align_orig_1to1(prev_target)
      #  _align_orig_1to1(cur_target)
      
        targets = _safe_remap(targets, prev_target_name, cur_target_name)

        if 'track_query_match_ids' in cur_target:

            # Prédictions pour track-queries correctement matchées
            pred_boxes_track  = outputs['pred_boxes'][t][cur_target['track_queries_TP_mask']].detach()
            pred_logits_track = outputs['pred_logits'][t][cur_target['track_queries_TP_mask']].sigmoid().detach()

            # Late fixes opèrent sur la paire prev->cur remappée
            boxes     = cur_target['boxes'].clone()
            track_ids = cur_target['track_ids'].clone()

            for p, pred_box in enumerate(pred_boxes_track):
                box = boxes[cur_target['track_query_match_ids'][p]].clone()
                track_id = track_ids[cur_target['track_query_match_ids'][p]].clone()

                # robustesse : si la mère n'est pas dans prev, on log et on skip
                if track_id not in prev_target['track_ids']:
                    n_skip_nomother += 1
                    dbg(f"[flex_div][skip_no_mother] img={img_id} fr={frame} track_id={int(track_id)}")
                    continue

                # si la cellule était déjà "flexible_divisions" à t-1, on ne force rien
                if prev_target['flexible_divisions'][prev_target['track_ids'] == track_id]:
                    continue

                # ----- LATE division fix -----
                # GT=division (box[-1]>0) ; préd=single (classe cell>0.5 et division<0.5)
                if box[-1] > 0 and pred_logits_track[p, 0] > 0.5 and pred_logits_track[p, -1] < 0.5:

                    area_box_1 = box[2] * box[3]
                    area_box_2 = box[6] * box[7]

                    # si division fortement asymétrique -> on n'impose pas la fusion
                    if area_box_1 > area_box_2 * 2 or area_box_2 > area_box_1 * 2:
                        continue

                    combined_box = combine_div_boxes(box)
                    iou_div = calc_iou(box, pred_box)

                    pred_single = pred_box.clone()
                    pred_single[4:] = 0
                    iou_combined = calc_iou(combined_box, pred_single)

                    if iou_combined - iou_div > 0 and iou_combined > 0.5:
                        # on remplace la division GT par une boîte combinée (single)
                        cur_idx = cur_target['track_query_match_ids'][p]

                        cur_target['boxes'][cur_idx]   = combined_box
                        cur_target['labels'][cur_idx]  = torch.tensor([0, 1], device=device)
                        cur_target['flexible_divisions'][cur_idx] = True

                        # maj *_orig : garder la 1re moitié, retirer la 2e
                        div_bool_1 = cur_target['boxes_orig'][:, :4].eq(box[None, :4]).all(1)
                        div_bool_2 = cur_target['boxes_orig'][:, :4].eq(box[None, 4:]).all(1)

                        div_track_id_1 = cur_target['track_ids_orig'][div_bool_1].clone()
                        div_track_id_2 = cur_target['track_ids_orig'][div_bool_2].clone()

                        cur_target['boxes_orig'][div_bool_1] = combined_box.clone()
                        cur_target['boxes_orig']             = cur_target['boxes_orig'][~div_bool_2]

                        cur_target['track_ids_orig'][div_bool_1] = cur_target['track_ids'][cur_idx].clone()
                        cur_target['track_ids_orig']             = cur_target['track_ids_orig'][~div_bool_2]

                        cur_target['flexible_divisions_orig'][div_bool_1] = True
                        cur_target['flexible_divisions_orig']             = cur_target['flexible_divisions_orig'][~div_bool_2]

                        # edge flag (bool propre)
                        e1 = cur_target['is_touching_edge_orig'][div_bool_1].flatten()
                        e2 = cur_target['is_touching_edge_orig'][div_bool_2].flatten()
                        edge_flag = bool((e1.any() or e2.any()))
                        cur_target['is_touching_edge'][cur_idx]      = edge_flag
                        cur_target['is_touching_edge_orig'][div_bool_1] = torch.tensor(edge_flag, device=device)
                        cur_target['is_touching_edge_orig']             = cur_target['is_touching_edge_orig'][~div_bool_2]

                        assert cur_target['labels_orig'][div_bool_1, 1].all().item() == 1
                        cur_target['labels_orig'] = cur_target['labels_orig'][~div_bool_2]

                        if use_masks:
                            mask = cur_target['masks'][cur_idx]
                            prev_mask = prev_target['masks'][prev_target['track_ids'] == track_id][0]
                            combined_mask = combine_div_masks(mask, prev_mask)
                            cur_target['masks'][cur_idx] = combined_mask

                            cur_target['masks_orig'][div_bool_1] = combined_mask
                            cur_target['masks_orig'] = cur_target['masks_orig'][~div_bool_2]

                        # man_track bookkeeping
                        track_id_ind     = man_track[:, 0] == track_id
                        div_track_id_1_ind = man_track[:, 0] == div_track_id_1
                        div_track_id_2_ind = man_track[:, 0] == div_track_id_2

                        man_track[track_id_ind, 2]     += 1
                        man_track[div_track_id_1_ind, 1] += 1
                        man_track[div_track_id_2_ind, 1] += 1

                        # si une des filles sort du FOV juste après
                        if (man_track[div_track_id_1_ind, 1] > man_track[div_track_id_1_ind, 2]
                            or man_track[div_track_id_2_ind, 1] > man_track[div_track_id_2_ind, 2]):
                            man_track[track_id_ind, 2] = torch.max(
                                man_track[div_track_id_1_ind, 2], man_track[div_track_id_2_ind, 2]
                            )
                            man_track[div_track_id_1_ind, 1:] = -1
                            man_track[div_track_id_2_ind, 1:] = -1

                            # mettre à jour les ids du futur si besoin
                            if div_track_id_1 in fut_target['track_ids'] and div_track_id_2 in fut_target['track_ids']:
                                raise NotImplementedError
                            elif div_track_id_1 in fut_target['track_ids']:
                                fut_target['track_ids'][fut_target['track_ids'] == div_track_id_1] = track_id.long().to(device)
                                if fut_target_name != 'fut_target' and div_track_id_1 in target[training_method]['fut_target']['track_ids']:
                                    target[training_method]['fut_target']['track_ids'][target[training_method]['fut_target']['track_ids'] == div_track_id_1] = track_id.long().to(device)
                            elif div_track_id_2 in fut_target['track_ids']:
                                fut_target['track_ids'][fut_target['track_ids'] == div_track_id_2] = track_id.long().to(device)
                                if fut_target_name != 'fut_target' and div_track_id_2 in target[training_method]['fut_target']['track_ids']:
                                    target[training_method]['fut_target']['track_ids'][target[training_method]['fut_target']['track_ids'] == div_track_id_2] = track_id.long().to(device)

                            fut_target['track_ids_orig'] = fut_target['track_ids'].clone()
                            if fut_target_name != 'fut_target':
                                target[training_method]['fut_target']['track_ids_orig'] = target[training_method]['fut_target']['track_ids'].clone()

                            # cas spé de dataset (division sur deux frames de suite)
                            if div_track_id_1 in man_track[:, -1] and div_track_id_2 in man_track[:, -1]:
                                fut_div_track_id_1, fut_div_track_id_2 = man_track[(man_track[:, -1] == div_track_id_1), 0]
                                man_track[man_track[:, 0] == fut_div_track_id_1, -1] = 0
                                man_track[man_track[:, 0] == fut_div_track_id_2, -1] = 0
                                fut_div_track_id_1, fut_div_track_id_2 = man_track[(man_track[:, -1] == div_track_id_2), 0]
                                man_track[man_track[:, 0] == fut_div_track_id_1, -1] = 0
                                man_track[man_track[:, 0] == fut_div_track_id_2, -1] = 0
                            elif div_track_id_1 in man_track[:, -1]:
                                fut_div_track_id_1, fut_div_track_id_2 = man_track[(man_track[:, -1] == div_track_id_1), 0]
                                man_track[man_track[:, 0] == fut_div_track_id_1, -1] = track_id
                                man_track[man_track[:, 0] == fut_div_track_id_2, -1] = track_id
                            elif div_track_id_2 in man_track[:, -1]:
                                fut_div_track_id_1, fut_div_track_id_2 = man_track[(man_track[:, -1] == div_track_id_2), 0]
                                man_track[man_track[:, 0] == fut_div_track_id_1, -1] = track_id
                                man_track[man_track[:, 0] == fut_div_track_id_2, -1] = track_id

                            assert (torch.arange(1, target[training_method]['man_track'].shape[0] + 1,
                                                 dtype=target[training_method]['man_track'].dtype,
                                                 device=target[training_method]['man_track'].device)
                                    == target[training_method]['man_track'][:, 0]).all()

                        n_late_fix += 1
                        dbg(f"[flex_div][late] img={img_id} fr={frame} track_id={int(track_id)} "
                            f"iou_div={float(iou_div):.3f} iou_combined={float(iou_combined):.3f}")

            # Remap cur->fut avant les early fixes (mêmes raisons)
            targets = _safe_remap(targets, cur_target_name, fut_target_name)

            # ----- EARLY division fix -----
            # GT=single (box[-1]==0) ; préd=division (toutes probas > 0.5)
# ----- EARLY division fix (bornée & seuils configurables) -----
            DIV_PROB_THR       = float(os.getenv("FLEX_DIV_EARLY_PROB", "0.30"))  # proba 'division' min
            IOU_THR_EARLY      = float(os.getenv("FLEX_DIV_EARLY_IOU",  "0.40"))  # IoU(div_box,pred) min
            MAX_EARLY_PER_FRAME= int(os.getenv("FLEX_DIV_EARLY_MAX_PER_FRAME", "2"))

            early_count = 0
            for p in range(pred_boxes_track.shape[0]):
                if early_count >= MAX_EARLY_PER_FRAME:
                    break

                cur_idx = cur_target['track_query_match_ids'][p]
                box     = boxes[cur_idx].clone()                 # GT au frame courant
                div_logit = float(pred_logits_track[p, -1].item())

                # Conditions d’amorçage : GT "single" + proba division suffisante
                if float(box[-1].item()) != 0.0 or div_logit <= DIV_PROB_THR:
                    continue

                # track_id associé
                track_id = track_ids[cur_idx].clone()
                track_id_ind = (man_track[:, 0] == track_id)

                # Si la cellule vient d'apparaître à t (apparition au frame prev), remonter à la mère
                if track_id_ind.any() and int(man_track[track_id_ind, 2].item()) == int(prev_target['framenb']):
                    track_id = man_track[track_id_ind, -1].clone()  # mother id

                # La cellule (ou sa mère) doit exister à t+1 et y être marquée "division"
                if not (fut_target['track_ids'] == track_id).any():
                    continue
                fut_inds = torch.nonzero(fut_target['track_ids'] == track_id, as_tuple=False).flatten()
                if fut_inds.numel() == 0:
                    continue
                fut_box = fut_target['boxes'][int(fut_inds[0].item())]
                if float(fut_box[-1].item()) <= 0.0:
                    continue

                # Construire la boîte divisée candidate et comparer les IoU
                div_box   = divide_box(box, fut_box)
                iou_div   = calc_iou(div_box,        pred_boxes_track[p])
                iou_single= calc_iou(box[:4],        pred_boxes_track[p][:4])

                if (iou_div - iou_single) <= 0.0 or iou_div <= IOU_THR_EARLY:
                    continue

                # On déclenche l'EARLY fix : on convertit la GT en "division"
                cur_target['boxes'][cur_idx]  = div_box
                cur_target['labels'][cur_idx] = torch.tensor([0, 0], device=device)
                # traçabilité côté *_orig
                if 'flexible_divisions_orig' in cur_target:
                    cur_target['flexible_divisions_orig'][cur_idx] = True

                # Récupérer les deux track_ids filles attendues à t+1 via man_track
                sel_next = (man_track[:, -1] == track_id)
                if sel_next.sum() != 2:
                    # si l'info n'est pas disponible proprement, on skip sans casser
                    dbg(f"[early-skip] daughters not found for mother={int(track_id)}")
                    continue

                fut_track_id_1, fut_track_id_2 = man_track[sel_next, 0]
                # Associer 1↔1 / 2↔2 selon la proximité
                fut_box_1 = fut_target['boxes_orig'][fut_target['track_ids_orig'] == fut_track_id_1][0]
                fut_box_2 = fut_target['boxes_orig'][fut_target['track_ids_orig'] == fut_track_id_2][0]
                if ((div_box[:2] - fut_box_1[:2]).square().sum()
                    + (div_box[4:6] - fut_box_2[:2]).square().sum()
                    > (div_box[:2] - fut_box_2[:2]).square().sum()
                    + (div_box[4:6] - fut_box_1[:2]).square().sum()):
                    fut_track_id_1, fut_track_id_2 = fut_track_id_2, fut_track_id_1

                # Mettre à jour *_orig (dupliquer la ligne et affecter les deux track_ids filles)
                ind_tgt_orig = torch.where(cur_target['boxes_orig'].eq(box).all(-1))[0][0]
                cur_target['boxes_orig'][ind_tgt_orig, :4] = div_box[:4]
                cur_target['boxes_orig'] = torch.cat(
                    (cur_target['boxes_orig'],
                    torch.cat((div_box[4:], torch.zeros_like(div_box[4:])))[None]),
                    dim=0
                )
                cur_target['track_ids_orig'][ind_tgt_orig] = fut_track_id_1
                cur_target['track_ids_orig'] = torch.cat(
                    (cur_target['track_ids_orig'], torch.tensor([fut_track_id_2], device=device))
                )
                cur_target['labels_orig'] = torch.cat(
                    (cur_target['labels_orig'], cur_target['labels_orig'][:1]), dim=0
                )
                cur_target['flexible_divisions_orig'] = torch.cat(
                    (cur_target['flexible_divisions_orig'], torch.tensor([True], device=device))
                )
                cur_target['is_touching_edge_orig'] = torch.cat(
                    (cur_target['is_touching_edge_orig'],
                    cur_target['is_touching_edge_orig'][ind_tgt_orig][None])
                )

                # Mises à jour des masques si présents
                if use_masks:
                    mask     = cur_target['masks'][cur_idx]
                    fut_mask = fut_target['masks'][int(fut_inds[0].item())]
                    div_mask = divide_mask(mask, fut_mask)
                    cur_target['masks'][cur_idx] = div_mask
                    cur_target['masks_orig'][ind_tgt_orig, :1] = div_mask[:1]
                    cur_target['masks_orig'] = torch.cat(
                        (cur_target['masks_orig'],
                        torch.cat((div_mask[1:], torch.zeros_like(div_mask[1:])))[None]),
                        dim=0
                    )

                # Comptabilité man_track (réduit la durée de la mère, augmente les filles)
                fut_track_id_1_ind = (man_track[:, 0] == fut_track_id_1)
                fut_track_id_2_ind = (man_track[:, 0] == fut_track_id_2)
                # track_id_ind est celui d'avant (peut être mis à jour si on a remonté à la mère)
                track_id_ind = (man_track[:, 0] == track_id)

                man_track[fut_track_id_1_ind, 1] -= 1
                man_track[fut_track_id_2_ind, 1] -= 1
                man_track[track_id_ind,      2] -= 1

                if man_track[track_id_ind, 1] > man_track[track_id_ind, 2]:
                    man_track[track_id_ind, 1:] = -1
                    man_track[fut_track_id_1_ind, -1] = 0
                    man_track[fut_track_id_2_ind, -1] = 0

                early_count += 1
                n_early_fix += 1
                dbg(f"[early] div_logit={div_logit:.3f} iou_div={float(iou_div):.3f} iou_single={float(iou_single):.3f} "
                    f"cur_idx={int(cur_idx)} mother={int(track_id)} daughters=({int(fut_track_id_1)},{int(fut_track_id_2)})")


        # sauvegarde man_track mis à jour pour cet item
        targets[t][training_method]['man_track'] = man_track

        dbg(f"[flex_div][done] img={img_id} frame={frame} "
            f"late={n_late_fix} early={n_early_fix} skip_no_mother={n_skip_nomother} "
            f"cur={cur_target['boxes'].shape[0]}/{cur_target['boxes_orig'].shape[0]} "
            f"fut={fut_target['boxes'].shape[0]}/{fut_target['boxes_orig'].shape[0]}")

    # NOTE: le bloc de "refresh" des masks track_queries en fin de fonction
    # garde la même structure que le code d’origine (hors boucle) :
   # --- REFRESH TRACK-QUERIES (robuste) ---
    if 'track_query_match_ids' in targets[-1][training_method][cur_target_name]:
        target = targets[-1]
        prev_target = target[training_method][prev_target_name]
        cur_target  = target[training_method][cur_target_name]

        # Remap cohérent prev->cur
        targets = _safe_remap(targets, prev_target_name, cur_target_name)

        device = outputs['pred_logits'].device

        # 1) Récupérer prev_track_ids de façon défensive
        prev_ids_raw = prev_target.get('track_ids', torch.tensor([], device=device))
        if not isinstance(prev_ids_raw, torch.Tensor):
            prev_ids_raw = torch.as_tensor(prev_ids_raw, device=device)

        use_subset = (
            'prev_ind' in cur_target
            and isinstance(cur_target['prev_ind'], (list, tuple))
            and len(cur_target['prev_ind']) > 1
            and cur_target['prev_ind'][1] is not None
            and torch.numel(cur_target['prev_ind'][1]) > 0
        )
        if use_subset:
            idx = cur_target['prev_ind'][1].to(prev_ids_raw.device)
            prev_track_ids = prev_ids_raw[idx]
        else:
            prev_track_ids = prev_ids_raw

        # 2) Matching prev->cur (gère les cas vides + un seul match par ligne)
        if prev_track_ids.numel() == 0 or cur_target['track_ids'].numel() == 0:
            cur_target['target_ind_matching']   = torch.zeros((prev_track_ids.shape[0],), dtype=torch.bool, device=device)
            cur_target['track_query_match_ids'] = torch.tensor([], dtype=torch.long, device=device)
        else:
            mat = prev_track_ids.unsqueeze(1).eq(cur_target['track_ids'])
            if mat.numel() > 0:
                first_true = mat.int().cumsum(dim=1) == 1
                mat = mat & first_true
            cur_target['target_ind_matching'] = mat.any(dim=1)
            nz = mat.nonzero(as_tuple=False)
            cur_target['track_query_match_ids'] = (nz[:, 1] if nz.numel()
                                                else torch.tensor([], dtype=torch.long, device=device))

        # 3) Reconstruire les masques
        num_queries_local = ((~cur_target['track_queries_mask']).sum()
                            if 'track_queries_mask' in cur_target
                            else torch.tensor(0, device=device))
        tq_mask = torch.ones_like(cur_target['target_ind_matching']).bool()

        cur_target['track_queries_mask'] = torch.cat([
            tq_mask,
            torch.tensor([True]  * int(cur_target['num_FPs']), device=device),
            torch.tensor([False] * int(num_queries_local),     device=device),
        ]).bool()

        cur_target['track_queries_TP_mask'] = torch.cat([
            cur_target['target_ind_matching'],
            torch.tensor([False] * int(cur_target['num_FPs']), device=device),
            torch.tensor([False] * int(num_queries_local),     device=device),
        ]).bool()

        cur_target['track_queries_fal_pos_mask'] = torch.cat([
            ~cur_target['target_ind_matching'],
            torch.tensor([True]  * int(cur_target['num_FPs']), device=device),
            torch.tensor([False] * int(num_queries_local),     device=device),
        ]).bool()

        # Sanity check (optionnel). Garde-le si tu veux, sinon commente-le.
        tp_cnt = int(cur_target['track_queries_TP_mask'].sum().item())
        assert tp_cnt == int(cur_target['track_query_match_ids'].numel()), \
            f"TP_mask={tp_cnt} != len(match_ids)={int(cur_target['track_query_match_ids'].numel())}"


    return targets




def update_object_detection(
    outputs,
    targets,
    indices,
    num_queries,
    training_method,
    prev_target_name,
    cur_target_name,
    fut_target_name,):

    import torch  # sécurité si pas importé en haut

    # --- helper robuste -----------------------------------------------------
    def _find_idx_by_track_id(tgt, track_id):
        """
        Retourne l'indice (int) du track_id dans tgt['track_ids'] ou None.
        """
        if tgt is None or ('track_ids' not in tgt):
            return None
        idx = torch.nonzero(tgt['track_ids'] == track_id, as_tuple=False).flatten()
        return int(idx[0]) if idx.numel() else None
    # ------------------------------------------------------------------------

    N = outputs['pred_logits'].shape[1]
    use_masks = 'masks' in targets[0][training_method][cur_target_name]
    device = outputs['pred_logits'].device

    # Indicies are saved in targets for calcualting object detction / tracking accuracy
    for t,(target,(ind_out,ind_tgt)) in enumerate(zip(targets,indices)):

        prev_target = target[training_method][prev_target_name]
        cur_target  = target[training_method][cur_target_name]
        fut_target  = target[training_method][fut_target_name]

        if cur_target['empty']:
            continue

        if training_method == 'dn_object':
            cur_target['track_queries_mask'] = torch.zeros_like(cur_target['track_queries_mask']).bool()

        man_track = target[training_method]['man_track']
        framenb   = cur_target['framenb']

        skip = []  # If a GT cell is split into two cells, we want to skip the second cell
        ind_keep = torch.tensor([True for _ in range(len(ind_tgt))]).bool()

        for ind_out_i, ind_tgt_i in zip(ind_out, ind_tgt):
            # Confirm prediction is an object query, not a track query
            if ind_out_i >= (N - num_queries) and ind_tgt_i not in skip:
                if 'track_queries_mask' in cur_target:
                    assert not cur_target['track_queries_mask'][ind_out_i]
                track_id = cur_target['track_ids'][ind_tgt_i].clone()

                track_id_ind = man_track[:, 0] == track_id
                # === Cas "vient de diviser" ============================================
                if man_track[track_id_ind, 1] == framenb and man_track[track_id_ind, -1] > 0:

                    # --- Mère (branche 2 en budding) ---
                    mother_id = man_track[track_id_ind, -1].clone().long()
                    assert mother_id in prev_target['track_ids']  # la mère existe à t-1
                    track_id_1 = track_id  # fille (apparue à t)

                    # Relabel variables to be consistent with cell 1 & 2
                    ind_tgt_1 = ind_tgt_i
                    ind_out_1 = ind_out_i
                    ind_1 = torch.where(ind_out == ind_out_1)[0][0]

                    # >>> BUDDING: toujours prendre la MÈRE comme "branche 2"
                    used_mother_fallback = True    # indique qu'on ne fait PAS la logique "2 filles -> 1"
                    mother_from_prev = False

                    # cherche la mère à t
                    mother_idx_cur = torch.nonzero(cur_target['track_ids'] == mother_id, as_tuple=False).flatten()
                    if mother_idx_cur.numel() > 0:
                        ind_tgt_2 = int(mother_idx_cur[0])
                        # s'assurer que la mère est bien matchée (présente dans ind_tgt)
                        hit2 = torch.nonzero(ind_tgt == ind_tgt_2, as_tuple=False).flatten()
                        if hit2.numel() == 0:
                            # mère non matchée par le matcher -> pas exploitable proprement
                            continue
                        ind_2 = int(hit2[0])
                        ind_out_2 = ind_out[ind_2]
                    else:
                        # fallback: utiliser la mère à t-1 si absente à t
                        mother_from_prev = True
                        mother_idx_prev = torch.nonzero(prev_target['track_ids'] == mother_id, as_tuple=False).flatten()
                        if mother_idx_prev.numel() == 0:
                            # pas de mère exploitable -> skip proprement cet event
                            continue
                        # pas d'ind_2 / ind_out_2 utilisables au frame courant
                    # <<< BUDDING ---------------------------------------------------------

                    # Récupère prédictions & GT pour la fille (et potentiellement la mère si présente à t)
                    pred_box_1 = outputs['pred_boxes'][t, ind_out_1].detach()
                    box_1      = cur_target['boxes'][ind_tgt_1]
                    assert box_1[-1] == 0, 'Cells have just divided. Each box should contain just one cell'

                    # Si la mère est bien présente à t (et matchée), on pourrait lire ses boxes,
                    # mais en budding on NE lance PAS la logique "combine 2->1".
                    if not mother_from_prev:
                        box_2     = cur_target['boxes'][ind_tgt_2]
                        pred_box_2= outputs['pred_boxes'][t, ind_out_2].detach()
                        assert box_2[-1] == 0, 'Cells have just divided. Each box should contain just one cell'

                    # >>> BUDDING: on n'entre pas dans la logique "2 filles -> 1 box"
                    # car en levure il n’y a qu’UNE fille + la mère qui continue.
                    # On évite donc tout le bloc "combined_box / iou_combined" ci-dessous.
                    if used_mother_fallback:
                        continue

                    # ---------------------------------------------------------------------
                    # (logique d'origine "2 filles -> 1" non utilisée en budding)
                    # ---------------------------------------------------------------------
                    boxes_1_2 = torch.cat((box_1[:4], box_2[:4]))
                    pred_boxes_1_2 = torch.cat((pred_box_1[:4], pred_box_2[:4]))

                    iou_sep, flip = calc_iou(pred_boxes_1_2, boxes_1_2, return_flip=True)

                    if flip:
                        iou_1 = calc_iou(pred_boxes_1_2[:4], boxes_1_2[4:])
                        pred_logits_1 = outputs['pred_logits'][t, ind_out_2].sigmoid()[0]
                        iou_2 = calc_iou(pred_boxes_1_2[4:], boxes_1_2[:4])
                        pred_logits_2 = outputs['pred_logits'][t, ind_out_1].sigmoid()[0]
                    else:
                        iou_1 = calc_iou(pred_boxes_1_2[:4], boxes_1_2[:4])
                        pred_logits_1 = outputs['pred_logits'][t, ind_out_1].sigmoid()[0]
                        iou_2 = calc_iou(pred_boxes_1_2[4:], boxes_1_2[4:])
                        pred_logits_2 = outputs['pred_logits'][t, ind_out_2].sigmoid()[0]

                    combined_box = combine_div_boxes(boxes_1_2)

                    potential_object_query_indices = [ind_out_id for ind_out_id in torch.arange(N-num_queries, N)
                                                      if ((ind_out_id not in ind_out or ind_out_id in [ind_out_1, ind_out_2])
                                                          and outputs['pred_logits'][t, ind_out_id, 0].sigmoid().detach() > 0.5)]

                    if len(potential_object_query_indices) == 0:
                        continue

                    potential_pred_boxes = outputs['pred_boxes'][t, potential_object_query_indices].detach()

                    iou_combined = generalized_box_iou(
                        box_cxcywh_to_xyxy(potential_pred_boxes[:, :4]),
                        box_cxcywh_to_xyxy(combined_box[None, :4]),
                        return_iou_only=True
                    )

                    max_ind = torch.argmax(iou_combined)
                    assert 0 <= iou_combined[max_ind] <= 1 and 0 <= iou_sep <= 1, 'Calc_iou out of range'

                    if (iou_combined[max_ind] - iou_sep > 0 and iou_combined[max_ind] > 0.5
                        and (iou_combined[max_ind] > iou_1 or pred_logits_1 < 0.5)
                        and (iou_combined[max_ind] > iou_2 or pred_logits_2 < 0.5)):

                        ind_out_combined = potential_object_query_indices[max_ind]

                        ind_keep[ind_2] = False
                        skip += [ind_tgt_1, ind_tgt_2]

                        cur_target['boxes'][ind_tgt_1]      = combined_box
                        cur_target['track_ids'][ind_tgt_1]  = mother_id
                        cur_target['track_ids'][ind_tgt_2]  = -1
                        cur_target['flexible_divisions'][ind_tgt_1] = True
                        cur_target['is_touching_edge'][ind_tgt_1] = (cur_target['is_touching_edge'][ind_tgt_1]
                                                                     or cur_target['is_touching_edge'][ind_tgt_2])

                        assert mother_id not in cur_target['track_ids_orig']
                        ind_tgt_orig_1 = cur_target['track_ids_orig'] == track_id_1
                        cur_target['track_ids_orig'][ind_tgt_orig_1]      = mother_id
                        cur_target['boxes_orig'][ind_tgt_orig_1]          = combined_box
                        cur_target['flexible_divisions_orig'][ind_tgt_orig_1] = True
                        cur_target['is_touching_edge_orig'][ind_tgt_orig_1]   = cur_target['is_touching_edge'][ind_tgt_1]

                        ind_orig_keep = cur_target['track_ids_orig'] != track_id_2
                        cur_target['track_ids_orig']        = cur_target['track_ids_orig'][ind_orig_keep]
                        cur_target['boxes_orig']            = cur_target['boxes_orig'][ind_orig_keep]
                        cur_target['labels_orig']           = cur_target['labels_orig'][ind_orig_keep]
                        cur_target['flexible_divisions_orig']= cur_target['flexible_divisions_orig'][ind_orig_keep]
                        cur_target['is_touching_edge_orig'] = cur_target['is_touching_edge_orig'][ind_orig_keep]

                        if use_masks:
                            mother_ind = torch.where(prev_target['track_ids'] == mother_id)[0][0]
                            prev_mask  = prev_target['masks'][mother_ind][:1]
                            mask_1     = cur_target['masks'][ind_tgt_1].detach()[:1]
                            mask_2     = cur_target['masks'][ind_tgt_2].detach()[:1]
                            sep_mask   = torch.cat((mask_1, mask_2), axis=0)
                            combined_mask = combine_div_masks(sep_mask, prev_mask)

                            cur_target['masks'][ind_tgt_1]        = combined_mask
                            cur_target['masks_orig'][ind_tgt_orig_1] = combined_mask
                            cur_target['masks_orig']              = cur_target['masks_orig'][ind_orig_keep]

                        ind_out[ind_1] = ind_out_combined

                        track_id_mot_ind = man_track[:, 0] == mother_id
                        track_id_1_ind   = man_track[:, 0] == track_id_1
                        track_id_2_ind   = man_track[:, 0] == track_id_2

                        man_track[track_id_mot_ind, 2] += 1
                        man_track[track_id_1_ind,   1] += 1
                        man_track[track_id_2_ind,   1] += 1

                        if (man_track[track_id_1_ind, 2] < man_track[track_id_1_ind, 1]
                            or man_track[track_id_2_ind, 2] < man_track[track_id_2_ind, 1]):

                            man_track[track_id_mot_ind, 2] = torch.max(
                                man_track[track_id_1_ind, 2], man_track[track_id_2_ind, 2]
                            )
                            man_track[track_id_1_ind, 1:] = -1
                            man_track[track_id_2_ind, 1:] = -1

                            if track_id_1 in fut_target['track_ids_orig']:
                                fut_target['track_ids_orig'][fut_target['track_ids_orig'] == track_id_1] = mother_id
                            elif track_id_2 in fut_target['track_ids_orig']:
                                fut_target['track_ids_orig'][fut_target['track_ids_orig'] == track_id_2] = mother_id

                            if track_id_1 in man_track[:, -1] and track_id_2 in man_track[:, -1]:
                                div_track_id_1, div_track_id_2 = man_track[(man_track[:, -1] == track_id_1), 0]
                                div_track_id_1_ind = man_track[:, 0] == div_track_id_1
                                div_track_id_2_ind = man_track[:, 0] == div_track_id_2
                                man_track[div_track_id_1_ind, -1] = 0
                                man_track[div_track_id_2_ind, -1] = 0
                                div_track_id_1, div_track_id_2 = man_track[(man_track[:, -1] == track_id_2), 0]
                                div_track_id_1_ind = man_track[:, 0] == div_track_id_1
                                div_track_id_2_ind = man_track[:, 0] == div_track_id_2
                                man_track[div_track_id_1_ind, -1] = 0
                                man_track[div_track_id_2_ind, -1] = 0
                            elif track_id_1 in man_track[:, -1]:
                                div_track_id_1, div_track_id_2 = man_track[(man_track[:, -1] == track_id_1), 0]
                                div_track_id_1_ind = man_track[:, 0] == div_track_id_1
                                div_track_id_2_ind = man_track[:, 0] == div_track_id_2
                                man_track[div_track_id_1_ind, -1] = mother_id
                                man_track[div_track_id_2_ind, -1] = mother_id
                            elif track_id_2 in man_track[:, -1]:
                                div_track_id_1, div_track_id_2 = man_track[(man_track[:, -1] == track_id_2), 0]
                                div_track_id_1_ind = man_track[:, 0] == div_track_id_1
                                div_track_id_2_ind = man_track[:, 0] == div_track_id_2
                                man_track[div_track_id_1_ind, -1] = mother_id
                                man_track[div_track_id_2_ind, -1] = mother_id

                # === Cas "sur le point de diviser" (logique d'origine) ==================
                elif (man_track[track_id_ind, 2] == framenb
                      and (man_track[:, -1] == track_id).sum() == 2
                      and training_method != 'dn_object'):

                    # (bloc d'origine, inchangé)
                    box = cur_target['boxes'][ind_tgt_i].clone()

                    fut_track_id_1, fut_track_id_2 = man_track[(man_track[:, -1] == track_id), 0]

                    fut_ind_tgt_1 = torch.where(fut_target['track_ids_orig'] == fut_track_id_1)[0][0]
                    fut_ind_tgt_2 = torch.where(fut_target['track_ids_orig'] == fut_track_id_2)[0][0]

                    fut_box_1 = fut_target['boxes_orig'][fut_ind_tgt_1, :4]
                    fut_box_2 = fut_target['boxes_orig'][fut_ind_tgt_2, :4]
                    fut_box = torch.cat((fut_box_1, fut_box_2))

                    div_box = divide_box(box, fut_box)

                    potential_object_query_indices = [ind_out_id for ind_out_id in torch.arange(N-num_queries, N)
                                                      if outputs['pred_logits'][t, ind_out_id, 0].sigmoid().detach() > 0.5
                                                      or ind_out_id == ind_out_i]

                    if len(potential_object_query_indices) > 1:
                        potential_pred_boxes = outputs['pred_boxes'][t, potential_object_query_indices].detach()

                        iou_div_all = generalized_box_iou(
                            box_cxcywh_to_xyxy(potential_pred_boxes[:, :4]),
                            box_cxcywh_to_xyxy(torch.cat((div_box[None, :4], div_box[None, 4:]), axis=0)),
                            return_iou_only=True
                        )

                        match_ind = torch.argmax(iou_div_all, axis=0).to('cpu')

                        if (potential_object_query_indices[match_ind[0]] != ind_out_i
                            and potential_object_query_indices[match_ind[1]] != ind_out_i):
                            continue

                        if len(torch.unique(match_ind)) == 2:
                            selected_pred_boxes = potential_pred_boxes[match_ind, :4]
                            iou_div = calc_iou(div_box, torch.cat((selected_pred_boxes[0], selected_pred_boxes[1])))

                            pred_box = outputs['pred_boxes'][t, ind_out_i, :4].detach()
                            iou = calc_iou(box, torch.cat((pred_box, torch.zeros_like(pred_box))))

                            assert 0 <= iou_div <= 1 and 0 <= iou <= 1

                            if iou_div - iou > 0 and iou_div > 0.5:

                                if (calc_iou(div_box[:4], selected_pred_boxes[0]) + calc_iou(div_box[4:], selected_pred_boxes[1])
                                    < calc_iou(div_box[4:], selected_pred_boxes[0]) + calc_iou(div_box[:4], selected_pred_boxes[1])):
                                    fut_track_id_1, fut_track_id_2 = fut_track_id_2, fut_track_id_1

                                cur_target['boxes'][ind_tgt_i] = torch.cat((div_box[:4], torch.zeros_like(div_box[:4])))
                                cur_target['boxes'] = torch.cat((cur_target['boxes'],
                                                                 torch.cat((div_box[4:], torch.zeros_like(div_box[:4])))[None]))

                                assert cur_target['labels'][ind_tgt_i, 1] == 1
                                cur_target['labels'] = torch.cat((cur_target['labels'], torch.tensor([0, 1])[None, ].to(device)))
                                cur_target['track_ids'][ind_tgt_i] = fut_track_id_1
                                cur_target['track_ids'] = torch.cat((cur_target['track_ids'], torch.tensor([fut_track_id_2]).to(device)))
                                cur_target['flexible_divisions'][ind_tgt_i] = True
                                cur_target['flexible_divisions'] = torch.cat((cur_target['flexible_divisions'], torch.tensor([True]).to(device)))
                                cur_target['is_touching_edge'] = torch.cat((cur_target['is_touching_edge'], cur_target['is_touching_edge'][ind_tgt_i][None]))

                                ind_keep = torch.cat((ind_keep, torch.tensor([True])))

                                ind_tgt_orig_i = torch.where(cur_target['boxes_orig'].eq(box).all(-1))[0][0]

                                cur_target['boxes_orig'][ind_tgt_orig_i] = torch.cat((div_box[:4], torch.zeros_like(div_box[:4])))
                                cur_target['boxes_orig'] = torch.cat((cur_target['boxes_orig'],
                                                                      torch.cat((div_box[4:], torch.zeros_like(div_box[:4])))[None]))

                                cur_target['labels_orig'] = torch.cat((cur_target['labels_orig'], torch.tensor([0, 1])[None, ].to(device)))
                                cur_target['track_ids_orig'][ind_tgt_orig_i] = fut_track_id_1
                                cur_target['track_ids_orig'] = torch.cat((cur_target['track_ids_orig'], torch.tensor([fut_track_id_2]).to(device)))
                                cur_target['flexible_divisions_orig'][ind_tgt_orig_i] = True
                                cur_target['flexible_divisions_orig'] = torch.cat((cur_target['flexible_divisions_orig'], torch.tensor([True]).to(device)))
                                cur_target['is_touching_edge_orig'] = torch.cat((cur_target['is_touching_edge_orig'], cur_target['is_touching_edge'][ind_tgt_orig_i][None]))

                                if use_masks:
                                    mask = cur_target['masks'][ind_tgt_i]
                                    fut_ind_tgt_1 = torch.where(fut_target['track_ids_orig'] == fut_track_id_1)[0][0]
                                    fut_ind_tgt_2 = torch.where(fut_target['track_ids_orig'] == fut_track_id_2)[0][0]
                                    fut_mask_1 = fut_target['masks_orig'][fut_ind_tgt_1][:1]
                                    fut_mask_2 = fut_target['masks_orig'][fut_ind_tgt_2][:1]
                                    fut_mask = torch.cat((fut_mask_1, fut_mask_2))
                                    div_mask = divide_mask(mask, fut_mask)

                                    cur_target['masks'][ind_tgt_i] = torch.cat((div_mask[:1], torch.zeros_like(div_mask[:1])))
                                    cur_target['masks'] = torch.cat((cur_target['masks'],
                                                                     torch.cat((div_mask[1:], torch.zeros_like(div_mask[:1])))[None]))

                                    cur_target['masks_orig'][ind_tgt_orig_i] = torch.cat((div_mask[:1], torch.zeros_like(div_mask[:1])))
                                    cur_target['masks_orig'] = torch.cat((cur_target['masks_orig'],
                                                                          torch.cat((div_mask[1:], torch.zeros_like(div_mask[:1])))[None]))

                                ind_out_copy = torch.cat((ind_out, torch.tensor([-10])))

                                if (potential_object_query_indices[match_ind[1]] != ind_out_i
                                    and potential_object_query_indices[match_ind[1]] in ind_out):
                                    ind_out[ind_out == potential_object_query_indices[match_ind[1]]] = -1
                                elif (potential_object_query_indices[match_ind[0]] != ind_out_i
                                      and potential_object_query_indices[match_ind[0]] in ind_out):
                                    ind_out[ind_out == potential_object_query_indices[match_ind[0]]] = -1

                                ind_out = torch.cat((ind_out, torch.tensor([potential_object_query_indices[match_ind[1]]])))
                                ind_tgt = torch.cat((ind_tgt, torch.tensor([cur_target['boxes'].shape[0]-1])))

                                ind_out[ind_out_copy == ind_out_i] = torch.tensor([potential_object_query_indices[match_ind[0]]])

                                if -1 in ind_out:
                                    unmatched_box = cur_target['boxes'][ind_out == -1]
                                    potential_object_query_indices = [ind_out_id for ind_out_id in torch.arange(N-num_queries, N)
                                                                      if ind_out_id not in ind_out
                                                                      and outputs['pred_logits'][t, ind_out_id, 0].sigmoid().detach() > 0.5]

                                    if len(potential_object_query_indices) == 0:
                                        potential_object_query_indices = [ind_out_id for ind_out_id in torch.arange(N-num_queries, N)
                                                                          if ind_out_id not in ind_out]

                                    if len(potential_object_query_indices) == 0:
                                        continue

                                    potential_pred_boxes = outputs['pred_boxes'][t, potential_object_query_indices].detach()

                                    iou_div_all = generalized_box_iou(
                                        box_cxcywh_to_xyxy(potential_pred_boxes[:, :4]),
                                        box_cxcywh_to_xyxy(unmatched_box[:, :4]),
                                        return_iou_only=True
                                    )

                                    if iou_div_all.sum() == 0:
                                        match_ind = torch.randint(low=0, high=len(potential_object_query_indices), size=(1,), dtype=torch.int)
                                    else:
                                        match_ind = torch.argmax(iou_div_all, axis=0).to('cpu')

                                    potential_object_query_ind = potential_object_query_indices[match_ind]
                                    assert potential_object_query_ind not in ind_out
                                    ind_out[ind_out == -1] = potential_object_query_ind

                                assert -1 not in ind_out
                                assert len(ind_out) == len(ind_tgt)
                                assert len(cur_target['boxes']) == len(cur_target['labels'])

                                fut_track_id_1_ind = man_track[:, 0] == fut_track_id_1
                                fut_track_id_2_ind = man_track[:, 0] == fut_track_id_2

                                man_track[track_id_ind, 2] -= 1
                                man_track[fut_track_id_1_ind, 1] -= 1
                                man_track[fut_track_id_2_ind, 1] -= 1

                                if man_track[track_id_ind, 1] > man_track[track_id_ind, 2]:
                                    man_track[track_id_ind, 1:] = -1
                                    man_track[fut_track_id_1_ind, -1] = 0
                                    man_track[fut_track_id_2_ind, -1] = 0

        if training_method == 'dn_object':
            if cur_target['num_FPs'] > 0:
                cur_target['track_queries_fal_pos_mask'][:-cur_target['num_FPs']][cur_target['track_ids'] == -1] = True
            else:
                cur_target['track_queries_fal_pos_mask'][cur_target['track_ids'] == -1] = True

        # réduire les GT aux indices à garder
        order = ind_tgt[ind_keep].sort()[0]
        cur_target['boxes']               = cur_target['boxes'][order]
        cur_target['labels']              = cur_target['labels'][order]
        cur_target['track_ids']           = cur_target['track_ids'][order]
        cur_target['flexible_divisions']  = cur_target['flexible_divisions'][order]
        cur_target['is_touching_edge']    = cur_target['is_touching_edge'][order]
        if use_masks:
            cur_target['masks'] = cur_target['masks'][order]

        if 'track_query_match_ids' in cur_target and training_method != 'dn_object':
            device = outputs['pred_logits'].device

            # 1) Récupérer prev_track_ids de façon défensive
            prev_ids_raw = prev_target.get('track_ids', torch.tensor([], device=device))
            if not isinstance(prev_ids_raw, torch.Tensor):
                prev_ids_raw = torch.as_tensor(prev_ids_raw, device=device)

            use_subset = (
                'prev_ind' in cur_target
                and isinstance(cur_target['prev_ind'], (list, tuple))
                and len(cur_target['prev_ind']) > 1
                and cur_target['prev_ind'][1] is not None
                and torch.numel(cur_target['prev_ind'][1]) > 0
            )
            if use_subset:
                idx = cur_target['prev_ind'][1].to(prev_ids_raw.device)
                prev_track_ids = prev_ids_raw[idx]
            else:
                prev_track_ids = prev_ids_raw

            # 2) Matching prev->cur (gère les cas vides)
            if prev_track_ids.numel() == 0 or cur_target['track_ids'].numel() == 0:
                cur_target['target_ind_matching']   = torch.zeros((prev_track_ids.shape[0],), dtype=torch.bool, device=device)
                cur_target['track_query_match_ids'] = torch.tensor([], dtype=torch.long, device=device)
            else:
                mat = prev_track_ids.unsqueeze(1).eq(cur_target['track_ids'])
                # ne garder qu'un seul True par ligne (première occurrence)
                if mat.numel() > 0:
                    first_true = mat.int().cumsum(dim=1) == 1
                    mat = mat & first_true
                cur_target['target_ind_matching'] = mat.any(dim=1)
                nz = mat.nonzero(as_tuple=False)
                cur_target['track_query_match_ids'] = (nz[:, 1] if nz.numel()
                                                    else torch.tensor([], dtype=torch.long, device=device))

            # 3) Reconstruire les masques
            num_queries_local = ((~cur_target['track_queries_mask']).sum()
                                if 'track_queries_mask' in cur_target else num_queries)
            tq_mask = torch.ones_like(cur_target['target_ind_matching']).bool()

            cur_target['track_queries_mask'] = torch.cat([
                tq_mask,
                torch.tensor([True]  * int(cur_target['num_FPs']), device=device),
                torch.tensor([False] * int(num_queries_local),     device=device),
            ]).bool()

            cur_target['track_queries_TP_mask'] = torch.cat([
                cur_target['target_ind_matching'],
                torch.tensor([False] * int(cur_target['num_FPs']), device=device),
                torch.tensor([False] * int(num_queries_local),     device=device),
            ]).bool()

            cur_target['track_queries_fal_pos_mask'] = torch.cat([
                ~cur_target['target_ind_matching'],
                torch.tensor([True]  * int(cur_target['num_FPs']), device=device),
                torch.tensor([False] * int(num_queries_local),     device=device),
            ]).bool()


        ind_out = ind_out[ind_keep]
        ind_tgt = ind_tgt[ind_keep]

        # reindexation compacte de ind_tgt
        while not torch.arange(len(ind_tgt))[:, None].eq(ind_tgt[None]).any(0).all():
            for i in range(len(ind_tgt)):
                if i not in ind_tgt:
                    ind_tgt[ind_tgt > i] = ind_tgt[ind_tgt > i] - 1

        indices[t] = (ind_out, ind_tgt)
        targets[t][training_method]['man_track'] = man_track

    return targets, indices
