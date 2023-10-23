# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch

from .backbone import build_backbone
from .deformable_detr import DeformableDETR, DeformablePostProcess
from .deformable_transformer import build_deforamble_transformer
from .detr import PostProcess, SetCriterion
# from .detr_segmentation import (DeformableDETRSegm, DeformableDETRSegmTracking, DETRSegm, DETRSegmTracking)
# from .detr_tracking import DeformableDETRTracking, DETRTracking
from .detr_segmentation import (DeformableDETRSegm, DeformableDETRSegmTracking)
from .detr_tracking import DeformableDETRTracking
from .matcher import build_matcher
# from .transformer import build_transformer


def build_model(args):
    num_classes = 1
    device = torch.device(args.device)
    backbone = build_backbone(args)
    matcher = build_matcher(args)

    detr_kwargs = {
        'backbone': backbone,
        'num_classes': num_classes - 1 if args.focal_loss else num_classes,
        'num_queries': args.num_queries,
        'aux_loss': args.aux_loss,
        'overflow_boxes': args.overflow_boxes,
        'device': device,
        'use_dab': args.use_dab,
        'dn_object_l1': args.dn_object_l1,
        'dn_object_l2': args.dn_object_l2,
        'dn_label': args.dn_label,
        'refine_object_queries': args.refine_object_queries,
        'use_div_ref_pts': args.use_div_ref_pts,
        'share_bbox_layers': args.share_bbox_layers,
        'with_box_refine': args.with_box_refine,
        'num_queries': args.num_queries,
        'num_feature_levels': args.num_feature_levels,
        'two_stage': args.two_stage,
        'multi_frame_attention': args.multi_frame_attention,
        'multi_frame_encoding': args.multi_frame_encoding,
        'merge_frame_features': args.merge_frame_features,
        'decoder_use_mask_as_ref': args.decoder_use_mask_as_ref,
        'iterative_masks': args.iterative_masks,
        'use_img_for_mask': args.use_img_for_mask}
    
    tracking_kwargs = {
        'matcher': matcher,
        'backprop_prev_frame': args.track_backprop_prev_frame,
        'dn_track': args.dn_track,
        'dn_track_l1': args.dn_track_l1,
        'dn_track_l2': args.dn_track_l2,
        'dn_object': args.dn_object,
        'dn_enc':args.dn_enc,
        'refine_div_track_queries': args.refine_div_track_queries,
        'no_data_aug': args.no_data_aug,
        'flex_div': args.flex_div,
        'epoch_to_start_using_flexible_divisions': args.epoch_to_start_using_flexible_divisions,
        'use_prev_prev_frame': args.use_prev_prev_frame,
        'dn_track_add_object_queries': args.dn_track_add_object_queries,
        'object_detection_only': args.object_detection_only,
        'num_queries': args.num_queries,
        'num_epochs': args.epochs,
        'dn_track_group': args.dn_track_group,
        'tgt_noise': args.tgt_noise}

    mask_kwargs = {
        'freeze_detr': args.freeze_detr,
        'return_intermediate_masks': args.return_intermediate_masks,
        'mask_dim': args.mask_dim,}

    if args.deformable:
        args.feature_channels = backbone.num_channels
        # transformer = build_deforamble_transformer(args)

        transformer_kwargs = {}
        transformer_kwargs['d_model'] = args.hidden_dim
        transformer_kwargs['num_queries'] = args.num_queries
        transformer_kwargs['num_feature_levels'] = args.num_feature_levels
        transformer_kwargs['two_stage'] = args.two_stage
        transformer_kwargs['nhead'] = args.nheads
        transformer_kwargs['num_encoder_layers'] = args.enc_layers
        transformer_kwargs['num_decoder_layers'] = args.dec_layers
        transformer_kwargs['dim_feedforward'] = args.dim_feedforward

        transformer_kwargs['dropout']=args.dropout
        transformer_kwargs['activation']="relu"
        transformer_kwargs['return_intermediate_dec']=True

        num_feature_levels = args.num_feature_levels
        if args.multi_frame_attention:
            num_feature_levels *= 2

        transformer_kwargs['num_feature_levels']=num_feature_levels
        transformer_kwargs['dec_n_points'] = args.dec_n_points
        transformer_kwargs['enc_n_points'] = args.enc_n_points
        transformer_kwargs['two_stage'] = args.two_stage
        transformer_kwargs['num_queries'] = args.num_queries
        transformer_kwargs['batch_size'] = args.batch_size
        transformer_kwargs['use_dab'] = args.use_dab

        transformer_kwargs['multi_frame_attention_separate_encoder'] = args.multi_frame_attention and args.multi_frame_attention_separate_encoder
        transformer_kwargs['init_enc_queries_embeddings'] = args.init_enc_queries_embeddings
        transformer_kwargs['dn_enc_l1'] = args.dn_enc_l1
        transformer_kwargs['dn_enc_l2'] = args.dn_enc_l2
        transformer_kwargs['mask_dim'] = args.mask_dim
        transformer_kwargs['init_boxes_from_masks'] = args.init_boxes_from_masks
        transformer_kwargs['feature_channels'] = args.feature_channels
        transformer_kwargs['device'] = args.device
        transformer_kwargs['masks'] = args.masks
        transformer_kwargs['dn_track_add_object_queries'] = args.dn_track_add_object_queries
        transformer_kwargs['enc_masks'] = args.enc_masks
        transformer_kwargs['enc_FN'] = args.enc_FN
        transformer_kwargs['avg_attn_weight_maps'] = args.avg_attn_weight_maps
        transformer_kwargs['use_img_for_mask'] = args.use_img_for_mask

        if args.tracking:
            if args.masks:
                model = DeformableDETRSegmTracking(mask_kwargs, tracking_kwargs, detr_kwargs,transformer_kwargs)
            else:
                model = DeformableDETRTracking(tracking_kwargs, detr_kwargs, transformer_kwargs)
        else:
            if args.masks:
                model = DeformableDETRSegm(mask_kwargs, detr_kwargs, transformer_kwargs)
            else:
                model = DeformableDETR(detr_kwargs, transformer_kwargs)
    # else:
    #     transformer = build_transformer(args)

    #     detr_kwargs['transformer'] = transformer

    #     if args.tracking:
    #         if args.masks:
    #             model = DETRSegmTracking(mask_kwargs, tracking_kwargs, detr_kwargs)
    #         else:
    #             model = DETRTracking(tracking_kwargs, detr_kwargs)
    #     else:
    #         if args.masks:
    #             model = DETRSegm(mask_kwargs, detr_kwargs)
    #         else:
    #             model = DETR(**detr_kwargs)

    weight_dict = {'loss_ce': args.bbox_loss_coef,
                   'loss_bbox': args.bbox_loss_coef,
                   'loss_giou': args.giou_loss_coef,}

    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef

    training_methods = []
    if args.dn_track:
        training_methods.append('dn_track')
    if args.dn_track_group:
        training_methods.append('dn_track_group')
    if args.dn_object:
        training_methods.append('dn_object')
    if args.dn_enc:
        training_methods.append('dn_enc')
    if args.CoMOT:
        training_methods.append('CoMOT')


    weight_dict_TM = {}
    for weight_dict_key in list(weight_dict.keys()):
        for training_method in training_methods:
            if training_method == 'CoMOT':
                continue

            weight_dict_TM.update({f'{weight_dict_key}_{training_method}': weight_dict[weight_dict_key]})

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers-1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})

            for training_method in training_methods:
                aux_weight_dict.update({k + f'_{i}_{training_method}': v for k, v in weight_dict.items()})

            if args.CoMOT:
                aux_weight_dict.update({k + f'_{i}_CoMOT': v for k, v in weight_dict.items()})

        if args.two_stage:
            aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})

        weight_dict.update(aux_weight_dict)

    weight_dict.update(weight_dict_TM)

    weight_dict.update({'loss': args.loss_coef})

    losses = ['labels', 'boxes']
    if args.masks:
        losses.append('masks')


    criterion = SetCriterion(
        num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        losses=losses,
        focal_loss=args.focal_loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        tracking=args.tracking,
        args=args,)
    criterion.to(device)

    return model, criterion
