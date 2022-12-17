# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import math

import torch
from torch import nn
from torch.nn.init import constant_, normal_, xavier_uniform_
import torch.nn.functional as F

from ..util.misc import inverse_sigmoid
from .ops.modules import MSDeformAttn
from .transformer import _get_clones, _get_activation_fn


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024,
                 dropout=0.1, activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, num_queries=30,
                 use_dab=False, high_dim_query_update=False, no_sine_embed=False,
                 multi_frame_attention_separate_encoder=False, 
                 refine_track_queries=False, refine_div_track_queries=False):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.num_queries = num_queries
        self.num_feature_levels = num_feature_levels
        self.multi_frame_attention_separate_encoder = multi_frame_attention_separate_encoder
        self.use_dab = use_dab

        self.refine_track_queries = refine_track_queries
        self.refine_div_track_queries = refine_div_track_queries

        if self.refine_track_queries:
            self.track_embedding = nn.Embedding(1,self.hidden_dim)

        if self.refine_div_track_queries:
            self.div_track_embedding = nn.Embedding(2,self.hidden_dim)

        enc_num_feature_levels = num_feature_levels
        if multi_frame_attention_separate_encoder:
            enc_num_feature_levels = enc_num_feature_levels // 2
        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          enc_num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec,
                                                    use_dab=use_dab, d_model=d_model, high_dim_query_update=high_dim_query_update, no_sine_embed=no_sine_embed)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            if self.use_dab:
                self.pos_trans = nn.Linear(d_model * 2, d_model)
                self.pos_trans_norm = nn.LayerNorm(d_model)
            else:
                self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
                self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            if not self.use_dab:
                self.reference_points = nn.Linear(d_model, 2)

        self.high_dim_query_update = high_dim_query_update
        if high_dim_query_update:
            assert not self.use_dab, "use_dab must be True"

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage and not self.use_dab:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        # num_pos_feats = 128
        num_pos_feats = 144
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None, targets=None, query_attn_mask=None):
        assert self.two_stage or query_embed is not None
        if not self.two_stage:
            assert torch.sum(torch.isnan(query_embed)) == 0, 'Nan in reference points'

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            # lvl_pos_embed = pos_embed + self.level_embed[lvl % self.num_feature_levels].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes[:spatial_shapes.shape[0]//2].new_zeros((1, )), spatial_shapes[:spatial_shapes.shape[0]//2].prod(1).cumsum(0)[:-1]))
        # level_start_index = torch.cat((spatial_shapes[:spatial_shapes.shape[0]].new_zeros((1, )), spatial_shapes[:spatial_shapes.shape[0]].prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        if self.multi_frame_attention_separate_encoder:
            prev_memory = self.encoder(
                src_flatten[:, :src_flatten.shape[1] // 2],
                spatial_shapes[:self.num_feature_levels // 2],
                valid_ratios[:, :self.num_feature_levels // 2],
                level_start_index,
                lvl_pos_embed_flatten[:, :src_flatten.shape[1] // 2],
                mask_flatten[:, :src_flatten.shape[1] // 2],
                )
            memory = self.encoder(
                src_flatten[:, src_flatten.shape[1] // 2:],
                spatial_shapes[self.num_feature_levels // 2:],
                valid_ratios[:, self.num_feature_levels // 2:],
                level_start_index,
                lvl_pos_embed_flatten[:, src_flatten.shape[1] // 2:],
                mask_flatten[:, src_flatten.shape[1] // 2:],
                )
            memory = torch.cat([memory, prev_memory], 1)
        else:
            memory = self.encoder(src_flatten, spatial_shapes, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        # prepare input for decoder
        bs, _, c = memory.shape

        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

            # hack implementation for two-stage Deformable DETR
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory)[...,:output_proposals.shape[-1]] + output_proposals

            assert self.num_queries <= enc_outputs_class.shape[1]
            topk = self.num_queries
            # topk = min(self.two_stage_num_proposals,enc_outputs_class.shape[1])
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points

            if self.use_dab:
                tgt = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
                query_embed = None
            else:
                pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
                query_embed, tgt = torch.split(pos_trans_out, c, dim=2)

        elif self.use_dab:
            reference_points = query_embed[..., self.d_model:].sigmoid() 
            tgt = query_embed[..., :self.d_model]
            query_embed = None
        else:
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)

            reference_points = self.reference_points(query_embed).sigmoid()

            assert torch.sum(torch.isnan(reference_points)) == 0, 'Nan in reference points'


        if targets is not None and 'track_query_hs_embeds' in targets[0]:

            prev_hs_embed = torch.stack([t['track_query_hs_embeds'] for t in targets])
            prev_boxes = torch.stack([t['track_query_boxes'] for t in targets])

            assert torch.sum(torch.isnan(prev_hs_embed)) == 0, 'Nan in track query_hs embeds'
            assert torch.sum(torch.isnan(prev_boxes)) == 0, 'Nan in track boxes'

            #### Group DETR - get rid of attn mask and do all masking on deformalbe_detr.py
            if not self.use_dab:
                prev_query_embed = torch.zeros_like(prev_hs_embed)
                query_embed = torch.cat([prev_query_embed, query_embed], dim=1)

            prev_tgt = prev_hs_embed
            tgt = torch.cat([prev_tgt, tgt], dim=1)

            reference_points = torch.cat([prev_boxes[..., :reference_points.shape[-1]], reference_points], dim=1)

            if 'dn_track' in targets[0]:
                
                targets_dn_track = [target['dn_track'] for target in targets]

                num_dn_track = targets_dn_track[0]['boxes'].shape[0]

                prev_hs_embed_dn_track = [t['track_query_hs_embeds'] for t in targets_dn_track]
                prev_boxes_dn_track = torch.stack([t['track_query_boxes'] for t in targets_dn_track])

                # if self.refine_div_track_queries:
                #     # We want to inject noise into the divided cells
                #     # I randomly flip because I don't want to memorize

                #     for tidx,target in enumerate(targets):
                #         rand_nb = torch.randint(2,(1,))[0]
                #         prev_hs_embed_dn_track[tidx][torch.where(target['tgt_div_ind'] == 1)] += self.div_track_embedding.weight[rand_nb]
                #         prev_hs_embed_dn_track[tidx][torch.where(target['tgt_div_ind'] == 2)] += self.div_track_embedding.weight[1 - rand_nb]

                prev_hs_embed_dn_track = torch.stack(prev_hs_embed_dn_track)

                assert torch.sum(torch.isnan(prev_hs_embed_dn_track)) == 0, 'Nan in track query_hs embeds'
                assert torch.sum(torch.isnan(prev_boxes_dn_track)) == 0, 'Nan in track boxes'

                if self.refine_track_queries:
                    prev_hs_embed_dn_track += self.track_embedding.weight

                #### Group DETR - get rid of attn mask and do all masking on deformalbe_detr.py
                if not self.use_dab:
                    prev_query_embed_dn_track = torch.zeros_like(prev_hs_embed_dn_track)
                    query_embed = torch.cat([query_embed,prev_query_embed_dn_track], dim=1)

                prev_tgt_dn_track = prev_hs_embed_dn_track
                tgt = torch.cat([tgt,prev_tgt_dn_track], dim=1)

                reference_points = torch.cat([reference_points,prev_boxes_dn_track[..., :reference_points.shape[-1]]], dim=1)

                new_query_attn_mask = torch.zeros((tgt.shape[1],tgt.shape[1]),device=tgt.device).bool()

                if query_attn_mask is not None:
                    new_query_attn_mask[:query_attn_mask.shape[0],:query_attn_mask.shape[1]] = query_attn_mask

                new_query_attn_mask[:-num_dn_track,-num_dn_track:] = True
                new_query_attn_mask[-num_dn_track:,:-num_dn_track] = True

                query_attn_mask = new_query_attn_mask

        init_reference_out = reference_points


        # decoder
        hs, inter_references = self.decoder(
            tgt, reference_points, memory, spatial_shapes,
            valid_ratios, level_start_index, query_embed, mask_flatten, query_attn_mask)

        assert torch.sum(torch.isnan(hs)) == 0, 'Nan in hs_embedding decoder outputs'

        inter_references_out = inter_references

        if self.two_stage:
            return (hs, memory, init_reference_out, inter_references_out,
                    enc_outputs_class, enc_outputs_coord_unact)
        return hs, memory, init_reference_out, inter_references_out, None, None


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes,level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, valid_ratios, level_start_index, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, src_level_start_index, src_padding_mask=None, query_attn_mask=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)

        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1), attn_mask=query_attn_mask)[0].transpose(0, 1)

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, src_level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False, use_dab=False, d_model=256, high_dim_query_update=False, no_sine_embed=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

        #### DAB-DETR
        self.use_dab = use_dab
        self.d_model = d_model
        self.no_sine_embed = no_sine_embed
        if use_dab:
            self.query_scale = MLP(d_model, d_model, d_model, 2)
            if self.no_sine_embed:
                self.ref_point_head = MLP(4, d_model, d_model, 3)
            else:
                self.ref_point_head = MLP(2 * d_model, d_model, d_model, 2)
        self.high_dim_query_update = high_dim_query_update
        if high_dim_query_update:
            self.high_dim_query_proj = MLP(d_model, d_model, d_model, 2)
        #### DAB-DETR

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_valid_ratios, src_level_start_index,
                query_pos=None, src_padding_mask=None, query_attn_mask=None):
        output = tgt

        #### DAB-DETR
        if self.use_dab:
            assert query_pos is None
        #### DAB-DETR

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]

            #### DAB-DETR
            if self.use_dab:
                if self.no_sine_embed:
                    raw_query_pos = self.ref_point_head(reference_points_input)
                else:
                    query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :],self.d_model) # bs, nq, d_model * 2
                    raw_query_pos = self.ref_point_head(query_sine_embed) # bs, nq, 256
                pos_scale = self.query_scale(output) if lid != 0 else 1
                query_pos = pos_scale * raw_query_pos
            if self.high_dim_query_update and lid != 0:
                query_pos = query_pos + self.high_dim_query_proj(output) 
            #### DAB-DETR


            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask, query_attn_mask)
            assert torch.sum(torch.isnan(output)) == 0, 'Output causing nan'

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)[:,:,:4]  # I am using the first box as the basis for the reference points
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    assert torch.sum(torch.isnan(new_reference_points)) == 0, 'Reference points causing nan'
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


def build_deforamble_transformer(args):

    num_feature_levels = args.num_feature_levels
    if args.multi_frame_attention:
        num_feature_levels *= 2

    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage,
        num_queries=args.num_queries,
        use_dab=args.use_dab,
        multi_frame_attention_separate_encoder=args.multi_frame_attention and args.multi_frame_attention_separate_encoder,
        )

class MLP(nn.Module):
    """ 
        Adapted from DAB-DETR
        Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def gen_sineembed_for_position(pos_tensor,d_model):
    '''Adapted from DAB-DETR'''
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 288)
    scale = 2 * math.pi
    dim_t = torch.arange(torch.div(d_model,2,rounding_mode='floor'), dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * torch.div(dim_t,2,rounding_mode='floor') / torch.div(d_model,2,rounding_mode='floor'))
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos
