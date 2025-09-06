# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area
import os 

def _order_key_xywh(box4: torch.Tensor):
    # box format is cx, cy, w, h (Cell-TRACTR)
    # returns (cx, cy) as ordering key
    return box4[..., 0], box4[..., 1]

def standardize_div_order(box8: torch.Tensor) -> torch.Tensor:
    """
    Ensure a consistent order of the two halves in an 8D division box:
      - primarily by cx (left->right)
      - tie-breaker by cy (top->bottom)
    Works on shape (8,) or (..., 8).
    """
    if box8.ndim == 1:
        a, b = box8[:4].clone(), box8[4:].clone()
        cx_a, cy_a = _order_key_xywh(a)
        cx_b, cy_b = _order_key_xywh(b)
        swap = (cx_a > cx_b) or (abs(float(cx_a - cx_b)) < 1e-6 and cy_a > cy_b)
        if swap:
            a, b = b, a
        return torch.cat((a, b), dim=0).to(box8.device)
    else:
        a, b = box8[..., :4].clone(), box8[..., 4:].clone()
        cx_a, cy_a = _order_key_xywh(a)
        cx_b, cy_b = _order_key_xywh(b)
        swap = (cx_a > cx_b) | ((cx_a - cx_b).abs() < 1e-6) & (cy_a > cy_b)
        a2 = torch.where(swap[..., None], b, a)
        b2 = torch.where(swap[..., None], a, b)
        return torch.cat((a2, b2), dim=-1).to(box8.device)

def box_cxcywh_to_xyxy(x):

    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def box_cxcy_to_xyxy(boxes,height,width):

    boxes[:,1::2] = boxes[:,1::2] * height
    boxes[:,::2] = boxes[:,::2] * width

    boxes[:,0] = boxes[:,0] - boxes[:,2] // 2
    boxes[:,1] = boxes[:,1] - boxes[:,3] // 2

    if boxes.shape[1] > 4:
        boxes[:,4] = boxes[:,4] - boxes[:,6] // 2
        boxes[:,5] = boxes[:,5] - boxes[:,7] // 2
    
    return boxes

# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2, return_iou_only = False):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """

    if boxes1.dim() == 1:
        boxes1 = boxes1[None]
        
    if boxes2.dim() == 1:
        boxes2 = boxes2[None]

    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    if return_iou_only:
        return iou

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks,cxcywh=False):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float, device=masks.device)
    x = torch.arange(0, w, dtype=torch.float, device=masks.device)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]


    if cxcywh:
        boxes = torch.stack([(x_min+x_max)/2/w, (y_min+y_max)/2/h, (x_max-x_min)/w, (y_max-y_min)/h], 1)
    else:
        boxes = torch.stack([x_min, y_min, x_max, y_max], 1)

    boxes[masks.sum((1,2)) == 0] = torch.tensor([0,0,0,0],dtype=boxes.dtype,device=masks.device)

    if cxcywh:
        assert (boxes[::2] > 1).sum() == 0 and (boxes[1::2] > 1).sum() == 0
    else:
        assert (boxes[::2] > w).sum() == 0 and (boxes[1::2] > h).sum() == 0
    return boxes

def mask_iou(inputs, targets):
    """
    Compute the IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    intersection = (inputs * targets).sum(1)
    union = (inputs.sum(-1) + targets.sum(-1) - intersection)
    iou = intersection / union
    return iou


def combine_div_boxes(box):

    new_box = torch.zeros_like(box)

    min_y = min(box[1] - box[3] / 2, box[5] - box[7] / 2)
    max_y = max(box[1] + box[3] / 2, box[5] + box[7] / 2)
    avg_y = (min_y + max_y) / 2
    new_box[1] = avg_y
    new_box[3] = max_y - min_y

    min_x = min(box[0] - box[2] / 2, box[4] - box[6] / 2)
    max_x = max(box[0] + box[2] / 2, box[4] + box[6] / 2)
    avg_x = (min_x + max_x) / 2
    new_box[0] = avg_x
    new_box[2] = max_x - min_x

    return new_box

def combine_div_masks(mask, prev_mask):

    combined_mask = torch.zeros_like(mask)
    combined_mask[0] += mask[0].clone()
    combined_mask[0] += mask[1].clone()

    div_cells_loc_y, div_cells_loc_x = torch.where(mask[0]+mask[1])

    prev_cell_loc_y, prev_cell_loc_x = torch.where(prev_mask[0])

    diff_loc_y, diff_loc_x = div_cells_loc_y.float().mean() - prev_cell_loc_y.float().mean(), div_cells_loc_x.float().mean() - prev_cell_loc_x.float().mean()

    prev_cell_loc = (torch.clamp(prev_cell_loc_y + int(diff_loc_y),0,mask.shape[1] - 1 ), torch.clamp(prev_cell_loc_x + int(diff_loc_x),0,mask.shape[2] - 1))

    combined_mask[0][prev_cell_loc] = 1
    
    return combined_mask

import torch

def divide_box(box, fut_box):
    """
    Ajuste la bbox mère et la bbox fille prédites au futur (fut_box)
    en les recentrant par rapport à la bbox mère courante (box),
    puis les borne à l'intérieur de la bbox mère courante.
    Entrées et sortie attendues: Tensors 1D de taille 8 [cx,cy,w,h, cx_f,cy_f,w_f,h_f].
    """
    box     = box.view(-1)
    fut_box = fut_box.view(-1)
    assert box.numel() == 8 and fut_box.numel() == 8, "divide_box attend des vecteurs (8,)"

    device = box.device
    dtype  = box.dtype

    new_box = torch.zeros_like(box)

    # On part des tailles de fut_box
    new_box[2:4] = fut_box[2:4]   # w,h mère
    new_box[6:8] = fut_box[6:8]   # w,h fille

    # Centre moyen (x,y) couvrant mère & fille de fut_box
    min_y = torch.minimum(fut_box[1] - fut_box[3] / 2, fut_box[5] - fut_box[7] / 2)
    max_y = torch.maximum(fut_box[1] + fut_box[3] / 2, fut_box[5] + fut_box[7] / 2)
    avg_y = (min_y + max_y) / 2
    dif_y = box[1] - avg_y

    min_x = torch.minimum(fut_box[0] - fut_box[2] / 2, fut_box[4] - fut_box[6] / 2)
    max_x = torch.maximum(fut_box[0] + fut_box[2] / 2, fut_box[4] + fut_box[6] / 2)
    avg_x = (min_x + max_x) / 2
    dif_x = box[0] - avg_x

    # Recentrage des centres (mère aux indices 0,1 ; fille aux 4,5)
    new_box[0::4] = fut_box[0::4] + dif_x
    new_box[1::4] = fut_box[1::4] + dif_y

    # Bornes imposées par la bbox mère courante (x0,y0,x1,y1)
    mother_x0 = box[0] - box[2] / 2
    mother_x1 = box[0] + box[2] / 2
    mother_y0 = box[1] - box[3] / 2
    mother_y1 = box[1] + box[3] / 2

    # Clamp des coins pour mère et fille
    new_y0 = torch.clamp(new_box[1::4] - new_box[3::4] / 2, min=mother_y0, max=mother_y1)
    new_y1 = torch.clamp(new_box[1::4] + new_box[3::4] / 2, min=mother_y0, max=mother_y1)
    new_x0 = torch.clamp(new_box[0::4] - new_box[2::4] / 2, min=mother_x0, max=mother_x1)
    new_x1 = torch.clamp(new_box[0::4] + new_box[2::4] / 2, min=mother_x0, max=mother_x1)

    # Recalcule (cx,cy,w,h) depuis coins clampés
    eps = torch.tensor(1e-6, device=device, dtype=dtype)
    new_box[1::4] = (new_y0 + new_y1) / 2
    new_box[3::4] = torch.clamp(new_y1 - new_y0, min=eps)
    new_box[0::4] = (new_x0 + new_x1) / 2
    new_box[2::4] = torch.clamp(new_x1 - new_x0, min=eps)

    # Standardiser l'ordre (si la fonction existe dans le module)
    try:
        new_box = standardize_div_order(new_box)
    except NameError:
        pass

    return new_box


def divide_mask(mask,fut_mask):

    div_mask = torch.zeros_like(mask)

    avg_loc_y, avg_loc_x = torch.where(mask[0])[0].float().mean(), torch.where(mask[0])[1].float().mean()

    fut_avg_loc_y, fut_avg_loc_x = torch.where(fut_mask[0] + fut_mask[1])[0].float().mean(), torch.where(fut_mask[0] + fut_mask[1])[1].float().mean()

    diff_loc_y, diff_loc_x = avg_loc_y - fut_avg_loc_y, avg_loc_x - fut_avg_loc_x

    fut_cell_1_loc = torch.where(fut_mask[0])
    fut_cell_2_loc = torch.where(fut_mask[1])

    fut_cell_1_loc = (torch.clamp(fut_cell_1_loc[0] + int(diff_loc_y),0,mask.shape[1] - 1 ), torch.clamp(fut_cell_1_loc[1] + int(diff_loc_x),0,mask.shape[2] - 1))
    fut_cell_2_loc = (torch.clamp(fut_cell_2_loc[0] + int(diff_loc_y),0,mask.shape[1] - 1), torch.clamp(fut_cell_2_loc[1] + int(diff_loc_x),0,mask.shape[2] - 1))

    div_mask[0][fut_cell_1_loc] = 1
    div_mask[1][fut_cell_2_loc] = 1

    return div_mask

def calc_iou(box_1,box_2, return_flip=False):

    assert box_1.ndim == 1 and box_2.ndim == 1

    if (box_1[-1] == 0 and box_2[-1] == 0) or (box_1.shape[0] == 4 and box_2.shape[0] == 4):
        iou = generalized_box_iou(
            box_cxcywh_to_xyxy(box_1[:4]),
            box_cxcywh_to_xyxy(box_2[:4]),
            return_iou_only=True
        )

    elif box_1[-1] > 0 and box_2[-1] > 0:
        iou_1 = generalized_box_iou(
            box_cxcywh_to_xyxy(box_1[:4]),
            box_cxcywh_to_xyxy(box_2[:4]),
            return_iou_only=True
        )

        iou_2 = generalized_box_iou(
            box_cxcywh_to_xyxy(box_1[4:]),
            box_cxcywh_to_xyxy(box_2[4:]),
            return_iou_only=True
        )

        iou = (iou_1 + iou_2) / 2

        iou_1_flip = generalized_box_iou(
            box_cxcywh_to_xyxy(box_1[:4]),
            box_cxcywh_to_xyxy(box_2[4:]),
            return_iou_only=True
        )

        iou_2_flip = generalized_box_iou(
            box_cxcywh_to_xyxy(box_1[4:]),
            box_cxcywh_to_xyxy(box_2[:4]),
            return_iou_only=True
        )

        iou_flip = (iou_1_flip + iou_2_flip) / 2

        if iou_flip > iou:
            flip = True
        else:
            flip = False

        iou = max(iou,iou_flip)

    else:
        iou = 0

    if return_flip:
        return iou, flip
    
    return iou


def add_noise_to_boxes(boxes,l_1,l_2,clamp=True):
    noise = torch.rand_like(boxes) * 2 - 1
    boxes[..., :2] += boxes[..., 2:] * noise[..., :2] * l_1
    boxes[..., 2:] *= 1 + l_2 * noise[..., 2:]
    if clamp:
        boxes = torch.clamp(boxes,0,1)
    return boxes


def combine_boxes_parallel(boxes_1,boxes_2):
    boxes = torch.stack((boxes_1,boxes_2),axis=1)
    # Check that the boxes have the correct shape (n, 2, 4)
    assert boxes.ndim == 3
    assert boxes.shape[1] == 2
    assert boxes.shape[2] == 4
    
    # Calculate new centers (average of the two centers)
    new_centers = boxes[:, :, :2].mean(dim=1)
    
    # Calculate new width and height (maximum of the two widths and heights)
    new_wh = boxes[:, :, 2:].max(dim=1)[0]
    
    # Concatenate the new centers and new width and height to form the combined boxes
    combined_boxes = torch.cat((new_centers, new_wh), dim=1)
    
    return combined_boxes