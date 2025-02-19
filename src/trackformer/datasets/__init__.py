# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Submodule interface.
"""
from argparse import Namespace
from pycocotools.coco import COCO
from torch.utils.data import Dataset, Subset
from torchvision.datasets import CocoDetection

from .mot import build_cells, SubsetSampler


def get_coco_api_from_dataset(dataset: Subset) -> COCO:
    """Return COCO class from PyTorch dataset for evaluation with COCO eval."""
    for _ in range(10):
        # if isinstance(dataset, CocoDetection):
        #     break
        if isinstance(dataset, Subset):
            dataset = dataset.dataset

    if not isinstance(dataset, CocoDetection):
        raise NotImplementedError

    return dataset.coco


def build_dataset(split: str, args: Namespace) -> Dataset:
    """Helper function to build dataset for different splits ('train' or 'val')."""

    dataset = build_cells(split,args)

    return dataset
