"""
MX-Font
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import torch

from .imagefolder_dataset import ImageTestDataset
from .ttf_dataset import TTFTrainDataset, TTFValDataset
from .ttf_utils import get_filtered_chars, read_font, render
from torch.utils.data import DataLoader


def get_trn_loader(cfg, primals, decomposition, transform, use_ddp=False, **kwargs):
    dset = TTFTrainDataset(
        primals=primals,
        decomposition=decomposition,
        transform=transform,
        **cfg
    )
    if use_ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(dset)
        kwargs["shuffle"] = False
    else:
        sampler = None
    loader = DataLoader(dset, sampler=sampler, collate_fn=dset.collate_fn, **kwargs)

    return dset, loader


def get_val_loader(cfg, transform, **kwargs):
    english_chars = list(range(65, 91)) + list(range(97, 123)) # A-Z, a-z
    russian_chars = list(range(0x0410, 0x0450))+[0x0401, 0x0451]  # А-я
    kazakh_chars = [0x04D8, 0x04D9, 0x0492, 0x0493, 0x049A, 0x049B, 0x04A2, 0x04A3, 0x04E8, 0x04E9, 0x04B0, 0x04B1, 0x04AE, 0x04AF, 0x04BA, 0x04BB, 0x0406, 0x0456]
    char_filter = [chr(i) for i in english_chars+russian_chars+kazakh_chars]
    dset = TTFValDataset(
        char_filter=char_filter,
        transform=transform,
        **cfg
    )
    loader = DataLoader(dset, collate_fn=dset.collate_fn, **kwargs)

    return dset, loader


def get_test_loader(cfg, transform, **kwargs):
    dset = ImageTestDataset(
        transform=transform,
        **cfg.dset.test,
    )
    loader = DataLoader(dset, batch_size=cfg.batch_size, collate_fn=dset.collate_fn, **kwargs)

    return dset, loader


__all__ = ["get_trn_loader", "get_val_loader", "get_test_loader", "get_filtered_chars", "read_font", "render"]
