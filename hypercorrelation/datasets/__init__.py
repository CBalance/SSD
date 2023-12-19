# -*- coding: utf-8 -*-

from torch.utils.data import DataLoader
from .dataset import FSC147, CARPK


def build_loader(config, mode, class_remove=False):
    data_path = config.DATA_PATH
    if mode == 'train':
        batch_size = config.BATCH_SIZE
    else:
        batch_size = 1
    num_workers = config.NUM_WORKERS
    train_set = FSC147(data_path, mode, config.SHOT, class_remove)


    return DataLoader(
        train_set,
        batch_size = batch_size,
        num_workers = num_workers,
        shuffle = (mode=='train'),
        collate_fn=FSC147.collate_fn
    )

def build_car_loader(config):
    data_path = config.DATA_PATH
    batch_size = 1
    num_workers = config.NUM_WORKERS
    train_set = CARPK(data_path, config.SHOT)

    return DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=FSC147.collate_fn
    )
