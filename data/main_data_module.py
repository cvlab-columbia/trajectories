from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import omegaconf
import torch.utils.data
from pytorch_lightning import LightningDataModule
from pytorch_lightning.trainer.states import TrainerFn
from torch.utils.data import DataLoader

_all__ = ['MainDataModule']


@dataclass
class MainDataModule(LightningDataModule):
    """This is NOT a Dataset class, it is a LightningModule wrapper around it"""
    dataset_class: classmethod(torch.utils.data.Dataset)
    dataset_params: omegaconf.dictconfig.DictConfig[str, Any]  # Dataset params
    dataloader_params: omegaconf.dictconfig.DictConfig[str, Any]  # Dataloader params
    split_use: str = None

    def __post_init__(self):
        super().__init__()
        if self.dataset_params is None:
            self.dataset_params = omegaconf.dictconfig.DictConfig({})
        self.data = {}

    def setup(self, stage: TrainerFn = None):
        if stage.name in ['PREDICTING', 'TESTING']:
            splits = (self.split_use,) if self.split_use is not None else ('test',)  # test is default
        elif stage.name == 'VALIDATING':
            splits = ('validate',)
        elif stage.name == 'FITTING':
            splits = ('train', 'validate')
        else:
            raise KeyError
        info_all_splits = None

        for split in splits:
            self.data[split] = self.dataset_class(split, info_all_splits=info_all_splits, **self.dataset_params)
            info_all_splits = self.data[split].info_all_splits

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader('train')

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader('validate')

    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader('test')

    def predict_dataloader(self) -> DataLoader:
        """
        Combine train, val, test datasets and load the data as in test mode
        """
        dataset_use = 'test' if self.split_use is None else self.split_use
        return self._get_dataloader(dataset_use)

    def _get_dataloader(self, split: str):
        shuffle = (split == 'train')
        drop_last = (split == 'train')
        collate_fn_ = self.data[split].collate_fn
        return DataLoader(self.data[split], shuffle=shuffle, drop_last=drop_last, collate_fn=collate_fn_,
                          **self.dataloader_params)
