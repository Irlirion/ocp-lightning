from typing import Optional
from functools import partial

from torch.utils.data import DataLoader

import pytorch_lightning as pl

from src.datamodules.datasets.single_point_lmdb_datasets import SinglePointLmdbDataset, data_list_collater


class SinglePointLmdbDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        val_dir: str,
        test_dir: Optional[str] = None,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = True,
        is_otf_graph: bool = False,
    ):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.collater = partial(data_list_collater, otf_graph=is_otf_graph)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.energy_train = SinglePointLmdbDataset(self.train_dir)
            self.energy_val = SinglePointLmdbDataset(self.val_dir)
        elif stage == "test" or stage is None:
            self.energy_test = SinglePointLmdbDataset(self.test_dir)

    def train_dataloader(self):
        return DataLoader(
            self.energy_train,
            batch_size=self.batch_size,
            collate_fn=self.collater,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.energy_val,
            batch_size=self.batch_size,
            collate_fn=self.collater,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.energy_test,
            batch_size=self.batch_size,
            collate_fn=self.collater,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def teardown(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.energy_train.close_db()
            self.energy_val.close_db()
        elif stage == "test" or stage is None:
            self.energy_test.close_db()
