import os
from pathlib import Path

import bisect
import pickle
from pathlib import Path

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch


class TrajectoryLmdbDataset(Dataset):
    r"""Dataset class to load from LMDB files containing relaxation trajectories.
    Useful for Structure to Energy & Force (S2EF) and Initial State to
    Relaxed State (IS2RS) tasks.

    Args:
        dir (str): Path to dir with LMDB file
        transform (callable, optional): Data transform function.
            (default: :obj:`None`)
    """

    def __init__(self, dir, transform=None):
        super(TrajectoryLmdbDataset, self).__init__()

        srcdir = Path(dir)
        db_paths = sorted(srcdir.glob("*.lmdb"))
        assert len(db_paths) > 0, f"No LMDBs found in {srcdir}"

        self._keys, self.envs = [], []
        for db_path in db_paths:
            self.envs.append(self.connect_db(db_path))
            length = pickle.loads(self.envs[-1].begin().get("length".encode("ascii")))
            self._keys.append(list(range(length)))

        keylens = [len(k) for k in self._keys]
        self._keylen_cumulative = np.cumsum(keylens).tolist()
        self.transform = transform
        self.num_samples = sum(keylens)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Figure out which db this should be indexed from.
        db_idx = bisect.bisect(self._keylen_cumulative, idx)
        # Extract index of element within that db.
        el_idx = idx
        if db_idx != 0:
            el_idx = idx - self._keylen_cumulative[db_idx - 1]
        assert el_idx >= 0

        # Return features.
        datapoint_pickled = (
            self.envs[db_idx]
            .begin()
            .get(f"{self._keys[db_idx][el_idx]}".encode("ascii"))
        )
        data_object = pickle.loads(datapoint_pickled)
        if self.transform is not None:
            data_object = self.transform(data_object)

        data_object.id = f"{db_idx}_{el_idx}"

        return data_object

    def connect_db(self, lmdb_path=None):
        env = lmdb.open(
            str(lmdb_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
        )
        return env

    def close_db(self):
        for env in self.envs:
            env.close()


class SinglePointLmdbDataset(Dataset):
    r"""Dataset class to load from LMDB files containing single point computations.
    Useful for Initial Structure to Relaxed Energy (IS2RE) task.

    Args:
        dir (str): Path to LMDB file
        transform (callable, optional): Data transform function.
            (default: :obj:`None`)
    """

    def __init__(self, dir, transform=None):
        super(SinglePointLmdbDataset, self).__init__()

        self.db_path = dir
        assert os.path.isfile(self.db_path), f"{self.db_path} not found"

        self.env = self.connect_db(self.db_path)

        self._keys = [f"{j}".encode("ascii") for j in range(self.env.stat()["entries"])]
        self.transform = transform

    def __len__(self):
        return len(self._keys)

    def __getitem__(self, idx):
        # Return features.
        datapoint_pickled = self.env.begin().get(self._keys[idx])
        data_object = pickle.loads(datapoint_pickled)
        data_object = (
            data_object if self.transform is None else self.transform(data_object)
        )

        return data_object

    def connect_db(self, lmdb_path=None):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
        )
        return env

    def close_db(self):
        self.env.close()


def data_list_collater(data_list, otf_graph=False):
    batch = Batch.from_data_list(data_list)

    if not otf_graph:
        try:
            n_neighbors = []
            for i, data in enumerate(data_list):
                n_index = data.edge_index[1, :]
                n_neighbors.append(n_index.shape[0])
            batch.neighbors = torch.tensor(n_neighbors)
        except NotImplementedError:
            print("LMDB does not contain edge index information, set otf_graph=True")
    return batch
