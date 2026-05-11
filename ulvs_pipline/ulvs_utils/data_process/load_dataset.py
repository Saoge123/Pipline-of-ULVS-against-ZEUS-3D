import os
import lmdb
import torch
from torch.utils.data import Dataset, Subset, DataLoader
import random
import pickle
from tqdm.auto import tqdm


class LoadDataset(Dataset):
    def __init__(self, dataset, transform=None):
        if isinstance(dataset, str) and os.path.exists(dataset) and '.pkl' in dataset:
            with open(dataset, 'rb') as frb:
                self.dataset = pickle.load(frb)
                self.is_lmdb = False
            if isinstance(self.dataset, dict):
                self.keys = list(self.dataset.keys())
            else:
                self.keys = list(range(len(self.dataset)))
        elif isinstance(dataset, str) and os.path.exists(dataset) and '.lmdb' in dataset:
            self.dataset = self.connect_db(dataset)
            self.is_lmdb = True
            with self.dataset.begin() as txn:
                self.keys = list(txn.cursor().iternext(values=False))
        else:
            self.dataset = dataset
            self.is_lmdb = False
            self.keys = list(self.dataset.keys())
        self.transform = transform

    def connect_db(self, lmdb_path):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
            )
        return env
    
    def close(self):
        self.dataset.close()

    def __getitem__(self, index):
        if self.is_lmdb:
            data = self.dataset.begin().get(f"{index}".encode("ascii"))
            data = pickle.loads(data)
        else:
            data = self.dataset[self.keys[index]]
        if self.transform is not None:
            return self.transform(data)
        return data
    
    def __len__(self):
        return len(self.keys)