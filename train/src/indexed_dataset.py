import numpy as np
import random
import pickle
import json
import os
from copy import deepcopy
from tqdm import tqdm

class IndexedDataset:
    def __init__(self, path: str, num_cache=1):
        super().__init__()
        self.path = path
        self.file = None
        if path.endswith(".data") or path.endswith(".idx"):
            path = os.path.splitext(path)[0]
        self.offsets = np.load(f"{path}.idx", allow_pickle=True).item()['offsets']
        self.file = open(f"{path}.data", 'rb', buffering=-1)
        self.cache = []
        self.num_cache = num_cache

    def check_index(self, i):
        if i < 0 or i >= len(self.offsets) - 1:
            raise IndexError('index out of range')

    def __del__(self):
        if self.file:
            self.file.close()

    def __getitem__(self, i):
        self.check_index(i)
        if self.num_cache > 0:
            for c in self.cache:
                if c[0] == i:
                    return c[1]
        self.file.seek(self.offsets[i])
        b = self.file.read(self.offsets[i + 1] - self.offsets[i])
        item = pickle.loads(b)
        if self.num_cache > 0:
            self.cache = [(i, deepcopy(item))] + self.cache[:-1]
        return item

    def __len__(self):
        return len(self.offsets) - 1

class IndexedDatasetBuilder:
    def __init__(self, path):
        if path.endswith(".npy") or path.endswith(".idx"):
            path = os.path.splitext(path)[0]
        self.path = path
        self.out_file = open(f"{path}.data", 'wb')
        self.byte_offsets = [0]

    def add_item(self, item):
        s = pickle.dumps(item)
        bytes = self.out_file.write(s)
        self.byte_offsets.append(self.byte_offsets[-1] + bytes)

    def finalize(self):
        self.out_file.close()
        np.save(open(f"{self.path}.idx", 'wb'), {'offsets': self.byte_offsets})


if __name__ == "__main__":
    max_length = 32
    embedd_dim = 64

    batch = []
    data_dir = "data/train"
    for i in range(0, 10):
        length = random.randint(5, 32)
        batch.append(np.random.randn(length, embedd_dim))

    builder = IndexedDatasetBuilder(data_dir)

    for sample in batch:
        builder.add_item(item=sample)

    builder.finalize()