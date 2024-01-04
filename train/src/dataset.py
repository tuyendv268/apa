from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
from tqdm import tqdm
import torch

import numpy as np
import random
import pickle
import json
import os

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
    
class PrepDataset(Dataset):
    def __init__(self, ids, phone_ids, word_ids, phone_scores, word_scores, \
            sentence_scores, durations, gops, relative_positions, wavlm_features_path):
        
        self.ids = ids
        self.phone_ids = phone_ids
        self.word_ids = word_ids

        self.phone_scores = phone_scores
        self.word_scores = word_scores
        self.sentence_scores = sentence_scores

        self.gops = gops
        self.durations = durations
        self.wavlm_features = IndexedDataset(wavlm_features_path)
        self.relative_positions = relative_positions

    def __len__(self):
        return self.phone_ids.shape[0]
    
    def parse_data(self, ids, phone_ids, word_ids, phone_scores, word_scores, \
            sentence_scores, durations, gops, wavlm_features, relative_positions):
        
        phone_ids = torch.tensor(phone_ids)
        word_ids = torch.tensor(word_ids)

        phone_scores = torch.tensor(phone_scores).float().clone()
        word_scores = torch.tensor(word_scores).float().clone()
        sentence_scores = torch.tensor(sentence_scores).float().clone()

        phone_scores[phone_scores != -1] /= 50
        word_scores[word_scores != -1] /= 50
        sentence_scores /= 50

        durations = torch.tensor(durations)
        gops = torch.tensor(gops)
        wavlm_features = torch.tensor(wavlm_features)
        relative_positions = torch.tensor(relative_positions)

        features = torch.concat([gops, durations.unsqueeze(-1), wavlm_features], dim=-1)        
        return {
            "ids": ids,
            "features": features,
            "phone_ids": phone_ids,
            "word_ids": word_ids,
            "phone_scores":phone_scores,
            "word_scores":word_scores,
            "sentence_scores":sentence_scores,
            "relative_positions": relative_positions
        }
        
    def __getitem__(self, index):
        phone_ids = self.phone_ids[index]
        word_ids = self.word_ids[index]
        ids = self.ids[index]

        phone_scores = self.phone_scores[index]
        word_scores = self.word_scores[index]
        sentence_scores = self.sentence_scores[index]

        gops = self.gops[index]
        durations = self.durations[index]
        wavlm_features = self.wavlm_features[index]
        relative_positions = self.relative_positions[index]

        return self.parse_data(
            ids=ids,
            phone_ids=phone_ids,
            word_ids=word_ids,
            phone_scores=phone_scores,
            word_scores=word_scores,
            sentence_scores=sentence_scores,
            gops=gops,
            durations=durations,
            wavlm_features=wavlm_features,
            relative_positions=relative_positions
        )
    
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
