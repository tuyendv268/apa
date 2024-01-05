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
    def __init__(
        self, ids, phone_ids_path, word_ids_path, \
        phone_scores_path, word_scores_path, sentence_scores_path, fluency_score_path, intonation_scores_path, \
        durations_path, gops_path, relative_positions_path, wavlm_features_path, relative2id_path, phone2id_path, max_length=128
    ):
        
        self.ids = ids
        self.phone_ids = IndexedDataset(phone_ids_path)
        self.word_ids = IndexedDataset(word_ids_path)

        self.phone_scores = IndexedDataset(phone_scores_path)
        self.word_scores = IndexedDataset(word_scores_path)
        self.sentence_scores = IndexedDataset(sentence_scores_path)
        self.fluency_scores = IndexedDataset(fluency_score_path)
        self.intonation_scores = IndexedDataset(intonation_scores_path)

        self.gops = IndexedDataset(gops_path)
        self.durations = IndexedDataset(durations_path)
        self.wavlm_features = IndexedDataset(wavlm_features_path)
        self.relative_positions = IndexedDataset(relative_positions_path)
        self.max_length = max_length

        self.relative2id = json.load(open(relative2id_path, "r", encoding="utf-8"))
        self.phone2id = json.load(open(phone2id_path, "r", encoding="utf-8"))

    def __len__(self):
        return len(self.ids)

    def pad_1d(self, inputs, pad_value=0):
        padding = pad_value*torch.ones(self.max_length-inputs.shape[0])
        inputs = torch.concat([inputs, padding], axis=0)

        return inputs

    def pad_2d(self, inputs, pad_value=0):
        padding = pad_value * torch.ones(self.max_length-inputs.shape[0], inputs.shape[-1])
        inputs = torch.concat([inputs, padding], axis=0)

        return inputs

    def padding(self, phone_ids, word_ids, phone_scores, word_scores, \
            sentence_scores, durations, gops, wavlm_features, relative_positions):

        phone_ids = self.pad_1d(phone_ids, pad_value=self.phone2id["PAD"])
        word_ids = self.pad_1d(word_ids, pad_value=-1)
        phone_scores = self.pad_1d(phone_scores, pad_value=-1)
        word_scores = self.pad_1d(word_scores, pad_value=-1)
        durations = self.pad_1d(durations, pad_value=0)
        relative_positions = self.pad_1d(relative_positions, pad_value=self.relative2id["PAD"])

        gops = self.pad_2d(gops, pad_value=0)
        wavlm_features = self.pad_2d(wavlm_features, pad_value=0)

        return phone_ids, word_ids, phone_scores, word_scores, \
            sentence_scores, durations, gops, wavlm_features, relative_positions
    
    def parse_data(self, ids, phone_ids, word_ids, phone_scores, word_scores, \
            sentence_scores, fluency_scores, intonation_scores, durations, gops, wavlm_features, relative_positions):
        
        phone_ids = torch.from_numpy(phone_ids)
        word_ids = torch.from_numpy(word_ids)

        phone_scores = torch.from_numpy(phone_scores).float().clone()
        word_scores = torch.from_numpy(word_scores).float().clone()
        sentence_scores = torch.tensor(sentence_scores).float().clone()
        fluency_scores = torch.tensor(fluency_scores).float().clone()
        intonation_scores = torch.tensor(intonation_scores).float().clone()

        durations = torch.from_numpy(durations).float()
        gops = torch.from_numpy(gops).float()
        wavlm_features = torch.from_numpy(wavlm_features).float()
        relative_positions = torch.from_numpy(relative_positions)

        phone_ids, word_ids, phone_scores, word_scores, \
            sentence_scores, durations, gops, wavlm_features, relative_positions = self.padding(
                phone_ids, word_ids, phone_scores, word_scores, \
                sentence_scores, durations, gops, wavlm_features, relative_positions
            )

        features = torch.concat([gops, durations.unsqueeze(-1), wavlm_features], dim=-1)   

        phone_scores[phone_scores != -1] /= 50
        word_scores[word_scores != -1] /= 50
        sentence_scores /= 50
        fluency_scores /= 50
        intonation_scores /= 50
     
        return {
            "ids": ids,
            "features": features,
            "phone_ids": phone_ids,
            "word_ids": word_ids,
            "phone_scores": phone_scores,
            "word_scores": word_scores,
            "sentence_scores": sentence_scores,
            "fluency_scores": fluency_scores,
            "intonation_scores": intonation_scores,
            "relative_positions": relative_positions
        }
        
    def __getitem__(self, index):
        phone_ids = self.phone_ids[index]
        word_ids = self.word_ids[index]
        ids = self.ids[index]

        phone_scores = self.phone_scores[index]
        word_scores = self.word_scores[index]
        sentence_scores = self.sentence_scores[index]
        fluency_scores = self.fluency_scores[index]
        intonation_scores = self.intonation_scores[index]

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
            fluency_scores=fluency_scores,
            intonation_scores=intonation_scores,
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
