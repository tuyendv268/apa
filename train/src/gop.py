from kaldi.util.table import RandomAccessMatrixReader
from kaldi.util.table import DoubleMatrixWriter
from kaldi.alignment import MappedAligner
from kaldi.fstext import SymbolTable
from kaldi.matrix import Matrix
from kaldi.lat.align import (
    WordBoundaryInfoNewOpts, 
    WordBoundaryInfo)
    
from scipy.special import softmax
from time import time
from tqdm import tqdm
from glob import glob
import pandas as pd
import numpy as np
import argparse
import shutil
import torch
import base64
import uuid
import json
import yaml
import sys
import os

from src.utils.kaldi import (
    load_ivector_period_from_conf,
    generate_df_phones_pure,
    extract_features_using_kaldi,
    load_config
)

def calculate_lpps(ids, alignments, features, lengths):
    batch_size, seq_length, embedd_dim = features.shape[0], features.shape[1], features.shape[2]
    
    indices = -1 * torch.ones(batch_size, seq_length)
    phonemes_batch = []
    for sid in range(len(ids)):
        phonemes, index = [], 0

        alignment = alignments[sid]
        for phoneme, start_frame, duration in alignment:
            if phoneme == "SIL":
                continue
            end_frame = start_frame + duration
            indices[sid][start_frame:end_frame] = index
            phonemes.append((phoneme, start_frame, duration))
            index += 1

        phonemes_batch.append(phonemes)

    phone_lengths = [len(alignment) for alignment in phonemes_batch]
    max_index = indices.max()

    for sid in range(len(ids)):
        length = lengths[sid]
        max_current_index = indices[sid].max()

        tmp_index = max_current_index + 1
        for index in range(length, seq_length):

            indices[sid][index] = tmp_index
            if tmp_index < max_index:
                tmp_index += 1

    indices[indices==-1] = indices.max() + 1

    indices = torch.nn.functional.one_hot(indices.long(), num_classes=int(indices.max().item())+1).cuda()
    indices = indices / indices.sum(1, keepdim=True)
    
    features = torch.matmul(indices.transpose(1, 2), features)
    gop_features = []
    for index in range(len(ids)):
        length = phone_lengths[index]
        feature = features[index]
        gop_features.append(feature[0:length])

    return ids, gop_features, phonemes_batch

def prepare_df_phones_pure(
        phones_path, phones_to_pure_int_path, phones_pure_path, final_mdl_path, transition_dir):
    df_transitions = generate_df_phones_pure(
        phones_path=phones_path, 
        phones_to_pure_int_path=phones_to_pure_int_path, 
        phones_pure_path=phones_pure_path, 
        final_mdl_path=final_mdl_path, 
        transition_dir=transition_dir
    )

    df_phones_pure = df_transitions.reset_index()

    return df_phones_pure
def run_gop_batch(ids, scores_phone_pure, alignments, lengths, phone_pure_to_id):
    ids, lpps_batch, phonemes_batch = calculate_lpps(ids, alignments, scores_phone_pure, lengths)

    features = []
    for lpps, phonemes in zip(lpps_batch, phonemes_batch):
        phone_pure_ids = [
            [index, int(phone_pure_to_id[phone_pure[0]]) - 1] for index, phone_pure in enumerate(phonemes)]
        
        lprs = torch.Tensor(
            [
                lpps[index[0], index[1]] 
                for index in phone_pure_ids
                ]
            ).cuda()
        
        lprs = lprs.unsqueeze(-1).expand(-1, lpps.shape[-1]).cuda()

        lprs = lpps - lprs
        feats = torch.cat([lpps, lprs], dim=-1)
        features.append(feats.cpu().tolist())
    
    return ids, features, phonemes_batch

