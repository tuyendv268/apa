import os
import json
import pandas as pd
from kaldiio import ReadHelper
from tqdm import tqdm
from src.gop import (
    load_config,
    prepare_df_phones_pure
)

import glob
import pickle
from multiprocessing.pool import Pool
import argparse
import torch

def calculate_lpps(alignments, features):
    seq_length, embedd_dim = features.shape[0], features.shape[1]
    
    indices = -1 * torch.ones(seq_length)
    phonemes, index = [], 0

    for phoneme, start_frame, duration in alignments:
        if phoneme == "SIL":
            continue
        end_frame = start_frame + duration
        indices[start_frame:end_frame] = index
        phonemes.append((phoneme, start_frame, duration))
        index += 1

    if -1 in indices:
        indices[indices==-1] = indices.max() + 1

        indices = torch.nn.functional.one_hot(
            indices.long(), num_classes=int(indices.max().item())+1).cuda()
        indices = indices / indices.sum(0, keepdim=True)
        
        features = torch.matmul(indices.transpose(0, 1), features.cuda())
        return features[:-1]
    
    else:
        indices[indices==-1] = indices.max() + 1

        indices = torch.nn.functional.one_hot(
            indices.long(), num_classes=int(indices.max().item())+1).cuda()
        indices = indices / indices.sum(0, keepdim=True)
        
        features = torch.matmul(indices.transpose(0, 1), features.cuda())
        return features

def load_alignments(path):
    alignment_df = pd.read_csv(
        path, names=["id", "alignment"], sep="\t"
    )
    alignment_df["alignment"] = alignment_df.alignment.apply(json.loads)

    id2alignment = alignment_df.set_index("id")["alignment"].to_dict()
    return id2alignment

def extract_phones_from_alignments(alignment):
    phonemes = []
    for phoneme, start_frame, duration in alignment:
        if phoneme == "SIL":
            continue
        phonemes.append((phoneme, start_frame, duration))

    return phonemes

def run_gop(data_dir, phone_pure_to_id):
    align_path = f'{data_dir}/ali.out'
    prob_path = f'{data_dir}/prob.ark'
    gop_path = f'{data_dir}/gop.pkl'

    prob_rspecf = ('ark:' + prob_path)
    prob_reader = ReadHelper(prob_rspecf)

    id2alignment = load_alignments(align_path)

    id2gop = {}
    for key, loglikes in tqdm(prob_reader):
        try:
            alignment = id2alignment[int(key)]
        except:
            alignment = id2alignment[key]

        phonemes = extract_phones_from_alignments(alignment)

        loglikes = torch.from_numpy(loglikes)

        lpps = calculate_lpps(
            alignments=alignment,
            features=loglikes
        )
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
        gop_feat = torch.cat([lpps, lprs], dim=-1)

        id2gop[key] = gop_feat.cpu().numpy()
    
    with open(gop_path, 'wb') as handle:
        pickle.dump(
            id2gop, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'##saved gop to: {gop_path}')

def run(data_dirs, phone_pure_to_id):
    for data_dir in tqdm(data_dirs, desc="GOP"):
        try:
            run_gop(
                data_dir=data_dir,
                phone_pure_to_id=phone_pure_to_id
            )
        except:
            print("Some errors occur")
            continue

if __name__ == "__main__":
    config_dict = load_config("configs/general.yaml")

    n_split = config_dict["n-split"]
    phones_path = config_dict["kaldi-phones-path"]
    final_mdl_path = config_dict["kaldi-chain-mdl-path"]
    phones_pure_path = config_dict["phones-pure-path"]
    phones_to_pure_int_path = config_dict["phone-to-pure-phone-path"]
    num_senones = config_dict["num-senones"]
    transition_dir = config_dict["trans-dir"]

    df_phones_pure = prepare_df_phones_pure(
        phones_path=phones_path, 
        phones_to_pure_int_path=phones_to_pure_int_path, 
        phones_pure_path=phones_pure_path, 
        final_mdl_path=final_mdl_path, 
        transition_dir=transition_dir
    )
    phone_pure_to_id = df_phones_pure.set_index("phone_name")["phone_pure"].to_dict()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--data_dir', type=str, default="data/train/info_qt_10_trainset-aug")
    
    args = parser.parse_args()

    data_dir = args.data_dir
    data_dirs = glob.glob(f'{data_dir}/*')

    run(
        data_dirs=data_dirs, 
        phone_pure_to_id=phone_pure_to_id
    )

