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

from src.models.acoustic_model import FTDNNAcoustic

from src.utils.kaldi import (
    load_ivector_period_from_conf,
    generate_df_phones_pure,
    extract_features_using_kaldi,
    load_config
)

def initialize(transition_model_path, tree_path, lang_graph_path, \
    words_path, disam_path, phones_path, word_boundary_path, acoustic_model_path, num_senones):

    aligner = MappedAligner.from_files(
        transition_model_path, tree_path, 
        lang_graph_path, words_path,
        disam_path, beam=40.0, acoustic_scale=1.0)
    
    phones  = SymbolTable.read_text(phones_path)
    word_boundary_info = WordBoundaryInfo.from_file(
        WordBoundaryInfoNewOpts(),
        word_boundary_path)

    acoustic_model = FTDNNAcoustic(num_senones=num_senones, device_name="cuda")
    acoustic_model.load_state_dict(torch.load(acoustic_model_path))
    acoustic_model.eval()

    return aligner, phones, word_boundary_info, acoustic_model

class Aligner(object):
    def __init__(self, configs):
        self.data_dir = configs["data-dir"]
        self.conf_path = configs["conf-path"]

        self.wav_scp_path = f'{self.data_dir}/wav.scp'
        self.text_path = f'{self.data_dir}/text'
        self.spk2utt_path = f'{self.data_dir}/spk2utt'
        self.mfcc_path = f'{self.data_dir}/mfcc.ark'
        self.ivectors_path = f'{self.data_dir}/ivectors.ark'
        self.feats_scp_path = f'{self.data_dir}/feats.scp'

        self.acoustic_model_path = configs['acoustic-model-path']
        self.transition_model_path = configs['transition-model-path']
        self.tree_path = configs['tree-path']
        self.disam_path = configs['disambig-path']
        self.word_boundary_path = configs['word-boundary-path']
        self.lang_graph_path = configs['lang-graph-path']
        self.words_path = configs['words-path']
        self.phones_path = configs['kaldi-phones-path']
        
        self.device = configs["device"]
        self.final_mdl_path = configs["kaldi-chain-mdl-path"]
        self.phones_pure_path = configs["phones-pure-path"]
        self.phones_to_pure_int_path = configs["phone-to-pure-phone-path"]
        self.num_senones = configs["num_senones"]
        self.transition_dir = configs["trans-dir"]

        self.df_phones_pure = self.prepare_df_phones_pure()
        self.phone_pure_senones_matrix = self.load_phone_pure_senone_matrix().to(self.device)
        self.phone_pure_to_id = self.df_phones_pure.set_index("phone_name")["phone_pure"].to_dict()

        self.lexicon = self.load_lexicon(configs['lexicon-path'])

        self.aligner, self.phones, self.word_boundary_info, self.acoustic_model = \
            initialize(
                transition_model_path=self.transition_model_path, 
                tree_path=self.tree_path, 
                lang_graph_path=self.lang_graph_path, 
                words_path=self.words_path, 
                disam_path=self.disam_path, 
                phones_path=self.phones_path, 
                word_boundary_path=self.word_boundary_path, 
                acoustic_model_path=self.acoustic_model_path,
                num_senones=self.num_senones
            )
        self.ivector_period = load_ivector_period_from_conf(self.conf_path)
        self.acoustic_model.eval().cuda()

    def load_phone_pure_senone_matrix(self):
        phone_pure_senone_matrix = matrix_gop_robust(
            df_phones_pure=self.df_phones_pure,
            number_senones=self.num_senones, 
            batch_size=1)
        
        phone_pure_senone_matrix = torch.tensor(phone_pure_senone_matrix).float()
        return phone_pure_senone_matrix.transpose(2, 1)

    def prepare_df_phones_pure(self):
        df_transitions = generate_df_phones_pure(
            phones_path=self.phones_path, 
            phones_to_pure_int_path=self.phones_to_pure_int_path, 
            phones_pure_path=self.phones_pure_path, 
            final_mdl_path=self.final_mdl_path, 
            transition_dir=self.transition_dir
        )

        df_phones_pure = df_transitions
        df_phones_pure = df_phones_pure.reset_index()

        return df_phones_pure
    
    def load_lexicon(self, path):
        lexicon = pd.read_csv(
            path, names=["word", "arapa"], sep="\t")
        
        vocab = set(lexicon["word"].to_list())

        return vocab
    
    def is_valid(self, transcript):
        words = transcript.split()
        for word in words:
            if word not in self.lexicon:
                return False
        
        return True
    
    def prepare_batch_data_in_kaldi_format(self, ids, data_dir, wav_paths, transcripts):
        wavscp_file = open(f'{data_dir}/wav.scp', "w", encoding="utf-8")
        text_file = open(f'{data_dir}/text', "w", encoding="utf-8")
        spk2utt_file = open(f'{data_dir}/spk2utt', "w", encoding="utf-8")
        utt2spk_file = open(f'{data_dir}/utt2spk', "w", encoding="utf-8")

        assert len(wav_paths) == len(transcripts)
        assert len(wav_paths) == len(ids)

        for index in range(len(wav_paths)):
            id = ids[index]
            wav_path = wav_paths[index]
            transcript = transcripts[index]
            
            wavscp_file.write(f'{id}\t{wav_path}\n')
            text_file.write(f'{id}\t{transcript}\n')
            spk2utt_file.write(f'{id}\t{id}\n')
            utt2spk_file.write(f'{id}\t{id}\n')
        
        wavscp_file.close()
        text_file.close()
        spk2utt_file.close()
        utt2spk_file.close()

        print(f'###saved data to {data_dir}')
    
    def run_batch(self, ids, wav_paths, transcripts):
        pid = os.getpid()
        data_dir = f'{self.data_dir}/{pid}'
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        wav_scp_path = f'{data_dir}/wav.scp'
        spk2utt_path = f'{data_dir}/spk2utt'
        mfcc_path = f'{data_dir}/mfcc.ark'
        ivectors_path = f'{data_dir}/ivector.ark'
        feats_scp_path = f'{data_dir}/feat.scp'

        self.prepare_batch_data_in_kaldi_format(
            ids=ids,
            data_dir=data_dir,
            wav_paths=wav_paths,
            transcripts=transcripts
        )
        extract_features_using_kaldi(
            conf_path=self.conf_path, 
            wav_scp_path=wav_scp_path, 
            spk2utt_path=spk2utt_path, 
            mfcc_path=mfcc_path, 
            ivectors_path=ivectors_path, 
            feats_scp_path=feats_scp_path
        )

        ids, alignments, scores_phone_pure, lengths = self.run_align_batch(
            ids=ids,
            transcripts=transcripts, 
            mfcc_path=mfcc_path,
            ivectors_path=ivectors_path)

        return ids, alignments, scores_phone_pure.tolist(), lengths.tolist()
    
    def pad_1d(self, inputs, max_length=None, pad_value=0.0):
        if max_length is None:
            max_length = max([sample.shape[0] for sample in inputs])     
            
        attention_masks = []
        for i in range(len(inputs)):
            if inputs[i].shape[0] < max_length:
                attention_mask = [1]*inputs[i].shape[0] + [0]*(max_length-inputs[i].shape[0])
                
                padding = torch.ones(
                    (max_length-inputs[i].shape[0], inputs[i].shape[-1]))
                
                inputs[i] = torch.cat((inputs[i], padding), dim=0)

            elif inputs[i].shape[0] >= max_length:
                inputs[i] = inputs[i][:, 0:max_length]

                attention_mask = [1]*max_length

            attention_mask = torch.tensor(attention_mask)
            attention_masks.append(attention_mask)
        
        return {
            "inputs": torch.stack(inputs),
            "attention_mask": torch.vstack(attention_masks)
        }

    def run_align_batch(self, ids, transcripts, mfcc_path, ivectors_path):
        mfccs_rspec = ("ark:" + mfcc_path)
        ivectors_rspec = ("ark:" + ivectors_path)

        features = []
        mfccs_reader = RandomAccessMatrixReader(mfccs_rspec)
        ivectors_reader = RandomAccessMatrixReader(ivectors_rspec)

        for id in ids:
            mfccs = mfccs_reader[id]
            ivectors = ivectors_reader[id]

            ivectors = np.repeat(ivectors, self.ivector_period, axis=0) 
            ivectors = ivectors[:mfccs.shape[0],:]
            x = np.concatenate((mfccs,ivectors), axis=1)
            feats = torch.from_numpy(x)

            features.append(feats)

        padded = self.pad_1d(inputs=features, pad_value=0.0)

        features = padded["inputs"].cuda()
        attention_mask = padded["attention_mask"]
        lengths = attention_mask.sum(1).cpu()

        with torch.no_grad():
            logits = self.acoustic_model(features)
            scores_phone_pure = torch.softmax(logits.cuda(), dim=-1)
            
            scores_phone_pure = torch.matmul(
                scores_phone_pure, self.phone_pure_senones_matrix)
                
            scores_phone_pure = torch.log(scores_phone_pure)

        logits = logits.cpu()
        scores_phone_pure = scores_phone_pure.cpu()

        alignments = []
        for index in range(logits.shape[0]):
            logit = logits[index].detach().numpy()
            length = lengths[index]

            logit = logit[0:length]
            transcript = transcripts[index]

            log_likes = Matrix(logit)
            output = self.aligner.align(log_likes, transcript)

            phone_alignment = self.aligner.to_phone_alignment(output["alignment"], self.phones)
            alignments.append(phone_alignment)
        
        return ids, alignments, scores_phone_pure, lengths
    
    def run_align(self, transcript, wav_path, mfcc_path, ivectors_path):
        mfccs_rspec = ("ark:" + mfcc_path)
        ivectors_rspec = ("ark:" + ivectors_path)

        mfccs_reader = RandomAccessMatrixReader(mfccs_rspec)
        ivectors_reader = RandomAccessMatrixReader(ivectors_rspec)

        mfccs = mfccs_reader[self.ID]
        ivectors = ivectors_reader[self.ID]

        ivectors = np.repeat(ivectors, self.ivector_period, axis=0) 
        ivectors = ivectors[:mfccs.shape[0],:]
        x = np.concatenate((mfccs,ivectors), axis=1)
        feats = torch.from_numpy(x).unsqueeze(0).cuda()

        with torch.no_grad():
            logits = self.acoustic_model(feats)

        log_likes = Matrix(logits.cpu().detach().numpy()[0])
        output = self.aligner.align(log_likes, transcript)

        phone_alignment = self.aligner.to_phone_alignment(output["alignment"], self.phones)

        return phone_alignment, logits

def get_pdfs_for_pure_phone(df_phones_pure, phone):
    pdfs = list(df_phones_pure.loc[(df_phones_pure['phone_pure'] == str(phone+1) )].forward_pdf)
    pdfs = pdfs + list(df_phones_pure.loc[(df_phones_pure['phone_pure'] == str(phone+1) )].self_pdf)
    pdfs = set(pdfs)
    return pdfs

def matrix_gop_robust(df_phones_pure, number_senones, batch_size):            
    pdfs_to_phone_pure_mask = []

    for phone_pure in range(0, len(list(df_phones_pure.phone_pure.unique()))):
        pdfs = get_pdfs_for_pure_phone(df_phones_pure, phone_pure)
        pdfs_to_phone_pure_file = np.zeros(number_senones)

        for pdf in pdfs:
            pdfs_to_phone_pure_file[int(pdf)] = 1.0 
        
        pdfs_to_phone_pure_mask.append(pdfs_to_phone_pure_file.tolist())
                                
    pdfs_to_phone_pure_mask_3D = []

    for i in range(0, batch_size):                
        pdfs_to_phone_pure_mask_3D.append(pdfs_to_phone_pure_mask)
    
    return pdfs_to_phone_pure_mask_3D

class GOP(object):
    def __init__(self, configs):
        print(f'###use device: {configs["device"]}')
        self.device = configs["device"]
        self.phones_path = configs["kaldi-phones-path"]
        self.final_mdl_path = configs["kaldi-chain-mdl-path"]
        self.phones_pure_path = configs["phones-pure-path"]
        self.phones_to_pure_int_path = configs["phone-to-pure-phone-path"]
        self.num_senones = configs["num_senones"]
        self.transition_dir = configs["trans-dir"]

        self.df_phones_pure = self.prepare_df_phones_pure()
        self.phone_pure_to_id = self.df_phones_pure.set_index("phone_name")["phone_pure"].to_dict()
    
    def prepare_df_phones_pure(self):
        df_transitions = generate_df_phones_pure(
            phones_path=self.phones_path, 
            phones_to_pure_int_path=self.phones_to_pure_int_path, 
            phones_pure_path=self.phones_pure_path, 
            final_mdl_path=self.final_mdl_path, 
            transition_dir=self.transition_dir
        )
        df_phones_pure = df_transitions
        df_phones_pure = df_phones_pure.reset_index()

        return df_phones_pure

    def run_batch(self, ids, alignments, scores_phone_pures, lengths):
        return self.run_gop_batch(
            ids=ids,
            scores_phone_pure=scores_phone_pures, 
            alignments=alignments, 
            lengths=lengths
        )
    
    def calculate_lpps(self, ids, alignments, features, lengths):
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
    
    def run_gop_batch(self, ids, scores_phone_pure, alignments, lengths):
        ids, lpps_batch, phonemes_batch = self.calculate_lpps(ids, alignments, scores_phone_pure, lengths)

        features = []
        for lpps, phonemes in zip(lpps_batch, phonemes_batch):
            phone_pure_ids = [
                [index, int(self.phone_pure_to_id[phone_pure[0]]) - 1] for index, phone_pure in enumerate(phonemes)]
            
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
    
if __name__ == "__main__":
    configs = load_config("configs/config_prep.yaml")

    wav_path = "/data/codes/serving/wav/She's paying for some jewelry at a checkout.wav"
    transcript = "She's paying for some jewelry at a check out".upper()

    aligner = Aligner(configs=configs)
    gop_recipe = GOP(configs=configs)

    start_time = time()
    alignments, logits, attention_mask = aligner.run_batch(
        wav_paths=[wav_path, ], 
        transcripts=[transcript, ], 
        utt_ids=["8888"])
    
    features, phonemes = gop_recipe.run_batch(
        alignments=alignments, 
        logits=logits, 
        attention_mask=attention_mask
    )

    end_time = time()

    # print("duration: ", end_time-start_time)
    # print(phonemes)
