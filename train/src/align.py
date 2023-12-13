from kaldi.util.table import RandomAccessMatrixReader
from kaldi.util.table import DoubleMatrixWriter
from kaldi.alignment import MappedAligner
from kaldi.fstext import SymbolTable
from kaldi.matrix import Matrix
from kaldi.lat.align import (
    WordBoundaryInfoNewOpts, 
    WordBoundaryInfo)

from torchaudio.kaldi_io import (
    read_mat_ark
)

from multiprocessing.pool import Pool
from tqdm import tqdm
from glob import glob
import pandas as pd
import numpy as np
import argparse
import shutil
import torch
import json
import yaml
import sys
import os

from src.models.acoustic_model import FTDNNAcoustic
from src.utils.kaldi import (
    load_ivector_period_from_conf,
    generate_df_phones_pure,
    extract_features_using_kaldi,
)

def load_data(metadata_path, wav_dir):    
    data = pd.read_csv(
        metadata_path, names=["path", "text"], sep="|")
    
    data["path"] = data.path.apply(lambda x: os.path.join(wav_dir, f'{x}.wav'))
    data["id"] = data.path.apply(lambda x: os.path.basename(x).split(".wav")[0])
    
    return data

def prepare_data_in_kaldi_format(data, data_dir):
    data.sort_values("id")
    
    wavscp_path = f'{data_dir}/wav.scp'
    text_path = f'{data_dir}/text'
    spk2utt_path = f'{data_dir}/spk2utt'
    utt2spk_path = f'{data_dir}/utt2spk'
    wavscp_file = open(wavscp_path, "w", encoding="utf-8")
    text_file = open(text_path, "w", encoding="utf-8")
    spk2utt_file = open(spk2utt_path, "w", encoding="utf-8")
    utt2spk_file = open(utt2spk_path, "w", encoding="utf-8")
    
    for index in data.index:
        wavscp = f'{data["id"][index]}\t{data["path"][index]}\n'
        text = f'{data["id"][index]}\t{data["text"][index]}\n'
        spk2utt = f'{data["id"][index]}\t{data["id"][index]}\n'
        utt2spk = f'{data["id"][index]}\t{data["id"][index]}\n'
        
        wavscp_file.write(wavscp)
        text_file.write(text)
        spk2utt_file.write(spk2utt)
        utt2spk_file.write(utt2spk)
        
    wavscp_file.close()
    text_file.close()
    spk2utt_file.close()
    utt2spk_file.close()

def load_config(path):
    config_fh = open(path, "r")
    config_dict = yaml.safe_load(config_fh)
    return config_dict

def extract_features_using_kaldi(
        conf_path, wav_scp_path, spk2utt_path, mfcc_path, ivectors_path, feats_scp_path):
    
    os.system(
        'compute-mfcc-feats --config='+conf_path+'/mfcc_hires.conf \
            scp,p:' + wav_scp_path+' ark:- | copy-feats \
            --compress=true ark:- ark,scp:' + mfcc_path + ',' + feats_scp_path)
        
    os.system(
        'ivector-extract-online2 --config='+ conf_path +'/ivector_extractor.conf ark:'+ spk2utt_path + '\
            scp:' + feats_scp_path + ' ark:' + ivectors_path)

def split_data(data, n_split, out_dir):    
    split_dirs = []
    n_sample_per_split = int(data.shape[0] / n_split)
    for index in range(n_split):
        sub_data_dir = f'{out_dir}/{index}'
        if not os.path.exists(sub_data_dir):
            os.mkdir(sub_data_dir)

        sub_data = data[index*n_sample_per_split: (index+1)*n_sample_per_split]
        prepare_data_in_kaldi_format(sub_data, sub_data_dir)

        split_dirs.append(sub_data_dir)

    return split_dirs

def run_extract_feature_in_parallel(conf_path, split_dirs, n_process=5):
    params = []
    for index, data_dir in tqdm(enumerate(split_dirs)):
        wav_scp_path = f'{data_dir}/wav.scp'
        spk2utt_path = f'{data_dir}/spk2utt'
        mfcc_path = f'{data_dir}/mfcc.ark'
        ivectors_path = f'{data_dir}/ivectors.ark'
        feats_scp_path = f'{data_dir}/feats.scp'

        params.append(
            [
                conf_path, wav_scp_path, spk2utt_path, 
                mfcc_path, ivectors_path, feats_scp_path
                ]
            )

    parallel_params = []
    n = len(split_dirs)
    step = int(n / n_process) + 1

    for i in range(0, n_process):
        parallel_params.append(params[i*step: (i+1)*step])

    with Pool(processes=n_process) as pool:
        pool.starmap(
            func=extract_features_using_kaldi, 
            iterable=params)

    # extract_features_using_kaldi(conf_path, wav_scp_path, spk2utt_path, mfcc_path, ivectors_path, feats_scp_path)

def initialize(config_dict):
    acoustic_model_path = config_dict['acoustic-model-path']
    transition_model_path = config_dict['transition-model-path']
    tree_path = config_dict['tree-path']
    disam_path = config_dict['disambig-path']
    word_boundary_path = config_dict['word-boundary-path']
    lang_graph_path = config_dict['lang-graph-path']
    words_path = config_dict['words-path']
    phones_path = config_dict['libri-phones-path']
    num_senones = config_dict['num-senones']
        
    aligner = MappedAligner.from_files(
        transition_model_path, tree_path, 
        lang_graph_path, words_path,
        disam_path, acoustic_scale=1.0)
    
    phones  = SymbolTable.read_text(phones_path)
    wb_info = WordBoundaryInfo.from_file(
        WordBoundaryInfoNewOpts(),
        word_boundary_path)

    acoustic_model = FTDNNAcoustic(num_senones=num_senones)
    acoustic_model.load_state_dict(torch.load(acoustic_model_path))
    acoustic_model.eval()

    return aligner, phones, wb_info, acoustic_model

def load_ivector_period_from_conf(conf_path):
    conf_fh = open(conf_path + '/ivector_extractor.conf', 'r')
    ivector_period_line = conf_fh.readlines()[1]
    ivector_period = int(ivector_period_line.split('=')[1])
    return ivector_period

def pad_1d(inputs, max_length=None, pad_value=0.0):
    if max_length is None:
        max_length = max([sample.shape[0] for sample in inputs])     
        
    attention_masks = []
    for i in range(len(inputs)):
        if inputs[i].shape[0] < max_length:
            attention_mask = [1]*inputs[i].shape[0] + [0]*(max_length-inputs[i].shape[0])
            
            padding = pad_value * torch.ones(
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

def load_phone_pure_senone_matrix(df_phones_pure, num_senones):
    phone_pure_senone_matrix = matrix_gop_robust(
        df_phones_pure=df_phones_pure,
        number_senones=num_senones, 
        batch_size=1)
    
    phone_pure_senone_matrix = torch.tensor(phone_pure_senone_matrix).float()
    return phone_pure_senone_matrix.transpose(2, 1)

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

def run(params):
    for config_dict, data_dir in params:
        run_align(
            config_dict=config_dict,
            data_dir=data_dir
        )

def run_align(config_dict, data_dir):
    conf_path = config_dict["conf-path"]
    num_senones = config_dict["num-senones"]

    acoustic_model_path = config_dict['acoustic-model-path']
    transition_model_path = config_dict['transition-model-path']
    tree_path = config_dict['tree-path']
    disam_path = config_dict['disambig-path']
    word_boundary_path = config_dict['word-boundary-path']
    lang_graph_path = config_dict['lang-graph-path']
    words_path = config_dict['words-path']
    phones_path = config_dict['kaldi-phones-path']
    num_senones = config_dict['num-senones']
    conf_path = config_dict['conf-path']

    final_mdl_path = config_dict["kaldi-chain-mdl-path"]
    phones_pure_path = config_dict["phones-pure-path"]
    phones_to_pure_int_path = config_dict["phone-to-pure-phone-path"]
    transition_dir = config_dict["trans-dir"]

    
    aligner, phones, word_boundary_info, acoustic_model = initialize(
        transition_model_path=transition_model_path, 
        tree_path=tree_path, 
        lang_graph_path=lang_graph_path,
        words_path=words_path, 
        disam_path=disam_path, 
        phones_path=phones_path, 
        word_boundary_path=word_boundary_path, 
        acoustic_model_path=acoustic_model_path, 
        num_senones=num_senones
    )
    acoustic_model.cuda()
    ivector_period = load_ivector_period_from_conf(conf_path)

    df_phones_pure = prepare_df_phones_pure(
        phones_path=phones_path, 
        phones_to_pure_int_path=phones_to_pure_int_path, 
        phones_pure_path=phones_pure_path, 
        final_mdl_path=final_mdl_path, 
        transition_dir=transition_dir
    )
    phone_pure_senones_matrix = load_phone_pure_senone_matrix(
        df_phones_pure=df_phones_pure, 
        num_senones=num_senones).squeeze(0).cuda()
    
    assert phone_pure_senones_matrix.shape[0] != 1

    text_path = f'{data_dir}/text'
    mfcc_path = f'{data_dir}/mfcc.ark'
    ivector_path = f'{data_dir}/ivectors.ark'
    wavscp_path = f'{data_dir}/wav.scp'
    align_path = f'{data_dir}/ali.out'
    prob_path = f'{data_dir}/prob.ark'

    text_df = pd.read_csv(
        text_path, names=["id", "text"], 
        dtype={"id":str}, sep="\t", index_col=0
    )
    text_df = text_df.to_dict()["text"]

    mfccs_rspec = ("ark:" + mfcc_path)
    ivectors_rspec = ("ark:" + ivector_path)
    prob_wspec= f"ark:| copy-feats --compress=true ark:- ark:{prob_path}"

    try:
        mfccs_reader = RandomAccessMatrixReader(mfccs_rspec)
        ivectors_reader = RandomAccessMatrixReader(ivectors_rspec)
        prob_writer = DoubleMatrixWriter(prob_wspec)
        align_file = open(align_path,"w+")

        for line in tqdm(open(wavscp_path, "r").readlines(), desc=f"Align (pid={os.getpid()})"):
            _id, _path = line.split("\t")

            mfccs = mfccs_reader[_id]
            ivectors = ivectors_reader[_id]
            text = text_df[_id]

            ivectors = np.repeat(ivectors, ivector_period, axis=0) 
            ivectors = ivectors[:mfccs.shape[0],:]
            features = np.concatenate((mfccs, ivectors), axis=1)

            features = torch.tensor(features).unsqueeze(0).cuda()
            with torch.no_grad():
                logits = acoustic_model(features)
                logits = logits[0]

                scores_phone_pure = torch.matmul(
                    logits.softmax(dim=-1), phone_pure_senones_matrix)
                scores_phone_pure = torch.log(scores_phone_pure)

                logits = logits.cpu().detach().numpy()
            
            loglikes = Matrix(logits)
            alignments = aligner.align(loglikes, text)

            phone_alignment = aligner.to_phone_alignment(alignments["alignment"], phones)
            json_obj = json.dumps(phone_alignment)

            align_file.write(f'{_id}\t{json_obj}\n')
            prob_writer[_id] = scores_phone_pure.cpu().numpy()
            
        prob_writer.close()
        align_file.close()
        mfccs_reader.close()
        ivectors_reader.close()
    except:
        return

def run_align_in_parallel(split_dirs, config_dict, n_processes):
    params = []
    for data_dir in split_dirs:
        params.append(
            [config_dict, data_dir]
        )

    parallel_params = []
    n = len(split_dirs)
    assert n % n_processes == 0
    step = int(n / n_processes)

    for i in range(0, n_processes):
        parallel_params.append(params[i*step: (i+1)*step])

    with Pool(processes=n_processes) as pool:
        pool.map(
            func=run, 
            iterable=parallel_params
        )
    
    print("###Align Done!!!")
