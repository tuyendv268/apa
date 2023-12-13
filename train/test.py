from kaldiio import ReadHelper
from kaldi.util.table import RandomAccessMatrixReader
from tqdm import tqdm 
import pandas as pd
import numpy as np
import pickle
import os

def load_ivector_period_from_conf(conf_path):
    conf_fh = open(conf_path + '/ivector_extractor.conf', 'r')
    ivector_period_line = conf_fh.readlines()[1]
    ivector_period = int(ivector_period_line.split('=')[1])
    return ivector_period

conf_path = "exps/kaldi/conf"
ivector_period = load_ivector_period_from_conf(conf_path)

data_dir = "/workspace/train/data/train/0"

text_path = f'{data_dir}/text'
acoustic_features_path = f'{data_dir}/acoustic_features.pkl'
ivector_path = f"{data_dir}/ivectors.ark"
mfcc_path = f"{data_dir}/mfcc.ark"
wavscp_path = f"{data_dir}/wav.scp"
align_path = f"{data_dir}/ali.out"

text_df = pd.read_csv(
    text_path, names=["id", "text"], 
    dtype={"id":str}, sep="\t", index_col=0
)
text_df = text_df.to_dict()["text"]

mfccs_rspec = ("ark:" + mfcc_path)
ivectors_rspec = ("ark:" + ivector_path)

mfccs_reader = RandomAccessMatrixReader(mfccs_rspec)
ivectors_reader = RandomAccessMatrixReader(ivectors_rspec)
align_file = open(align_path,"w+")

acoustic_features = {}

for line in tqdm(open(wavscp_path, "r").readlines(), desc=f"Align (pid={os.getpid()})"):
    _id, _path = line.split("\t")

    mfccs = mfccs_reader[_id]
    ivectors = ivectors_reader[_id]
    text = text_df[_id]

    ivectors = np.repeat(ivectors, ivector_period, axis=0) 
    ivectors = ivectors[:mfccs.shape[0],:]
    features = np.concatenate((mfccs, ivectors), axis=1)

    acoustic_features[_id] = features

with open(acoustic_features_path, 'wb') as handle:
    pickle.dump(
        acoustic_features, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'##saved gop to: {acoustic_features_path}')
