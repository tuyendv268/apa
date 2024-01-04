#!/usr/bin/env bash
. ./path.sh

nnet3-copy --binary=false \
    exp/kaldi/exp/chain_cleaned/tdnn_1d_sp_2k1/final.mdl \
    exp/kaldi/exp/chain_cleaned/tdnn_1d_sp_2k1/final.txt

python convert-ftdnn-to-torch.py