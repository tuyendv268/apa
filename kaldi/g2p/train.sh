#!/usr/bin/env bash

# mfa train_g2p \
#     --config_path config.yml \
#     --num_jobs 16 \
#     --phonetisaurus \
#     --clean \
#     /data/codes/apa/kaldi/g2p/lexicon/processed/lexicon.txt \
#     /data/codes/apa/kaldi/g2p/exp/g2p-model.zip

mfa train_g2p \
    --config_path config.yml \
    --num_jobs 16 \
    --phonetisaurus \
    --clean \
    /data/codes/apa/kaldi/g2p/lexicon/lexicon.txt \
    /data/codes/apa/kaldi/g2p/exp/g2p-cmu-prep-elsa.zip