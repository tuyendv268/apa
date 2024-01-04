#!/usr/bin/env bash

# mfa g2p \
#     --config_path config.yml \
#     --num_jobs 12 \
#     --num_pronunciations 1 \
#     --dictionary_path /data/codes/apa/kaldi/g2p/data/train \
#     /data/codes/apa/kaldi/g2p/data/test-word.txt \
#     /data/codes/apa/kaldi/g2p/exp/g2p-cmu-cam.zip \
#     /data/codes/apa/kaldi/g2p/data/cmu-cam-output.txt

# mfa g2p \
#     --config_path config.yml \
#     --num_jobs 12 \
#     --num_pronunciations 1 \
#     --dictionary_path /data/codes/apa/kaldi/g2p/data/cmu-cam-lexicon \
#     /data/codes/apa/kaldi/g2p/data/test-word.txt \
#     /data/codes/apa/kaldi/g2p/exp/g2p-cmu.zip \
#     /data/codes/apa/kaldi/g2p/data/cmu-output.txt

# mfa g2p \
#     --config_path config.yml \
#     --num_jobs 12 \
#     --dictionary_path /data/codes/apa/kaldi/g2p/data/cmudict-cambridge-lexicon \
#     /data/codes/apa/kaldi/g2p/lexicon/processed/cambridge-words.txt \
#     /data/codes/apa/kaldi/g2p/exp/g2p-model.zip \
#     /data/codes/apa/kaldi/g2p/data/output.txt

    # --num_pronunciations 1 \

mfa g2p \
    --config_path config.yml \
    --num_jobs 4 \
    --num_pronunciations 1 \
    --dictionary_path /data/codes/apa/kaldi/stt/resources/lexicon \
    /data/codes/apa/kaldi/stt/notebooks/word.txt \
    /data/codes/apa/kaldi/g2p/exp/g2p-cmu-prep-elsa.zip \
    /data/codes/apa/arpa