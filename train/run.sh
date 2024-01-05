#!/usr/bin/env bash
. ./path.sh

data_dir=data/train/train-data-type-12
wav_dir=/data/audio_data/prep_submission_audio/12
metadata_path=data/metadata/csv/train-data-type-12.csv

# data_dir=data/test/test-data-type-9
# wav_dir=data/wav/9
# metadata_path=data/metadata/csv/test-data-type-9.csv

# data_dir=data/train/train-data-type-9
# wav_dir=data/wav/9
# metadata_path=data/metadata/csv/train-data-type-9.csv


python run_align.py \
    --data_dir $data_dir \
    --wav_dir $wav_dir \
    --metadata_path $metadata_path

python run_gop.py \
    --data_dir $data_dir \