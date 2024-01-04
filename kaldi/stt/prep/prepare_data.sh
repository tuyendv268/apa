#!/usr/bin/env bash

cd ..
stage=1

. ./cmd.sh
. ./path.sh
. parse_options.sh

rm -r data/{train_elsa,train_deepgram,train_clean_100,train_clean_360,train_other_500,test_deepgram,test_elsa,test_clean}
cp -r data/prep/{train_elsa,train_deepgram,train_clean_100,train_clean_360,train_other_500,test_deepgram,test_elsa,test_clean} data/

if [ $stage -le 1 ]; then
    for part in train_elsa train_deepgram train_clean_100 train_clean_360 train_other_500 test_deepgram test_elsa test_clean; do
        steps/make_mfcc.sh --cmd "$train_cmd" --nj 40 data/$part exp/make_mfcc/$part $mfccdir
        utils/fix_data_dir.sh data/$part
        steps/compute_cmvn_stats.sh data/$part exp/make_mfcc/$part $mfccdir
        utils/fix_data_dir.sh data/$part
    done
fi

if [ $stage -le 2 ]; then
    utils/combine_data.sh \
        data/prep_dataset data/train_elsa data/train_deepgram

    utils/combine_data.sh \
        data/merged_train_100 data/train_clean_100 data/prep_dataset

    utils/combine_data.sh \
        data/merged_train_460 data/train_clean_360 data/merged_train_100

    utils/combine_data.sh \
        data/merged_train_960 data/train_other_500 data/merged_train_460
fi

if [ $stage -le 3 ]; then
    utils/subset_data_dir.sh data/merged_train_460 10000 data/train_10k
    utils/subset_data_dir.sh data/merged_train_460 20000 data/train_20k
    utils/subset_data_dir.sh data/merged_train_460 30000 data/train_30k
fi
