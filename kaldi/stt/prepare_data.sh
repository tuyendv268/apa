#!/usr/bin/env bash

stage=1

. ./cmd.sh
. ./path.sh
. parse_options.sh

rm -r data/{prep_data_type_9,prep_data_type_10,train_clean_100,train_clean_360,train_other_500,test_type_10,test_type_9}
rm -r data/{merged_train_460,merged_train_100,merged_train_960,train_10k,train_20k,train_30k}
cp -r data/stt-data/kaldi/{prep_data_type_9,prep_data_type_10,train_clean_100,train_clean_360,train_other_500,test_type_10,test_type_9} data/

if [ $stage -le 1 ]; then
    for part in prep_data_type_9 prep_data_type_10 train_clean_100 train_clean_360 train_other_500 test_type_10 test_type_9; do
        steps/make_mfcc.sh --cmd "$train_cmd" --nj 40 data/$part exp/make_mfcc/$part $mfccdir
        utils/fix_data_dir.sh data/$part
        steps/compute_cmvn_stats.sh data/$part exp/make_mfcc/$part $mfccdir
        utils/fix_data_dir.sh data/$part
    done
fi

if [ $stage -le 2 ]; then
    utils/combine_data.sh \
        data/merged_train_100 data/train_clean_100 data/prep_data_type_10

    utils/combine_data.sh \
        data/merged_train_460 data/train_clean_360 data/merged_train_100

    utils/combine_data.sh \
        data/merged_train_960_tmp data/train_other_500 data/merged_train_460

    utils/combine_data.sh \
        data/merged_train_960 data/prep_data_type_9 data/merged_train_960_tmp

fi

if [ $stage -le 3 ]; then
    utils/subset_data_dir.sh data/merged_train_960 15000 data/train_10k
    utils/subset_data_dir.sh data/merged_train_960 25000 data/train_20k
    utils/subset_data_dir.sh data/merged_train_960 35000 data/train_30k
fi
