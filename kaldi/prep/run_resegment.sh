#!/usr/bin/env bash

ivector_extractor=exp/nnet3_cleaned/extractor
data=data/train_10k
lang=data/lang
srcdir=exp/tri1_ali_10k
dir=exp/tri1_debug

bash steps/cleanup/find_bad_utts.sh \
    $data $lang $srcdir $dir