#!/usr/bin/env bash

cd ..
lm_url=www.openslr.org/resources/11
stage=1

. ./cmd.sh
. ./path.sh
. parse_options.sh

if [ $stage -le 1 ]; then
    local/download_lm.sh $lm_url data/local/lm
fi

if [ $stage -le 2 ]; then
    local/prepare_dict.sh \
        --stage 3 \
        --nj 30 --cmd "$train_cmd" \
        data/local/lm data/local/lm data/local/dict_nosp

    utils/prepare_lang.sh data/local/dict_nosp \
        "<UNK>" data/local/lang_tmp_nosp data/lang_nosp

    local/format_lms.sh --src-dir data/lang_nosp data/local/lm
fi

if [ $stage -le 3 ]; then
    utils/build_const_arpa_lm.sh data/local/lm/lm_tglarge.arpa.gz \
        data/lang_nosp data/lang_nosp_test_tglarge
    utils/build_const_arpa_lm.sh data/local/lm/lm_fglarge.arpa.gz \
        data/lang_nosp data/lang_nosp_test_fglarge
fi
