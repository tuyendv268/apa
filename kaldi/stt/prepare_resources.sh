#!/usr/bin/env bash

lm_url=www.openslr.org/resources/11
stage=1

. ./cmd.sh
. ./path.sh
. parse_options.sh

if [ $stage -le 1 ]; then
    local/download_lm.sh $lm_url data/local/lm
fi

if [ $stage -le 2 ]; then
    local/format_lms.sh --src-dir data/lang_nosp data/local/lm
fi

if [ $stage -le 3 ]; then
    utils/build_const_arpa_lm.sh data/local/lm/lm_tglarge.arpa.gz \
        data/lang_nosp data/lang_nosp_test_tglarge
    utils/build_const_arpa_lm.sh data/local/lm/lm_fglarge.arpa.gz \
        data/lang_nosp data/lang_nosp_test_fglarge
fi
