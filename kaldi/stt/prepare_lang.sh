#!/usr/bin/env bash

. utils/parse_options.sh || exit 1;
. ./path.sh || exit 1

utils/prepare_lang.sh \
   data/local/dict "<UNK>" \
   data/lang_nosp_tmp data/lang_nosp