#!/usr/bin/env bash


# Set this to somewhere where you want to put your data, or where
# someone else has already put it.  You'll want to change this
# if you're not on the CLSP grid.
data=data

# base url for downloads.
data_url=www.openslr.org/resources/12
lm_url=www.openslr.org/resources/11
mfccdir=mfcc
stage=1

. ./cmd.sh
. ./path.sh
. parse_options.sh

# you might not want to do this for interactive shells.
set -e

# if [ $stage -le 1 ]; then
#   # download the data.  Note: we're using the 100 hour setup for
#   # now; later in the script we'll download more and use it to train neural
#   # nets.
#   for part in test-clean; do
#     local/download_and_untar.sh $data $data_url $part
#   done
#
#   local/download_lm.sh $lm_url data/local/lm
# fi

# if [ $stage -le 2 ]; then
#   # format the data as Kaldi data directories
#   for part in test-clean; do
#     # use underscore-separated names in data directories.
#     local/data_prep.sh $data/LibriSpeech/$part data/$(echo $part | sed s/-/_/g)
#   done
# fi


# if [ $stage -le 3 ]; then
#   # when the "--stage 3" option is used below we skip the G2P steps, and use the
#   # lexicon we have already downloaded from openslr.org/11/
# #   local/prepare_dict.sh --stage 3 --nj 30 --cmd "$train_cmd" \
# #    data/local/lm data/local/lm data/local/dict_nosp

# #   utils/prepare_lang.sh data/local/dict_nosp \
# #    "<UNK>" data/local/lang_tmp_nosp data/lang_nosp

#   local/format_lms.sh --src-dir data/lang_nosp data/local/lm
# fi

# if [ $stage -le 4 ]; then
#   # Create ConstArpaLm format language model for full 3-gram and 4-gram LMs
#   utils/build_const_arpa_lm.sh data/local/lm/lm_tglarge.arpa.gz \
#     data/lang_nosp data/lang_nosp_test_tglarge
#   utils/build_const_arpa_lm.sh data/local/lm/lm_fglarge.arpa.gz \
#     data/lang_nosp data/lang_nosp_test_fglarge
# fi


if [ $stage -le 6 ]; then
  for part in test_clean; do
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 40 data/$part exp/make_mfcc/$part $mfccdir
    steps/compute_cmvn_stats.sh data/$part exp/make_mfcc/$part $mfccdir
  done
fi

if [ $stage -le 7 ]; then
  # Make some small data subsets for early system-build stages.  Note, there are 29k
  # utterances in the train_clean_100 directory which has 100 hours of data.
  # For the monophone stages we select the shortest utterances, which should make it
  # easier to align the data from a flat start.

  utils/subset_data_dir.sh --shortest data/test_clean 1000 data/train_2kshort
  utils/subset_data_dir.sh data/test_clean 2000 data/train_5k
  utils/subset_data_dir.sh data/test_clean 2000 data/train_10k
fi

if [ $stage -le 8 ]; then
  # train a monophone system
  steps/train_mono.sh --boost-silence 1.25 --nj 20 --cmd "$train_cmd" \
                      data/train_2kshort data/lang_nosp exp/mono
fi

if [ $stage -le 9 ]; then
  steps/align_si.sh --boost-silence 1.25 --nj 10 --cmd "$train_cmd" \
                    data/train_5k data/lang_nosp exp/mono exp/mono_ali_5k

  # train a first delta + delta-delta triphone system on a subset of 5000 utterances
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
                        2000 10000 data/train_5k data/lang_nosp exp/mono_ali_5k exp/tri1
fi

if [ $stage -le 10 ]; then
  steps/align_si.sh --nj 10 --cmd "$train_cmd" \
                    data/train_10k data/lang_nosp exp/tri1 exp/tri1_ali_10k


  # train an LDA+MLLT system.
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
                          --splice-opts "--left-context=3 --right-context=3" 2500 15000 \
                          data/train_10k data/lang_nosp exp/tri1_ali_10k exp/tri2b
fi

if [ $stage -le 11 ]; then
  # Align a 10k utts subset using the tri2b model
  steps/align_si.sh  --nj 10 --cmd "$train_cmd" --use-graphs true \
                     data/train_10k data/lang_nosp exp/tri2b exp/tri2b_ali_10k

  # Train tri3b, which is LDA+MLLT+SAT on 10k utts
  steps/train_sat.sh --cmd "$train_cmd" 2500 15000 \
                     data/train_10k data/lang_nosp exp/tri2b_ali_10k exp/tri3b

fi
