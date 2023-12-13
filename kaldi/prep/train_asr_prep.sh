#!/usr/bin/env bash

cd ..
stage=11
. ./cmd.sh
. ./path.sh
. parse_options.sh

# if [ $stage -le 1 ]; then
  # Make some small data subsets for early system-build stages.  Note, there are 29k
  # utterances in the train_clean_100 directory which has 100 hours of data.
  # For the monophone stages we select the shortest utterances, which should make it
  # easier to align the data from a flat start.

    # utils/subset_data_dir.sh data/merged_train 10000 data/train_10k
    # utils/subset_data_dir.sh data/merged_train 20000 data/train_20k
    # utils/subset_data_dir.sh data/merged_train 30000 data/train_30k
# fi

if [ $stage -le 2 ]; then
  # train a monophone system
  steps/train_mono.sh --boost-silence 1.25 --nj 20 --cmd "$train_cmd" \
                      data/train_10k data/lang_nosp exp/mono
fi

if [ $stage -le 3 ]; then
  steps/align_si.sh --boost-silence 1.25 --nj 10 --cmd "$train_cmd" \
                    data/train_20k data/lang_nosp exp/mono exp/mono_ali_20k

  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
                        2000 10000 data/train_20k data/lang_nosp exp/mono_ali_20k exp/tri1
fi

if [ $stage -le 4 ]; then
  steps/align_si.sh --nj 10 --cmd "$train_cmd" \
                    data/train_30k data/lang_nosp exp/tri1 exp/tri1_ali_30k


  # train an LDA+MLLT system.
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
                          --splice-opts "--left-context=3 --right-context=3" 2500 15000 \
                          data/train_30k data/lang_nosp exp/tri1_ali_30k exp/tri2b
fi

if [ $stage -le 5 ]; then
  # Align a 10k utts subset using the tri2b model
  steps/align_si.sh  --nj 10 --cmd "$train_cmd" --use-graphs true \
                     data/train_30k data/lang_nosp exp/tri2b exp/tri2b_ali_30k

  # Train tri3b, which is LDA+MLLT+SAT on 10k utts
  steps/train_sat.sh --cmd "$train_cmd" 2500 15000 \
                     data/train_30k data/lang_nosp exp/tri2b_ali_30k exp/tri3b

fi

if [ $stage -le 6 ]; then
  # align the entire train_clean_100 subset using the tri3b model
  steps/align_fmllr.sh --nj 20 --cmd "$train_cmd" \
     data/merged_train_100 data/lang_nosp \
    exp/tri3b exp/tri3b_ali_merged_train_100

  # train another LDA+MLLT+SAT system on the entire 100 hour subset
  steps/train_sat.sh  --cmd "$train_cmd" 4200 40000 \
                      data/merged_train_100 data/lang_nosp \
                      exp/tri3b_ali_merged_train_100 exp/tri4b
fi

if [ $stage -le 7 ]; then
  steps/get_prons.sh --cmd "$train_cmd" \
                     data/merged_train_100 data/lang_nosp exp/tri4b
  utils/dict_dir_add_pronprobs.sh --max-normalize true \
                                  data/local/dict_nosp \
                                  exp/tri4b/pron_counts_nowb.txt exp/tri4b/sil_counts_nowb.txt \
                                  exp/tri4b/pron_bigram_counts_nowb.txt data/local/dict

  utils/prepare_lang.sh data/local/dict \
                        "<UNK>" data/local/lang_tmp data/lang
  local/format_lms.sh --src-dir data/lang data/local/lm

  utils/build_const_arpa_lm.sh \
    data/local/lm/lm_tglarge.arpa.gz data/lang data/lang_test_tglarge
  utils/build_const_arpa_lm.sh \
    data/local/lm/lm_fglarge.arpa.gz data/lang data/lang_test_fglarge
fi

if [ $stage -le 8 ]; then
  steps/align_fmllr.sh --nj 40 --cmd "$train_cmd" \
                       data/merged_train_460 data/lang exp/tri4b exp/tri4b_ali_merged_train_460

  # create a larger SAT model, trained on the 460 hours of data.
  steps/train_sat.sh  --cmd "$train_cmd" 5000 100000 \
                      data/merged_train_460 data/lang exp/tri4b_ali_merged_train_460 exp/tri5b
fi


if [ $stage -le 9 ]; then
  steps/align_fmllr.sh --nj 40 --cmd "$train_cmd" \
                       data/merged_train_960 data/lang exp/tri5b exp/tri5b_merged_train_960

  steps/train_quick.sh --cmd "$train_cmd" \
                       7000 150000 data/merged_train_960 data/lang exp/tri5b_merged_train_960 exp/tri6b

  utils/mkgraph.sh data/lang_test_tgsmall \
                   exp/tri6b exp/tri6b/graph_tgsmall

  for test in test_deepgram test_elsa test_clean; do
      steps/decode_fmllr.sh --nj 20 --cmd "$decode_cmd" \
                            exp/tri6b/graph_tgsmall data/$test exp/tri6b/decode_tgsmall_$test
      steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_{tgsmall,tgmed} \
                         data/$test exp/tri6b/decode_{tgsmall,tgmed}_$test
      steps/lmrescore_const_arpa.sh \
        --cmd "$decode_cmd" data/lang_test_{tgsmall,tglarge} \
        data/$test exp/tri6b/decode_{tgsmall,tglarge}_$test
      steps/lmrescore_const_arpa.sh \
        --cmd "$decode_cmd" data/lang_test_{tgsmall,fglarge} \
        data/$test exp/tri6b/decode_{tgsmall,fglarge}_$test
  done
fi

if [ $stage -le 10 ]; then
    local/run_cleanup_segmentation.sh
fi

if [ $stage -le 11 ]; then
    local/chain/run_tdnn.sh
fi