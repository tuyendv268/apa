
#!/usr/bin/env bash


# Set this to somewhere where you want to put your data, or where
# someone else has already put it.  You'll want to change this
# if you're not on the CLSP grid.

# base url for downloads.
mfccdir=mfcc
stage=1

#extract ivectors 
steps/online/nnet2/extract_ivectors.sh --nj 1 \
    --cmd run.pl data/test_clean_hires data/lang \
    exp/nnet3_cleaned/extractor exp/nnet3_cleaned/ivectors_test_clean_hires

#make a graph 
utils/mkgraph.sh --self-loop-scale 1.0 \
    --remove-oov data/lang exp/nnet3_cleaned/tdnn_sp \
    exp/nnet3_cleaned/tdnn_sp/graph_tgsmall

#decode using the graph (lattice file is created(lat.1.gz) ) 
steps/nnet3/decode.sh --acwt 1.0 \
    --post-decode-acwt 10.0 --nj 1 \
    --online-ivector-dir exp/nnet3_cleaned/ivector_test_clean_hires \
    exp/nnet3_cleaned/tdnn_sp/graph_tgsmall \
    data/test_clean_hires exp/nnet3_cleaned/tdnn_sp/decode_test_tgsmall

#to get the transcript
../../../src/latbin/lattice-best-path \
    ark:'gunzip -c exp/nnet3_cleaned/tdnn_sp/decode_test_tgsmall/lat.1.gz |' ark,t:| \
    utils/int2sym.pl -f 2- data/lang/words.txt > out.txt

src/latbin/lattice-1best \
    --acoustic-scale=0.1 ark:lat.1 ark:1best.lats

src/latbin/nbest-to-linear \
    ark:1best.lats ark:1best.ali 'ark,t:|int2sym.pl -f 2- words.txt > text'

src/bin/ali-to-phones --ctm-output \
    exp/nnet3_cleaned/tdnn_sp/final.mdl ark:1best.ali 1best.ctm
