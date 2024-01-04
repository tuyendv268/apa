data=/data2/data/asr_dataset

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


if [ $stage -le 1 ]; then
  # download the data.  Note: we're using the 100 hour setup for
  # now; later in the script we'll download more and use it to train neural
  # nets.
  for part in test-clean train-clean-100  train-clean-360 train-other-500; do
    local/download_and_untar.sh $data $data_url $part
  done
fi

if [ $stage -le 2 ]; then
  # format the data as Kaldi data directories
  for part in test-clean train-clean-100  train-clean-360 train-other-500; do
    # use underscore-separated names in data directories.
    local/data_prep.sh $data/LibriSpeech/$part data/$(echo $part | sed s/-/_/g)
  done
fi
