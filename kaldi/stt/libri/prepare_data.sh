#!/usr/bin/env bash

cd ..
stage=1

. ./cmd.sh
. ./path.sh
. parse_options.sh

data_dir=data/
data_url=www.openslr.org/resources/12

if [ $stage -le 1 ]; then
    for part in dev-clean; do
        local/download_and_untar.sh $data_dir $data_url $part
fi