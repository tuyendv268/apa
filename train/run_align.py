import os
from src.align import (
    run_extract_feature_in_parallel,
    run_align_in_parallel,
    load_config,
    load_data,
    split_data,
)
import argparse

if __name__ == "__main__":
    config_dict = load_config("configs/general.yaml")

    num_processes = 5
    n_split = config_dict["n-split"]
    conf_path = config_dict["conf-path"]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--data_dir', type=str, default="data/train/info_qt_10_trainset-aug")
    
    parser.add_argument(
        '-w', '--wav_dir', type=str, default="/data/audio_data/prep_submission_audio/10")
    
    parser.add_argument(
        '-m', '--metadata_path', type=str, default="prep_data/csv/info_qt_10_trainset-aug.csv")

    args = parser.parse_args()

    data_dir = args.data_dir
    wav_dir = args.wav_dir
    metadata_path = args.metadata_path

    data = load_data(
        metadata_path=metadata_path,
        wav_dir=wav_dir
    )
    print("###num split: ", n_split)

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    split_dirs = split_data(
        data=data, n_split=n_split, out_dir=data_dir
    )

    run_extract_feature_in_parallel(
        conf_path=conf_path,
        n_process=n_split,
        split_dirs=split_dirs
    )

    run_align_in_parallel(
        split_dirs=split_dirs, 
        config_dict=config_dict, 
        n_processes=num_processes
    )