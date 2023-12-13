import os
from src.align import (
    run_extract_feature_in_parallel,
    run_align_in_parallel,
    load_config,
    load_data,
    split_data,
)

if __name__ == "__main__":
    config_dict = load_config("configs/general.yaml")

    num_processes = 5
    n_split = config_dict["n-split"]
    conf_path = config_dict["conf-path"]

    data_dir = "data/train/train-12-v2"
    wav_dir = "/data/audio_data/prep_submission_audio/12/"
    metadata_path = "prep_data/csv/train-data-type-12-v2.csv"

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