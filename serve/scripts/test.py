from time import time
import torch
import os

from src.cores import (
    Aligner,
    GOP
)
from serving.src.utils.kaldi import (
    load_config
)

if __name__ == "__main__":
    configs = load_config("configs/config_prep.yaml")

    wav_path = "/data/codes/serving/wav/She's paying for some jewelry at a checkout.wav"
    transcript = "She's paying for some jewelry at a check out".upper()

    aligner = Aligner(configs=configs)
    gop_recipe = GOP(configs=configs)

    start_time = time()
    ids, alignments, scores_phone_pures, lengths = aligner.run_batch(
        wav_paths=[wav_path, ], 
        transcripts=[transcript, ], 
        ids=["8888"])
    
    scores_phone_pures = torch.tensor(scores_phone_pures).cuda()
    ids, features, phonemes = gop_recipe.run_batch(
        ids=ids, 
        alignments=alignments, 
        scores_phone_pures=scores_phone_pures, 
        lengths=lengths
    )
    end_time = time()
    print(phonemes)
    print(len(phonemes[0]))
    print(len(features[0]))
    for i in features[0][0:5]:
        print(i[0:10])

    # print("duration: ", end_time-start_time)
    # print(phonemes)
