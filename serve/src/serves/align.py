from ray import serve

import soundfile as sf
import numpy as np
import os
from src.cores import (
    Aligner,
)

import numpy as np
import soundfile as sf
import re

def normalize(text):
    text = re.sub(r'[\!@#$%^&*\(\)\\\.\"\,\?\;\:\+\-\_\/\|~`]', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.upper().strip()
    return text

@serve.deployment(
    # num_replicas=2, 
    # max_concurrent_queries=64,
    num_replicas=1, 
    max_concurrent_queries=16,
    # autoscaling_config={
    #     "min_replicas": 1,
    #     "initial_replicas": 2,
    #     "max_replicas": 4,
    #     "target_num_ongoing_requests_per_replica": 10
    # },
    # health_check_period_s=10,
    # health_check_timeout_s=30,
    # graceful_shutdown_timeout_s=20,
    # graceful_shutdown_wait_loop_s=2,
    ray_actor_options={
        "num_cpus": 0.2, "num_gpus": 0.2
        }
    )
class Forced_Aligner:
    def __init__(self, configs):
        self.model = Aligner(configs=configs)
        self.configs = configs

        self.init_dir()

    def init_dir(self):
        if not os.path.exists(self.configs["data-dir"]):
            os.mkdir(self.configs["data-dir"])

        data_dir = f'{self.configs["data-dir"]}/{os.getpid()}'
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        audio_dir = f'{data_dir}/wav'
        if not os.path.exists(audio_dir):
            os.mkdir(audio_dir)
        
        self.audio_dir = audio_dir

    def preprocess(self, batch):
        # audio_dir = self.init_dir()

        audio_paths, ids, transcripts = [], [], []
        print("###Align_batch_size: ", len(batch))
            
        for _id, _sample in enumerate(batch):
            _id = f'audio-{_id}'
            _transcript = normalize(_sample["transcript"])
            _audio_path = f'{self.audio_dir}/{_id}.wav'
            
            waveform = np.array(_sample["audio"])
            sf.write(_audio_path, waveform, samplerate=16000)

            ids.append(_id)
            audio_paths.append(_audio_path)
            transcripts.append(_transcript)

        return {
            "ids": ids,
            "transcripts": transcripts,
            "audio_paths": audio_paths,
        }

    @serve.batch(max_batch_size=128, batch_wait_timeout_s=0.1)
    async def run(self, batch):
        batch = self.preprocess(batch)

        ids, alignments, scores_phone_pures, lengths = self.model.run_batch(
            wav_paths=batch["audio_paths"], 
            transcripts=batch["transcripts"],
            ids=batch["ids"]
        )
        
        outputs = self.postprocess(
            ids=ids, 
            alignments=alignments, 
            scores_phone_pures=scores_phone_pures, 
            lengths=lengths
        )

        return outputs

    def postprocess(self, ids, alignments, scores_phone_pures, lengths):
        outputs = []

        assert len(ids) == len(alignments)
        assert len(ids) == len(scores_phone_pures)
        assert len(ids) == len(lengths)

        for index in range(len(ids)):
            _id = ids[index]
            _alignment = alignments[index]
            _scores_phone_pure = scores_phone_pures[index]
            _length = lengths[index]

            sample = {
                "id": _id,
                "alignment": _alignment,
                "scores_phone_pure": _scores_phone_pure,
                "length": _length
            }

            outputs.append(sample)

        return outputs
