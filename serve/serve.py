from starlette.requests import Request
from ray.serve.handle import DeploymentHandle

from ray import serve
import asyncio

from src.utils.serve import (
    parse_response
)
from src.serves.gop import GOP_Recipe
from src.serves.align import Forced_Aligner
from src.serves.scoring import Scoring_Model
from src.serves.wavlm import WavLM_Model

from src.utils.kaldi import (
    load_config
)

from time import gmtime, strftime
import soundfile as sf
import numpy as np
import librosa
import os

@serve.deployment(
    # num_replicas=2,
    # max_concurrent_queries=128,
    num_replicas=1,
    max_concurrent_queries=16,
    # health_check_period_s=10,
    # health_check_timeout_s=30,
    # graceful_shutdown_timeout_s=20,
    # graceful_shutdown_wait_loop_s=2,
    route_prefix="/scoring",
    ray_actor_options={
        "num_cpus": 0.05,
        }
)
class Main:
    def __init__(self, aligner, gop, wavlm, scoring):
        self.aligner: DeploymentHandle = aligner.options(
            use_new_handle_api=True,
        )
        self.gop: DeploymentHandle = gop.options(
            use_new_handle_api=True,
        )
        self.wavlm: DeploymentHandle = wavlm.options(
            use_new_handle_api=True,
        )
        self.scoring: DeploymentHandle = scoring.options(
            use_new_handle_api=True,
        )

    async def run_align_gop(self, sample):
        align_outputs = await self.aligner.run.remote(
            sample
        )

        gop_outputs = await self.gop.run.remote(
            align_outputs
        )

        assert align_outputs["id"] == gop_outputs["id"]

        alignments = align_outputs["alignment"]
        gops = gop_outputs["gop"]

        return alignments, gops

    async def run_scoring(self, gops, wavlm_features, alignments):
        inputs = {
            "gop": gops,
            "wavlm_feature": wavlm_features,
            "alignment": alignments
        }

        scores = await self.scoring.run.remote(
            inputs
        )
        
        return scores

    def log_to_file(self, sample):
        transcript = sample["transcript"]
        audio = sample["audio"]

        cur_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        log_dir = "logs"

        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        
        out_dir = f'{log_dir}/{cur_time}'

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        
        trans_path = f'{out_dir}/transcript'
        with open(trans_path, "w", encoding="utf-8") as f:
            f.write(transcript)
        
        wav_path = f'{out_dir}/audio.wav'
        audio = np.array(audio)
        audio = audio/np.abs(audio.max())
        sf.write(wav_path, audio, samplerate=16000)

        print(f'### log request to file: {out_dir}')
    
    async def __call__(self, http_request: Request):
        sample = await http_request.json() 
        self.log_to_file(sample)

        wavlm_features, (alignments, gops) = await asyncio.gather(
            self.wavlm.run.remote(sample), 
            self.run_align_gop(sample)
        )

        scores = await self.run_scoring(
            gops=gops, 
            alignments=alignments,
            wavlm_features=wavlm_features
        )

        response = parse_response(
            alignments=alignments,
            scores=scores,
            transcript=sample["transcript"]
        )
        for gop, align in zip(gops, alignments):
            print(align, gop[0:5])

        return response

configs = load_config("configs/general.yaml")
align_app = Forced_Aligner.bind(configs)
gop_app = GOP_Recipe.bind(configs)

configs = load_config("configs/model.yaml")
gopt_ckpt_path = "exp/ckpts/ckpts-eph=18-mse=0.08669999986886978/model.pt"
# gopt_ckpt_path = "../train/exp/fine-tuning/ckpts-eph=19-mse=0.2442999929189682/model.pt"
score_app = Scoring_Model.bind(configs, gopt_ckpt_path)

pretrained_path = "exp/ckpts/wavlm-base+.pt"
wavlm_app = WavLM_Model.bind(pretrained_path)

app = Main.bind(align_app, gop_app, wavlm_app, score_app)
