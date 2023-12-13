from src.models.wavlm_model import WavLM, WavLMConfig
from ray import serve
import torch

from src.utils.serve import (
    pad_1d,
    pad_2d,
)

@serve.deployment(
    num_replicas=1, 
    max_concurrent_queries=64,
    ray_actor_options={
        "num_cpus": 0.1, "num_gpus": 0.2
        }
    )
class WavLM_Model:
    def __init__(self, pretrained_path):        
        wavlm_model, wavlm_config = self.init_models(pretrained_path)

        self.wavlm_model = wavlm_model.half().eval().cuda()
        self.wavlm_config = wavlm_config

    def init_models(self, wavlm_ckpt_path=None):
        wavlm_state_dict = torch.load(wavlm_ckpt_path, map_location="cpu")

        wavlm_config = WavLMConfig(wavlm_state_dict['cfg'])
        wavlm_model = WavLM(wavlm_config)

        wavlm_model.load_state_dict(wavlm_state_dict['model'])

        return wavlm_model, wavlm_config


    def preprocess(self, batch):
        waveforms = []
        for sample in batch:
            audio = torch.tensor(sample["audio"])

            waveforms.append(audio)

        waveforms = pad_1d(
            inputs=waveforms, 
            max_length=None, 
            pad_value=0.0
        )

        return waveforms.cuda()

    @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.1)
    async def run(self, batch):
        waveforms = self.preprocess(batch)

        waveforms = waveforms.half()

        with torch.no_grad():
            features = self.wavlm_model.extract_features(waveforms)
            
        outputs = self.postprocess(
            features=features[0]
        )

        return outputs

    def postprocess(self, features):
        features = features.tolist()
        
        return features

