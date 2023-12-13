from starlette.requests import Request
from ray.serve.handle import DeploymentHandle

from src.models.wavlm_model import WavLM, WavLMConfig
from src.models.score_model import PrepModel

import ray
from ray import serve
import asyncio

import soundfile as sf
import numpy as np
import json
import os
import uuid
import torch
import re
from src.cores import (
    Aligner,
    GOP
)
from src.utils.kaldi import (
    load_config
)

import pandas as pd
import requests
import json
import io
import re

from src.utils.arpa import arpa_to_ipa
import numpy as np
import requests
import librosa
import json
import ray
import soundfile as sf
import pandas as pd
import torch

def normalize(text):
    text = re.sub(r'[\!@#$%^&*\(\)\\\.\"\,\?\;\:\+\-\_\/\|~`]', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.upper().strip()
    return text

def get_word_ids(alignments):
    word_ids, curr_word_id = [], -1
    for index in alignments.index:
        phone = alignments["phone"][index]

        if phone.endswith("_B") or phone.endswith("_S"):
            curr_word_id += 1
        
        word_ids.append(curr_word_id)
    return word_ids

def parse_alignments(alignments):
    align_df = pd.DataFrame(
        alignments, columns=["phone", "start", "duration"])
    
    align_df = align_df[align_df.phone != "SIL"].reset_index()
    align_df = align_df[["phone", "start", "duration"]]
    align_df["end"] = align_df["start"] + align_df["duration"]

    align_df["end"] = align_df["end"] * 0.01
    align_df["start"] = align_df["start"] * 0.01
    align_df["duration"] = align_df["duration"] * 0.01

    align_df["word_ids"] = get_word_ids(align_df.copy())
    align_df["phone"] = align_df.phone.apply(lambda x: x.split("_")[0])

    align_df["ipa"] = align_df.phone.apply(arpa_to_ipa)

    return align_df

def get_word_scores(indices, scores):
    indices = torch.tensor(indices)
    scores = torch.tensor(scores)
    
    indices = torch.nn.functional.one_hot(
        indices.long(), num_classes=int(indices.max().item())+1)
    indices = indices / indices.sum(0, keepdim=True)
    
    scores = torch.matmul(
        indices.transpose(0, 1), scores.float())

    return scores.tolist()

def postprocess(outputs):
    word_scores = get_word_scores(
        indices=outputs["word_ids"],
        scores=outputs["word_score"]
    )

    outputs["word_score"] = outputs.word_ids.apply(lambda x: word_scores[x])

    outputs["word_score"] = outputs.word_score.apply(lambda x: round(x*50, 0))
    outputs["phone_score"] = outputs.phone_score.apply(lambda x: round(x*50, 0))
    outputs["utterance_score"] = outputs.utterance_score.apply(lambda x: round(x*50, 0))

    return outputs

def convert_dataframe_to_json(metadata, transcript):
    sentence = {
        "text": transcript,
        "score": round(metadata["utterance_score"].mean(), 2),
        "duration": 0,
        "ipa": "",
        "words": []
    }

    words = transcript.split()
    for word_id, word in enumerate(words):
        phonemes = metadata[metadata.word_ids == word_id]
        tmp_word = {
            "text": word,
            "score": metadata[metadata.word_ids == word_id].word_score.mean(),
            "arpabet": "",
            "ipa": "",
            "phonemes": [],
            "start_index": 0,
            "end_index": 0,
            "start_time": 0,
            "end_time": 0,
        }
        for index in phonemes.index:
            phone = {
                "arpabet": phonemes["phone"][index],
                "score": round(phonemes["phone_score"][index], 2),
                "ipa": phonemes["ipa"][index],
                "start_index": 0,
                "end_index": 0,
                "start_time": phonemes["start"][index],
                "end_time": phonemes["end"][index],
                "sound_most_like": phonemes["phone"][index],
            }

            tmp_word["phonemes"].append(phone)

        tmp_word["start_time"] = tmp_word["phonemes"][0]["start_time"]
        tmp_word["end_time"] = tmp_word["phonemes"][-1]["end_time"]

        sentence["words"].append(tmp_word)

    return {
        "version": "v0.1.0",
        "utterance": [sentence, ]
    }

def parse_response(alignments, scores, transcript):
    scores = pd.DataFrame(scores)
    alignments = parse_alignments(alignments)

    outputs = pd.concat([alignments, scores], axis=1)
    outputs = postprocess(outputs)

    response = convert_dataframe_to_json(
        metadata=outputs, 
        transcript=transcript
    )
    return response

@serve.deployment(
    num_replicas=2,
    max_concurrent_queries=128,
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

    
    async def __call__(self, http_request: Request):
        sample = await http_request.json() 

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

        return response

def pad_1d(inputs, max_length=None, pad_value=0.0):
    if max_length is None:
        lengths = [len(sample) for sample in inputs]
        max_length = max(lengths)

    for i in range(len(inputs)):
        if inputs[i].shape[0] < max_length:
            inputs[i] = torch.cat(
                (
                    inputs[i], 
                    pad_value * torch.ones(max_length-inputs[i].shape[0])),
                dim=0
            )
        else:
            inputs[i] = inputs[i][0:max_length]
    inputs = torch.stack(inputs, dim=0)
    return inputs

def pad_2d(inputs, max_length=None, pad_value=0.0):
    if max_length is None:
        lengths = [len(sample) for sample in inputs]
        max_length = max(lengths)

    for i in range(len(inputs)):
        if inputs[i].shape[0] < max_length:
            inputs[i] = torch.cat(
                (
                    inputs[i], 
                    pad_value * torch.ones((max_length-inputs[i].shape[0], inputs[i].shape[1]))),
                dim=0
            )
        else:
            inputs[i] = inputs[i][0:max_length]
    inputs = torch.stack(inputs, dim=0)
    return inputs

@serve.deployment(
    num_replicas=1, 
    max_concurrent_queries=64,
    ray_actor_options={
        "num_cpus": 0.1, "num_gpus": 0.2
        }
    )
class Scoring_Model:
    def __init__(self, configs, gopt_ckpt_path):        
        self.model = self.init_model(
            configs=configs, 
            gopt_ckpt_path=gopt_ckpt_path
        )
        phone_dict_path = "resources/phone_dict.json"
        relative_to_id_path = "resources/relative2id.json"

        self.phone_dict = json.load(open(phone_dict_path, "r"))
        self.rel_to_id = json.load(open(relative_to_id_path, "r"))

        self.model.eval().cuda()

    def init_model(self, configs, gopt_ckpt_path):
        prep_model = PrepModel(
            embed_dim=configs['embed-dim'], 
            num_heads=configs['num-heads'], 
            depth=configs['depth'], 
            input_dim=configs['input-dim'], 
            max_length=configs['max-length'], 
            num_phone=configs['num-phone']
        )

        gopt_state_dict = torch.load(gopt_ckpt_path, map_location="cpu")
        prep_model.load_state_dict(gopt_state_dict)
        print(f"###load state dict from {gopt_ckpt_path}")
        return prep_model
    
    def extract_feature(self, alignments, features):
        index = 0
        phonemes = []
        indices = -1 * torch.ones(alignments[-1][1] + alignments[-1][2])
        for phoneme, start_frame, duration in alignments:
            end_frame = start_frame + duration
            indices[start_frame:end_frame] = index
            phonemes.append(phoneme)
            index += 1

        indices[indices==-1] = indices.max() + 1

        indices = torch.nn.functional.one_hot(indices.long(), num_classes=int(indices.max().item())+1).cuda()
        indices = indices / indices.sum(0, keepdim=True)
        
        if features.shape[0] != indices.shape[0]:
            features = features[0:indices.shape[0]]

        features = torch.matmul(indices.transpose(0, 1), features).cpu()

        return features, phonemes

    def merge_ssl_feature_in_frame_level_to_phone_level(self, features, alignments):
        indices = torch.arange(features.shape[0]).unsqueeze(-1).cuda()
        expanded_indices = indices.expand((-1, 2)).flatten()
        
        features = features[expanded_indices]
        features, phonemes = self.extract_feature(alignments, features)

        return features[0:len(phonemes)]

    def preprocess_one_sample(self, gops, alignments, wavlm_features):
        processed_alignments = []
        phone_ids, durations, relative_positions = [], [], []
        for phone, start_frame, duration_in_frame in alignments:
            if phone == "SIL":
                continue

            phone, relative_position = phone.split("_")[0], phone.split("_")[-1]

            pure_phone = re.sub("\d", "", phone)
            phone_id = self.phone_dict[pure_phone]
            relative_position = self.rel_to_id[relative_position]
            duration = round(duration_in_frame * 0.02, 4)

            durations.append(duration)
            phone_ids.append(phone_id)
            relative_positions.append(relative_position)
            processed_alignments.append((phone, start_frame, duration_in_frame))
        
        wavlm_features = torch.tensor(wavlm_features).cuda()
        ssl_features = self.merge_ssl_feature_in_frame_level_to_phone_level(
            features=wavlm_features,
            alignments=processed_alignments
        )

        ssl_features = ssl_features
        phone_ids = torch.tensor(phone_ids)
        relative_positions = torch.tensor(relative_positions)
        durations = torch.tensor(durations)
        gops = torch.tensor(gops)

        return phone_ids, gops, ssl_features, relative_positions, durations

    def preprocess(self, batch):
        alignments, wavlm_features, gops = [], [], []
        inputs = {
            "phone_ids": [],
            "gops":[], 
            "ssl_features":[],
            "relative_positions": [],
            "durations": []
        }

        for sample in batch:
            alignment = sample["alignment"]
            feature = sample["wavlm_feature"]
            gop = sample["gop"]

            phone_ids, gops, ssl_features, relative_positions, durations = self.preprocess_one_sample(
                gops=gop,
                alignments=alignment, 
                wavlm_features=feature)

            inputs["phone_ids"].append(phone_ids)
            inputs["gops"].append(gops)
            inputs["relative_positions"].append(relative_positions)
            inputs["ssl_features"].append(ssl_features)
            inputs["durations"].append(durations)

        inputs["phone_ids"] = pad_1d(inputs["phone_ids"], max_length=None, pad_value=0)
        inputs["durations"] = pad_1d(inputs["durations"], max_length=None, pad_value=0)
        inputs["relative_positions"] = pad_1d(inputs["relative_positions"], max_length=None, pad_value=0)
        inputs["gops"] = pad_2d(inputs["gops"], max_length=None, pad_value=0)
        inputs["ssl_features"] = pad_2d(inputs["ssl_features"], max_length=None, pad_value=0)

        return inputs

    def run_scoring(self, wavlm_features, gop_features, rel_position, phone_ids, durations):
        features = torch.concat(
            [
                gop_features, 
                durations, 
                wavlm_features
            ], dim=-1)

        utterance_scores, phone_scores, word_scores = self.model(
            x=features, phn=phone_ids, rel_pos=rel_position)
        
        utterance_scores = utterance_scores.squeeze(-1)
        phone_scores = phone_scores.squeeze(-1)
        word_scores = word_scores.squeeze(-1)

        return utterance_scores, phone_scores, word_scores

    @serve.batch(max_batch_size=64, batch_wait_timeout_s=0.1)
    async def run(self, batch):
        batch = self.preprocess(batch=batch)

        print('batch["phone_ids"]: ', batch["phone_ids"].shape)
        print('batch["gops"]: ', batch["gops"].shape)
        print('batch["relative_positions"]: ', batch["relative_positions"].shape)
        print('batch["ssl_features"]: ', batch["ssl_features"].shape)
        print('batch["durations"]: ', batch["durations"].shape)

        wavlm_features = batch["ssl_features"].cuda()
        gop_features = batch["gops"].cuda()
        relative_positions = batch["relative_positions"].cuda()
        durations = batch["durations"].cuda().unsqueeze(-1)
        phone_ids = batch["phone_ids"].cuda()

        utterance_scores, phone_scores, word_scores = self.run_scoring(
            wavlm_features=wavlm_features, 
            gop_features=gop_features, 
            rel_position=relative_positions,
            durations=durations,
            phone_ids=phone_ids)
        
        utterance_scores = utterance_scores.tolist()
        phone_scores = phone_scores.tolist()
        word_scores = word_scores.tolist()

        outputs = self.postprocess(
            utterance_scores=utterance_scores, 
            phone_scores=phone_scores, 
            word_scores=word_scores
        )

        return outputs

    def postprocess(self, utterance_scores, phone_scores, word_scores):
        outputs = []

        for index in range(len(utterance_scores)):
            utterance_score = utterance_scores[index]
            phone_score = phone_scores[index]
            word_score = word_scores[index]

            sample = {
                "utterance_score": utterance_score,
                "phone_score": phone_score,
                "word_score": word_score,
            }

            outputs.append(sample)
        
        return outputs

def pad_1d(inputs, max_length=None, pad_value=0.0):
    if max_length is None:
        lengths = [len(sample) for sample in inputs]
        max_length = max(lengths)

    for i in range(len(inputs)):
        if inputs[i].shape[0] < max_length:
            inputs[i] = torch.cat(
                (
                    inputs[i], 
                    pad_value * torch.ones(max_length-inputs[i].shape[0])),
                dim=0
            )
        else:
            inputs[i] = inputs[i][0:max_length]
    inputs = torch.stack(inputs, dim=0)
    return inputs
    
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


@serve.deployment(
    num_replicas=2, 
    max_concurrent_queries=64,
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
        "num_cpus": 0.2, "num_gpus": 0.25
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
        for _id, _sample in enumerate(batch):
            _id = f'audio-{_id}'
            _transcript = _sample["transcript"]
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
    
@serve.deployment(
    num_replicas=1, 
    max_concurrent_queries=64,
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
        "num_cpus": 0.1, "num_gpus": 0.25
        }
    )
class GOP_Recipe:
    def __init__(self, configs):
        self.model = GOP(configs=configs)

    def preprocess(self, batch):
        """
        sample = {
            "id": _id,
            "alignment": _alignment,
            "scores_phone_pure": _scores_phone_pure,
            "length": _length
        }
        """
        ids, alignments, scores_phone_pures, lengths = [], [], [], []

        for sample in batch:
            _id = sample["id"]
            _alignment = sample["alignment"]
            _scores_phone_pure = sample["scores_phone_pure"]
            _length = sample["length"]

            ids.append(_id)
            alignments.append(_alignment)
            scores_phone_pures.append(_scores_phone_pure)
            lengths.append(_length)

        scores_phone_pures = torch.tensor(scores_phone_pures).cuda()
        return ids, alignments, scores_phone_pures, lengths


    @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.1)
    async def run(self, batch):
        ids, alignments, scores_phone_pures, lengths = self.preprocess(batch=batch)
        
        ids, gops, phonemes = self.model.run_batch(
            ids=ids,
            alignments=alignments, 
            scores_phone_pures=scores_phone_pures, 
            lengths=lengths
        )

        outputs = self.postprocess(
            ids=ids, 
            gops=gops, 
            phonemes=phonemes
        )

        return outputs

    def postprocess(self, ids, gops, phonemes):
        outputs = []

        assert len(ids) == len(gops)
        assert len(ids) == len(phonemes)
        
        for index in range(len(ids)):
            _id = ids[index]
            _gop = gops[index]
            # _phoneme = phonemes[index]
            
            sample = {
                "id": _id,
                "gop": _gop,
                # "phoneme": _phoneme
            }
            outputs.append(sample)

        return outputs

configs = load_config("configs/general.yaml")
align_app = Forced_Aligner.bind(configs)
gop_app = GOP_Recipe.bind(configs)

configs = load_config("configs/model.yaml")
gopt_ckpt_path = "exps/ckpts/ckpts-eph=21-mse=0.1411/model.pt"
score_app = Scoring_Model.bind(configs, gopt_ckpt_path)

pretrained_path = "exps/ckpts/wavlm-base+.pt"
wavlm_app = WavLM_Model.bind(pretrained_path)

app = Main.bind(align_app, gop_app, wavlm_app, score_app)
