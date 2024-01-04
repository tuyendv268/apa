from src.models.score_model import PrepModel

from ray import serve

import json
import torch
import re

from src.utils.serve import (
    pad_1d,
    pad_2d,
)

import json
import re

import json
import torch

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
        phone_dict_path = "exp/dicts/phone_dict.json"
        relative_to_id_path = "exp/dicts/relative2id.json"

        self.phone_dict = json.load(open(phone_dict_path, "r"))
        self.rel_to_id = json.load(open(relative_to_id_path, "r"))

        self.model.eval()

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

        indices = torch.nn.functional.one_hot(indices.long(), num_classes=int(indices.max().item())+1)
        indices = indices / indices.sum(0, keepdim=True)
        
        if features.shape[0] != indices.shape[0]:
            features = features[0:indices.shape[0]]

        features = torch.matmul(indices.transpose(0, 1), features).cpu()

        return features, phonemes

    def merge_ssl_feature_in_frame_level_to_phone_level(self, features, alignments):
        indices = torch.arange(features.shape[0]).unsqueeze(-1)
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
        
        wavlm_features = torch.tensor(wavlm_features)
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

        wavlm_features = batch["ssl_features"]
        gop_features = batch["gops"]
        relative_positions = batch["relative_positions"]
        durations = batch["durations"].unsqueeze(-1)
        phone_ids = batch["phone_ids"]

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
