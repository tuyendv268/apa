
import torch
import re

import pandas as pd
import re

from src.utils.arpa import arpa_to_ipa
import pandas as pd
import torch

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