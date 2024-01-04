import json
import librosa
import pandas as pd
import torchaudio
from pandarallel import pandarallel
from multiprocessing.pool import Pool
from tqdm import tqdm
import requests
import os

in_path = "/data/codes/apa/kaldi/stt/data/stt-data/jsonl/info_question_type-10_01082022_18092023.jsonl"
out_path = "/data/codes/apa/kaldi/stt/data/stt-data/info_question_type-10_01082022_18092023.jsonl"

with open(in_path, "r", encoding="utf-8") as f:
    lines = f.readlines()
    lines = [json.loads(line.strip()) for line in lines]

def call_api(audio_path):
    headers = {
        'authorization': 'PREP a0967870c50b53af66ca698f7df459e9'
    }

    files = [
        ('wavefile', ("audio", open(audio_path, 'rb'), 'audio/wav'))
    ]
    response = requests.post(
        "http://192.168.100.40:6869/api/stt/en?return_confidence=false", 
        data={}, headers=headers, files=files)
    
    return response.json()["transcript"]


def infer_stt(inputs):
    outputs = []

    path = f"/data/codes/apa/kaldi/stt/logs-type-10/{os.getpid()}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for sent in tqdm(inputs, desc=f'pid={os.getpid()}'):
            try:
                elsa = sent["text"]
                id = sent["id"]
                total_time = sent["total_time"]
                audio_path = sent["audio_path"]

                prep = call_api(audio_path)
                print(prep)

                output = {
                    "id": id,
                    "total_time": total_time,
                    "elsa": elsa,
                    "prep": prep,
                    "audio_path": audio_path
                }
    
                json_obj = json.dumps(output, ensure_ascii=False)
                f.write(f'{json_obj}\n')

                outputs.append(output)
            except:
                continue

    return outputs

num_process = 16

params = []
step = int(len(lines)/num_process) + 1
for i in range(num_process):
    params.append(lines[i*step:(i+1)*step])

os.system("rm -r /data/codes/apa/kaldi/stt/logs-type-10/*")
with Pool(processes=num_process) as pool:
    outputs = pool.map(infer_stt, params)

with open(out_path, "w", encoding="utf-8") as f:
    for out_process in outputs:
        for out in out_process:
            json_obj = json.dumps(out, ensure_ascii=False)
            f.write(f'{out}\n')

    print(f'saved data to: {out_path}')