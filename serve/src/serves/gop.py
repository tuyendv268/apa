from ray import serve

import torch
from src.cores import (
    GOP
)

import torch

@serve.deployment(
    # num_replicas=2, 
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
        "num_cpus": 0.1, "num_gpus": 0.15
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
