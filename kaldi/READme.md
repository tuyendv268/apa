### Kaldi
1. Build docker images:
    ```
    docker build -t prep/kaldi-gpu-cu12:latest .
    ```

2. Run:
    ```
    sudo nvidia-docker run -it --gpus '"device=0,1"' \
        -v /data/codes/apa/kaldi/:/workspace \
        mmcauliffe/montreal-forced-aligner:latest \
        /bin/bash
    ```

    ```
    sudo nvidia-docker run -it --gpus '"device=0,1"' --cpus 26 \
    -v /data/codes/apa/kaldi:/data/codes/apa/kaldi \
    -v /data/audio_data/prep_submission_audio:/data/audio_data/prep_submission_audio \
    -v /data2/data/asr_dataset:/data2/data/asr_dataset \
    nvcr.io/nvidia/kaldi:22.10-py3

    ```