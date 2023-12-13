### Pronunciation scoring
1. Build docker images:
    ```
    docker build -t prep/pykaldi-gpu:latest .
    ```

    ```
    pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
    ```
2. Run:
    ```
    sudo nvidia-docker run -it --gpus '"device=0,1"' \
        -p 9999:9999 \
        -v /data/codes/apa/:/workspace \
        -v /data/audio_data/prep_submission_audio:/data/audio_data/prep_submission_audio \
        prep/pykaldi-gpu-python3.9:latest \
        /bin/bash
    ```
3. Output format: 

    ```
        """ 
            {
                "text": "...",
                "arpabet": "...",
                "score": "...",
                "words": [
                    {
                        "text": "...",
                        "arpabet": "...",
                        "score": "...",
                        "phones": [
                            {
                                "arpabet": "...",
                                "ipa": "...",
                                "score": "..."
                            },
                            {
                                "arpabet": "...",
                                "ipa": "...",
                                "score": "..."
                            }
                        ]
                    },
                ]
            }
        """
    ```