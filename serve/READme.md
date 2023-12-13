### Pronunciation scoring
1. Build docker images:
    ```
    docker build -t prep/pykaldi-gpu:latest .
    ```
2. Run:

    ```
    sudo nvidia-docker run -it --gpus '"device=0,1"' -p 9999:9999 \
        --shm-size=10.24gb \
        -v /data/codes/serving/:/data/codes/serving/ \
        --name ray-serve prep/pykaldi-gpu-python3.9:latest \
        /bin/bash
    ```

    ```
    sudo nvidia-docker run -it --gpus '"device=0,1"' \
        -p 9999:9999 \
        -v /data/codes/apa/:/workspace \
        prep/pykaldi-gpu-python3.9:latest \
        /bin/bash
    ```
3. Run (demo directory):

    ```
    bash run.sh
    ```

4. Output format: 

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