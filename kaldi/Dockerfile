FROM nvidia/cuda:12.0.1-cudnn8-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt install software-properties-common -y
RUN apt update
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install python3.8 -y

RUN apt-get install flac -y
RUN apt-get install python3-pip python-tk -y
RUN pip3 install kaldi_io scikit-learn imblearn imbalanced-learn seaborn
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        g++ \
        make \
        automake \
        autoconf \
        bzip2 \
        unzip \
        wget \
        sox \
        libtool \
        git \
        subversion \
        python2.7 \
        zlib1g-dev \
        gfortran \
        ca-certificates \
        patch \
        ffmpeg \
	vim && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python2.7 /usr/bin/python

RUN git clone --depth 1 https://github.com/kaldi-asr/kaldi.git /opt/kaldi && \
    cd /opt/kaldi/tools && \
    ./extras/install_mkl.sh && \
    make -j $(nproc) && \
    cd /opt/kaldi/src && \
    ./configure --shared && \
    make depend -j $(nproc) && \
    make -j $(nproc) && \
    find /opt/kaldi  -type f \( -name "*.o" -o -name "*.la" -o -name "*.a" \) -exec rm {} \; && \
    find /opt/intel -type f -name "*.a" -exec rm {} \; && \
    find /opt/intel -type f -regex '.*\(_mc.?\|_mic\|_thread\|_ilp64\)\.so' -exec rm {} \; && \
    rm -rf /opt/kaldi/.git
