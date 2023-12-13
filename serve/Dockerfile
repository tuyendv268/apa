FROM python:3.9.12

RUN pip install pyyaml

USER root

RUN apt-get update && \
    apt-get install -y python2.7 \
    autoconf \
    automake \
    cmake \
    curl \
    g++ \
    git \
    graphviz \
    libatlas3-base \
    libtool \
    make \
    pkg-config \
    sox \
    subversion \
    unzip \
    wget \
    zlib1g-dev

RUN pip install setuptools
RUN pip install --upgrade setuptools

RUN pip install --upgrade pip  numpy   pyparsing    jupyter    ninja

RUN git clone  https://github.com/pykaldi/pykaldi.git /pykaldi

RUN apt-get install gfortran -y

RUN cd /pykaldi/tools \
   && ./check_dependencies.sh \
   &&  ./install_protobuf.sh \
   &&  ./install_clif.sh 

RUN git clone -b pykaldi_02 https://github.com/pykaldi/kaldi.git /pykaldi/tools/kaldi \
    && cd /pykaldi/tools/kaldi/tools \
    && ./extras/install_mkl.sh

RUN cd /pykaldi/tools \
    &&  ./install_kaldi.sh 

RUN cd /pykaldi \
    && KALDI_DIR=/pykaldi/tools/kaldi python setup.py install
RUN pip install kaldiio scipy soundfile