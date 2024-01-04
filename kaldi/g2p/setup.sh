
apt-get update \
    && apt-get install -y python-dev swig
curl https://bootstrap.pypa.io/pip/2.7/get-pip.py \
    --output get-pip.py

python2 get-pip.py
python2 -m pip install numpy six
extras/install_sequitur.sh
