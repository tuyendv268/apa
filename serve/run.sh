#!/usr/bin/env bash
. ./path.sh

export TMPDIR=/workspace/serve/exps/logs
serve run serve:app --host 0.0.0.0 --port 9999
