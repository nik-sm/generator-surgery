#!/usr/bin/env bash

[[ $# -eq 2 ]] || { echo "Required args: model_name, img_dir" >&2; exit 1; }

MODEL=$1

python run_experiments.py --img_dir $2 \
--model ${MODEL} \
