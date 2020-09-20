#!/usr/bin/env bash

[[ $# -eq 3 ]] || { echo "Required args: model_name, img_dir, device" >&2; exit 1; }

MODEL=$1

CUDA_VISIBLE_DEVICES=$3 python run_experiments.py \
    --img_dir $2 \
    --model ${MODEL} \
    --run_dir ${MODEL} \
    --run_name test \
    --set_seed \
    --overwrite

# dcgan, began, vae
