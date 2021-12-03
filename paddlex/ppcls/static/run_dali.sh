#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export FLAGS_fraction_of_gpu_memory_to_use=0.80

python3.7 -m paddle.distributed.launch \
    --gpus="0,1,2,3,4,5,6,7" \
    ppcls/static/train.py \
    -c ./ppcls/configs/ImageNet/ResNet/ResNet50_fp16.yaml \
    -o Global.use_dali=True
