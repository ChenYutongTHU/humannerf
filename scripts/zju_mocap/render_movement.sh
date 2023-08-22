#!/usr/bin/env bash

SUBJECT=$1
if [ -z "${SUBJECT}" ]
then
    SUBJECT=387
fi

for exp in nonrigid/wodelay
do
CUDA_VISIBLE_DEVICES=0 python run.py \
    --type movement \
    --cfg ./configs/human_nerf/zju_mocap/${SUBJECT}/adventure.yaml \
    experiment ${exp} \
    load_net latest eval.metrics ["lpips"\,"mse"\,"ssim"]
done
