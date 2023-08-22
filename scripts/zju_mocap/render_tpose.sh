#!/usr/bin/env bash

SUBJECT=$1
if [ -z "${SUBJECT}" ]
then
    SUBJECT=387
fi

# render T-pose
for exp in loss/l1.0m0.0 #loss/l0.0m0.2 adventure nonrigid/wonr
do
CUDA_VISIBLE_DEVICES=3 python run.py \
    --type tpose \
    --cfg ./configs/human_nerf/zju_mocap/${SUBJECT}/adventure.yaml \
    experiment ${exp} \
    load_net latest eval.metrics []
done
