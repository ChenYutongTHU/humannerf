#!/usr/bin/env bash

SUBJECT=$1
if [ -z "${SUBJECT}" ]
then
    SUBJECT=387
fi

FREE_VIEW_FRAME_IDX=$2
if [ -z "${FREE_VIEW_FRAME_IDX}" ]
then
    FREE_VIEW_FRAME_IDX=0
fi

for exp in nonrigid/wodelay
do
CUDA_VISIBLE_DEVICES=0 python run.py \
    --type freeview \
    --cfg ./configs/human_nerf/zju_mocap/${SUBJECT}/adventure.yaml \
    load_net latest \
    freeview.frame_idx ${FREE_VIEW_FRAME_IDX} \
    experiment ${exp} eval.metrics []
done 

