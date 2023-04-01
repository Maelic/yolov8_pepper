#!/bin/sh
project=yolov8-benchmark
model=yolov8n
epochs=5
resolution=640

name=$model-$resolution-$epochs
checkpoint=$model.pt

RED='\033[0;31m'
NC='\033[0m' # No Color
On_Yellow='\033[43m'
echo "${RED} Launching training with args: \n ${On_Yellow} - project: $project \n \
 - model: $model \n \
 - epochs: $epochs \n \
 - resolution: $resolution \n \
 - name: $name ${NC}"

# YOLO nano
yolo task=detect mode=train epochs=$epochs batch=-1 model=$checkpoint data=custom_YCB.yaml imgsz=$resolution name=$name wandb=$project
