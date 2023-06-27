#!/bin/sh
project=yolov8-few_shot-20
model=yolov8n
epochs=20
resolution=320
out_path=/home/maelic/Documents/training_/yolov8_few_shot/training

name=$model-$resolution-$epochs-big
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
yolo task=detect mode=train epochs=$epochs batch=-1 model=$checkpoint data=custom_IndoorVG.yaml imgsz=$resolution project=$out_path name=$name wandb=$project optimizer=SGD cos_lr=True
