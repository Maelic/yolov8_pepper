#!/bin/sh
# Script variables
RED='\033[0;31m'
NC='\033[0m' # No Color
On_Yellow='\033[43m'

# Training and dataset variables
# WARNING: project name should match the one from roboflow!
pepper_ip=pepper2.local
project=robocup-images
model=yolov8n
epochs=200
resolution=320
# Paths
base_path=$(pwd)/robocup2023
dataset_path=$base_path/dataset
model_path=$base_path/model

# start a timer
start=$(date +%s.%N)

# test if model path exist
if [ ! -d "$model_path" ]; then
  mkdir $model_path
else
    echo "${RED} Model path already exist, deleting content! ${NC}"
    rm -rf $model_path/*
fi

# test if dataset path exist
if [ ! -d "$model_path" ]; then
  mkdir $model_path
else
    echo "${RED} Model path already exist, deleting content! ${NC}"
    rm -rf $model_path/*
fi

task=robocup
name=$model-$task
checkpoint=$model.pt

# STEP 1: download dataset with roboflow
echo "\n ${RED} Downloading dataset with args: \n ${On_Yellow} - location: $dataset_path ${NC} \n"
python3 robocup2023/download.py --location $dataset_path --project $project

# STEP 2: train model

echo "\n ${RED} Launching training with args: \n ${On_Yellow} - project: $project \n \
 - model: $model \n \
 - epochs: $epochs \n \
 - resolution: $resolution \n \
 - name: $name ${NC} \n"

yaml_file=$dataset_path/data.yaml
echo "yaml file: $yaml_file"

# training command
yolo task=detect mode=train epochs=$epochs batch=-1 model=$checkpoint data=$yaml_file imgsz=$resolution project=$model_path name=$name wandb=$project patience=50

# STEP 3: export to onnx
model_name_pt=$model_path/$name/weights/$task.pt
model_name_onnx=$model_path/$name/weights/$task.onnx
mv $model_path/$name/weights/best.pt $model_name_pt

yolo export model=$model_name_pt format=onnx simplify=True opset=11

# print total time elapse
end=$(date +%s.%N)
runtime=$(python3 -c "print(${end} - ${start})")
echo "\n ${RED} Total time elapse: $runtime ${NC} \n"

# STEP 4: export model to robot
# rename the best.pt checkpoint
echo "$\n {RED} Exporting model to robot with args: \n ${On_Yellow} - model name: $name \n \
 - model path: $model_path \n \
 - model name onnx: $model_name_onnx \n \
 - dest path: $dest_path ${NC} \n"

dest_path=/home/nao/robobreizh_pepper_ws/src/perception_pepper/scripts/models/ObjectDetection/YOLOV8/weights/robocup
tmp_file=$dataset_path/$task.txt
scp $model_name_onnx nao@$pepper_ip:$dest_path/$task.onnx
# get object list from yaml file
python3 robocup2023/get_object_list.py --yaml $yaml_file --dest $tmp_file
scp $tmp_file nao@$pepper_ip:$dest_path/$task.txt