#!/bin/bash

current_create_time=`date +"%Y_%m_%d_%H_%M_%S"`
echo "Begin to train model: '${1}' on gpu ${2}"
#python train.py --cfg configs/${1}.yaml --gpu ${2} |& tee -a docs/logs/${1}_${current_create_time}.log
python train.py --cfg configs/${1}.yaml --gpu ${2}