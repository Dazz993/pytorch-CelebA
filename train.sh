#!/bin/bash

echo "Begin to train model: '${1}' on gpu ${2}"
python train.py --cfg configs/${1}.yaml --gpu ${2} |& tee -a docs/logs/${1}.log