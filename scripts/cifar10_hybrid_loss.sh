#!/bin/bash

for model in vgg11
do
    echo "python experiments/CIFAR10_train.py  --arch=$model  --save-dir="data/save_$model" |& tee -a log_$model"
    python experiments/CIFAR10_train.py  --arch=$model  --save-dir=data/save_$model 
done