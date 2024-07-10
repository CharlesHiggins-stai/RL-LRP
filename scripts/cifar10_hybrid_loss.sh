#!/bin/bash

for model in vgg11;
    do
    for it in $(seq 1 5);
        do
        for lmda in 0.5 0.1 0.01 0.001 0.0001;
            do
            echo "python experiments/CIFAR10_train.py  --arch=$model  --_lambda $lmda --save-dir="data/save_$model" |& tee -a log_$model"
            python experiments/CIFAR10_train.py  --arch=$model   --_lambda $lmda --save-dir=data/save_$model_lambda_$lmda  
        done
    done
done