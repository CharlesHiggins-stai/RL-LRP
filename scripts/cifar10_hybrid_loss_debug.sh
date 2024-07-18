#!/bin/bash

for model in vgg11;
    do
    for it in 0;
        do
        for lmda in 0.1 0.01 0.001 0.0001;
            do
            echo "python experiments/CIFAR10_train.py  --arch=$model  --_lambda $lmda --save-dir="data/save_$model" --teacher_heatmap_mode ground_truth_target |& tee -a log_$model"
            python experiments/CIFAR10_train.py  --arch=$model   --_lambda $lmda --save-dir=data/save_$model_lambda_$lmda  --teacher_heatmap_mode ground_truth_target
        done
    done
done