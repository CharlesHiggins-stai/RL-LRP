#!/bin/bash
batch_size=32
lr=0.004982947729152929
momentum=0.9
_lambda=0.0944716
top_percent=0.28175657408869537
model=vgg11


for i in $(seq 0 1 5);
    do
        echo "python experiments/CIFAR10_train_clean.py  --arch=$model  --_lambda $_lambda --save-dir="data/save_$model" |& tee -a log_$model"
        python experiments/CIFAR10_train_clean.py  --arch=$model  --batch-size $batch_size  --lr $lr --top_percent $top_percent --_lambda $_lambda --save-dir=data/save_$model_lambda_$_lambda_$i --teacher_heatmap_mode 'default' 

        echo "python experiments/CIFAR10_train_clean.py  --arch=$model  --_lambda $_lambda --save-dir="data/save_$model" |& tee -a log_$model"
        python experiments/CIFAR10_train_clean.py  --arch=$model  --batch-size $batch_size  --lr $lr --top_percent $top_percent --_lambda $_lambda --save-dir=data/save_$model_lambda_$_lambda_$i_sanity_check --teacher_heatmap_mode 'sanity_check' 
    done
