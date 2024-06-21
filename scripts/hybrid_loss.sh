# bin/bash
# Run the hybrid loss experiment
# define arguments here
seed=0
batch_size=64
lr=1e-3
output_dir="/home/charleshiggins/RL-LRP/data/MNIST/raw/model_files"
save_frequency=25
data_dir="/home/charleshiggins/RL-LRP/testing_sandbox/data"
accuracy_threshold=95
# _lambda=0.5
max_epochs=100
max_steps=2100 
visualize_frequency=5

for i in $(seq 0 1 5); do

    for i in $(seq 0.0 0.1 1); do
        echo "Running hybrid loss experiment with lambda $i"
        python3 experiments/hybrid_loss_mnist.py --seed $seed --batch_size $batch_size --lr $lr --output_dir $output_dir --save_frequency $save_frequency --data_dir $data_dir --accuracy_threshold $accuracy_threshold --_lambda $i --max_epochs $max_epochs --visualize_freq $visualize_frequency --tags "experiment" "hybrid loss"
        echo "Hybrid loss experiment with lambda $i complete"
    done
    echo "Hybrid loss experiment complete"


    for i in $(seq 0.0 0.1 1); do
        echo "Running hybrid loss experiment with ascending lambda"
        python3 experiments/hybrid_loss_mnist.py --seed $seed --batch_size $batch_size --lr $lr --output_dir $output_dir --save_frequency $save_frequency --data_dir $data_dir --accuracy_threshold $accuracy_threshold --_lambda $i --max_epochs $max_epochs --visualize_freq $visualize_frequency --loss_mode "increasing" --max_steps $max_steps --tags "experiment" "hybrid loss" "loss scheduler" "ascending"
        echo "Hybrid loss experiment with lambda $i complete"
    done
    echo "Hybrid loss experiment complete"

    for i in $(seq 0.0 0.1 1); do
        echo "Running hybrid loss experiment with descending lambda"
        python3 experiments/hybrid_loss_mnist.py --seed $seed --batch_size $batch_size --lr $lr --output_dir $output_dir --save_frequency $save_frequency --data_dir $data_dir --accuracy_threshold $accuracy_threshold --_lambda $i --max_epochs $max_epochs --visualize_freq $visualize_frequency --loss_mode "decreasing" --max_steps $max_steps --tags "experiment" "hybrid loss" "loss scheduler" "descending"
        echo "Hybrid loss experiment with lambda $i complete"
    done
    echo "Hybrid loss experiment complete"

done 