for i in $seq 1 5; do
    for mode in "ground_truth_label" "learner_label" "default"; do
        python experiments/pure_loss_mnist.py --mode $mode
        echo "done"
    done 
done