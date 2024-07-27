import wandb
from CIFAR10_train import *

if __name__ == '__main__':

    # Hyperparameter sweep configuration
    hyperparam_dict = {
        'name': 'alpha-beta-sweep',
        "method": "bayes",
        "name": "hybrid_loss_sweep",
        "metric": {"goal": "maximize", "name": "test/best_prec1"},
        "parameters": {
            "lr": {"max": 1e-2, "min": 1e-5},
            "batch_size": {"values": [32, 64, 128]},
            "_lambda": {"max": 0.5, "min": 0.001},
            "mode": {"values": ["ascending", "descending", None]},
            "step_size": {"max": 1e-5, "min": 1e-7},
            "top_percent": {"max": 0.9, "min": 0.1},
            "teacher_heatmap_mode": {"values": ["ground_truth_target", "learner_label", "default"]}
        }
    }

    # Create the sweep
    sweep_id = wandb.sweep(sweep=hyperparam_dict, project="LRP-CIFAR10")
    # Run the sweep agent with the wrapper function
    # sweep_id = "charles-higgins/reverse_LRP_mnist/q6bk7fd4"
    wandb.agent(sweep_id, function=main, count=20)