import wandb
from CIFAR10_train import *

if __name__ == '__main__':

    # Hyperparameter sweep configuration
    hyperparam_dict = {
        "method": "bayes",
        "name": "hybrid_loss_sweep",
        "metric": {"goal": "maximize", "name": "test/best_prec1"},
        "parameters": {
            "lr": {"max": 1e-2, "min": 1e-5},
            "batch_size": {"values": [32, 64]},
            "_lambda": {"max": 0.5, "min": 0.001},
            "mode": {"values": ["ascending", "descending", None]},
            "step_size": {"max": 1e-5, "min": 1e-7}
        }
    }

    # Create the sweep
    sweep_id = wandb.sweep(sweep=hyperparam_dict, project="reverse_LRP_mnist")
    # Run the sweep agent with the wrapper function
    wandb.agent(sweep_id, function=main, count=20)