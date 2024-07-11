import wandb

hyperparam_dict = {
    "method": "bayes",
    "name": "hybrid_loss_sweep",
    "metric": {"goal": "maximize", "name": "prec1"},
    "parameters": {
        "lr" : {"max": 1e-2, "min": 1e-5},
        "batch_size": {"values": [32, 64, 128]},
        "_lambda": {"max": 0.5, "min": 0.001},
        "mode": {"values": ["ascending", "descending", None]},
        "step_size": {"max": 1e-3, "min": 1e-5}
    }
    
}
sweep_id = wandb.sweep(sweep=hyperparam_dict, project="reverse_LRP_mnist")

