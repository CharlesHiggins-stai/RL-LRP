conda create --name rl_lrp_3 python=3.11

conda activate rl_lrp

conda install numpy==1.* pytorch torchvision pillow matplotlib seaborn tensorboard wandb -c pytorch -c nvidia -c conda-forge