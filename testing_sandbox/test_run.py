import sys
sys.path.append('..')
from experiments import perform_gradcam, perform_lrp_captum
from internal_utils import preprocess_images, condense_to_heatmap, blur_image_batch, add_random_noise_batch, get_data_imagenette, get_teacher_model, get_CIFAR10_dataloader
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from experiments import WrapperNet, WrapperNetContrastive
import torch
from internal_utils import update_dictionary_patch
from baselines.trainVggBaselineForCIFAR10.vgg import vgg11
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import numpy as np
import torch.nn as nn
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def get_teacher_model(teacher_checkpoint_path):
    checkpoint = torch.load(teacher_checkpoint_path)
    # assume teacher model is vgg11 for now
    teacher = vgg11()
    try: 
        checkpoint = update_dictionary_patch(checkpoint)
        teacher.load_state_dict(checkpoint['new_state_dict'])
    except:
        print('Incorrect patch specified')
    return teacher
# data = get_CIFAR10_dataloader(train=True, batch_size=8)
# data_test = get_CIFAR10_dataloader(train=False, batch_size=8)
# input_images, labels = next(iter(data))
# teacher_model = WrapperNet(get_teacher_model("/home/charleshiggins/RL-LRP/baselines/trainVggBaselineForCIFAR10/save_vgg11/checkpoint_299.tar"), hybrid_loss=True)
# # define params
# learner_model = WrapperNet(vgg11(), hybrid_loss=True)


class CosineDistanceLoss(torch.nn.Module):
    def __init__(self):
        super(CosineDistanceLoss, self).__init__()

    def forward(self, input1, input2):
        # Flatten the images: shape from [b, 1, 28, 28] to [b, 784]
        input1_flat = input1.view(input1.size(0), -1)
        input2_flat = input2.view(input2.size(0), -1)
        
        # Compute cosine similarity, then convert to cosine distance
        cosine_sim = F.cosine_similarity(input1_flat, input2_flat)
        cosine_dist = 1 - cosine_sim
        
        # Calculate the mean of the cosine distances
        loss = cosine_dist.mean()
        # loss = F.mse_loss(input1_flat, input2_flat)
        return loss
    

# Define SSIM loss (we'll minimize 1 - SSIM)
class SSIMLoss(torch.nn.Module):
    def __init__(self, data_range=1.0, size_average=True, channel=3):
        super(SSIMLoss, self).__init__()
        self.ssim_module = SSIM(data_range=data_range, size_average=size_average, channel=channel)
    
    def forward(self, img1, img2):
        ssim_value = self.ssim_module(img1, img2)
        return 1 - ssim_value
    
class Tracker:
    # small tracker class to compute moving averages
    def __init__(self):
        self.ce_losses = []
        self.ssim_losses = []
        self.accs = []
    
    def update(self, ce_loss, ssim_loss, acc):
        # add values to the tracker
        self.ce_losses.append(ce_loss)
        self.ssim_losses.append(ssim_loss)
        self.accs.append(acc)
    
    
    def get_avg(self):
        # get the average of the last 10 values
        if len(self.ce_losses) <= 10:
            return np.mean(self.ce_losses), np.mean(self.ssim_losses), np.mean(self.accs)
        else: 
            return np.mean(self.ce_losses[-10:]), np.mean(self.ssim_losses[-10:]), np.mean(self.accs[-10:])
    
    def get_results(self):
        results = {
            "CrossEntropyLoss": np.array(self.ce_losses),
            "SSIMLoss": np.array(self.ssim_losses),
            "Accuracy": np.array(self.accs)
        }
        return results

            
    
def remove_grad_for_all_but_last_layer(learner_model, optimizer, scheduler, verbose=False):
    for name, module in learner_model.model.named_modules():
        if not isinstance(module, nn.Sequential) \
        and not isinstance(module, WrapperNet) \
        and not len(list(module.children())) > 0 \
            and type(module) not in [nn.ReLU, nn.MaxPool2d, nn.AdaptiveAvgPool2d, nn.LogSoftmax, nn.Dropout]:
            if verbose:
                print(name)
                print(module)
                print("####################### \n")
            if "classifier.6" not in name:
                print(f"removing grad from: {name}")
                for param in module.parameters():
                    param.requires_grad = False
            else:
                print(f"Grad will continue for {name}")
    optimizer = torch.optim.SGD(learner_model.parameters(), lr=0.5, momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

    return learner_model, optimizer, scheduler


def train_baseline(data, teacher_model, epochs=100):
    # we pass in the optimizer and scheduler to allow for the learning rate to be updated and changed
    # to keep things the same
    learner_model = WrapperNet(vgg11(), hybrid_loss=True)
    optimizer = torch.optim.SGD(learner_model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    ssim_loss = SSIMLoss()
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    tracker = Tracker()
    for epchs in range(epochs):
        # first record the loss and output just to check -- but don't record anything
        for i, (images, labels) in enumerate(data):
            pp_images = preprocess_images(images)
            with torch.no_grad():
                _, teacher_heatmap = teacher_model(pp_images)
                output, model_heatmap = learner_model(pp_images)
                img_loss = ssim_loss(model_heatmap, teacher_heatmap)
            
            model_out = learner_model.model(pp_images)
            # loss for measurement, but not for backprop
            
            # now training loop starts
            optimizer.zero_grad()
            acc_loss = cross_entropy_loss(model_out, labels)
            acc_loss.backward()
            torch.nn.utils.clip_grad_norm_(learner_model.parameters(), 1)
            optimizer.step()
            # scheduler.step()
            # store results
            correct = model_out.argmax(dim=1).eq(labels).sum().item()
            correct_pct = 100 * correct/labels.shape[0] 
            tracker.update(ce_loss = acc_loss.item(), ssim_loss = img_loss.item(), acc = correct_pct)
            mov_ce, mov_ssim, mov_acc = tracker.get_avg()
            print(f"iteration: {epchs}:{i} \t accuracy: {correct_pct:.4f} ({mov_acc:.4f})\t image loss: {img_loss.float():.4f} ({mov_ssim:.4f})\t cross entropy loss: {acc_loss.float():.4f} ({mov_ce:.4f}) \t learning rate: {optimizer.param_groups[0]['lr']:.4f}")
            # clean up hooks
            learner_model.remove_hooks()
            learner_model.reapply_hooks()
    results = tracker.get_results()
    print("#" * 10)
    print("#" * 10)
    print("Baseline training completed")
    print("#" * 10)
    print("#" * 10)
    return results, learner_model

# get the dataset
data = get_CIFAR10_dataloader(train=True, batch_size=64)
data_test = get_CIFAR10_dataloader(train=False, batch_size=64)
# load data and preprocess
input_images, labels = next(iter(data))
pp_images = preprocess_images(input_images)
# break to avoiding having to reload the data everytime
# pp_images, labels = pp_images[:3], labels[:3]
# define losses
mse_loss = torch.nn.MSELoss()
cos_loss = CosineDistanceLoss()
cross_entropy_loss = torch.nn.CrossEntropyLoss()
ssim_loss = SSIMLoss()
# get clean teacher models
teacher_model = WrapperNet(get_teacher_model("/home/charleshiggins/RL-LRP/baselines/trainVggBaselineForCIFAR10/save_vgg11/checkpoint_299.tar"), hybrid_loss=True)

CHANGE_POINT = 0
EPOCHS = 2
tracker = Tracker()

print("target labels: ", labels)
with torch.no_grad():
    _, teacher_heatmap = teacher_model(pp_images, labels)

# train a baseline for regulat performance -- this is a model trained only on cross-entropy loss
# baseline_results, baseline_model = train_baseline(data, teacher_model, epochs=EPOCHS)

# redefine optimizer and learner model to avoid any confusing results
learner_model = WrapperNet(vgg11(), hybrid_loss=True)
# define optimizers
optimizer = torch.optim.SGD(learner_model.parameters(), lr=0.5, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.91)
############################################
# Now we will train the model with only the SSIM loss
############################################
for epch in range(0, CHANGE_POINT):
    for i, (images, labels) in enumerate(data):
        pp_images = preprocess_images(images)
        with torch.no_grad():
            teacher_out, teacher_heatmap = teacher_model(pp_images, labels)
        model_out, model_heatmap = learner_model(pp_images, labels)
        # loss for measurement, but not for backprop
        acc_loss = cross_entropy_loss(model_out, labels)
        # now training loop starts
        optimizer.zero_grad()
        img_loss = 0.1 * ssim_loss(model_heatmap, teacher_heatmap)
        img_loss.backward()
        torch.nn.utils.clip_grad_norm_(learner_model.parameters(), 1)
        optimizer.step()
        scheduler.step()
        # store results
        correct = model_out.argmax(dim=1).eq(labels).sum().item()
        correct_pct = 100 * correct/labels.shape[0] 
        tracker.update(ce_loss = acc_loss.item(), ssim_loss = img_loss.item(), acc = correct_pct)
        mov_ce, mov_ssim, mov_acc = tracker.get_avg()
        print(f"iteration: {epch}: {i} \t accuracy: {correct_pct:.4f} ({mov_acc:.4f})\t image loss: {img_loss.float():.4f} ({mov_ssim:.4f})\t cross entropy loss: {acc_loss.float():.4f} ({mov_ce:.4f}) \t learning rate: {optimizer.param_groups[0]['lr']:.4f}")
        # clean up hooks
        learner_model.remove_hooks()
        learner_model.reapply_hooks()
        teacher_model.remove_hooks()
        teacher_model.reapply_hooks()

# Write logic now to update or change the model/parameters/optimizers 
del optimizer
optimizer = torch.optim.SGD(learner_model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)

    
# training loop after change
print("############" * 4)
print("## Training loop after change ##")
print("############" * 4)
for epch in range(CHANGE_POINT, EPOCHS):
    for i, (images, labels) in enumerate(data):
        pp_images = preprocess_images(images)
        with torch.no_grad():
            teacher_out, teacher_heatmap = teacher_model(pp_images)
        model_out, model_heatmap = learner_model(pp_images)
        # loss for measurement, but not for backprop
        # img_loss = ssim_loss(model_heatmap, teacher_heatmap)
        # now training loop starts
        optimizer.zero_grad()
        img_loss = ssim_loss(model_heatmap, teacher_heatmap)
        img_loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(learner_model.parameters(), 1)
        # run through model again
        model_out, model_heatmap = learner_model(pp_images)
        optimizer.zero_grad()
        acc_loss = cross_entropy_loss(model_out, labels)
        acc_loss.backward()
        # total = 0.1 * img_loss + acc_loss
        # total.backward()
        # torch.nn.utils.clip_grad_norm_(learner_model.parameters(), 1)
        optimizer.step()
        # scheduler.step()
        # store results
        correct = model_out.argmax(dim=1).eq(labels).sum().item()
        correct_pct = 100 * correct/labels.shape[0] 
        tracker.update(ce_loss = acc_loss.item(), ssim_loss = img_loss.item(), acc = correct_pct)
        mov_ce, mov_ssim, mov_acc = tracker.get_avg()
        print(f"iteration:{epch} : {i} \t accuracy: {correct_pct:.4f} ({mov_acc:.4f})\t image loss: {img_loss.float():.4f} ({mov_ssim:.4f})\t cross entropy loss: {acc_loss.float():.4f} ({mov_ce:.4f}) \t learning rate: {optimizer.param_groups[0]['lr']:.4f}")
        # clean up hooks
        learner_model.remove_hooks()
        teacher_model.remove_hooks()
        learner_model.reapply_hooks()
        teacher_model.reapply_hooks()
results = tracker.get_results()         

baseline_results, baseline_model = train_baseline(data, teacher_model, epochs=EPOCHS)

def visualise_results(baseline_results, working_results):
    """
    Plot the results from dictionaries using seaborn. 
    Convert to dataframes and then plot using seaborn and matplotlib.
    Results objects are in the form of a dict: 
    results = {
            "CrossEntropyLoss": np.array(self.ce_losses),
            "SSIMLoss": np.array(self.ssim_losses),
            "Accuracy": np.array(self.accs)
        }
    """
    
    # Convert the dictionaries to dataframes
    baseline_df = pd.DataFrame.from_dict(baseline_results)
    working_df = pd.DataFrame.from_dict(working_results)
    
    # Create a figure and axes for subplots
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), sharex=True)
    metrics = baseline_df.columns
    
    # Plot each metric in a separate subplot
    for i, metric in enumerate(metrics):
        sns.lineplot(ax=axes[i], x=range(len(baseline_df[metric])), y=baseline_df[metric].rolling(10).mean(), label='Baseline')
        sns.lineplot(ax=axes[i], x=range(len(working_df[metric])), y=working_df[metric].rolling(10).mean(), label='Explanation + CE loss')
        
        axes[i].set_title(metric)
        axes[i].set_xlabel('Epochs')
        axes[i].set_ylabel('Value')
        axes[i].legend()

    # Adjust layout
    plt.tight_layout()
    # Save the figure
    plt.savefig("results_plot_1.png", dpi=300, bbox_inches='tight')
    plt.show()
    
visualise_results(baseline_results, tracker.get_results())