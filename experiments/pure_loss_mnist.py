
import sys
sys.path.append('/Users/charleshiggins/Personal/CharlesPhD/CodeRepo/xai_intervention/RL-LRP/')
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import baselines.trainVggBaselineForCIFAR10.vgg as vgg
from experiments import SimpleNet, apply_threshold, CosineDistanceLoss, ManualCNN, HybridCosineDistanceCrossEntopyLoss
from matplotlib import pyplot as plt
import numpy as np
import argparse
import wandb
from PIL import Image
import io
# comment out when running locally
from experiments import WrapperNet
# comment out when running locally  

def train_model(model, optimizer, criterion, train_loader, device, attention_function, print_freq=10):
    total_losses = []
    cosine_losses = []
    cross_entroy_losses = []
    empty_batches = 0
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device) 
        with torch.no_grad():
            classifications = model.model(data)
            indxs = (classifications.argmax(dim=1) == target).nonzero()
            if indxs.size(0) == 0:
                empty_batches += 1
                continue
        correct_indicies = indxs[0]
        pruned_data = data[correct_indicies]
        pruned_target = target[correct_indicies]
        model.train()
        target_map = attention_function(pruned_data, threshold=0.95)
        output_classification, output = model(pruned_data)
        total_loss, cosine_loss, cross_entropy_loss = criterion(output, target_map, output_classification, pruned_target)
        optimizer.zero_grad()
        cross_entropy_loss.backward()
        optimizer.step()
        total_losses.append(total_loss.item())
        cosine_losses.append(cosine_loss.item())
        cross_entroy_losses.append(cross_entropy_loss.item())
        if i % print_freq == 0:
            print(f"Batch: {i}, Total Loss: {np.mean(total_losses)}, Cosine Loss: {np.mean(cosine_losses)}, Cross Entropy Loss: {np.mean(cross_entroy_losses)}, empty_batches: {empty_batches}")
    wandb.log(
        {"train/combined_loss": np.mean(total_losses),
            "train/cosine_loss": np.mean(cosine_losses),
            "train/cross_entropy_loss": np.mean(cross_entroy_losses),
            "train/empty_batches": empty_batches
        })
    return model, np.mean(total_losses)

def train_simple_model(model, optimizer, criterion, train_loader, test_loader, device, max_epochs, print_freq=10):
    for x in range(max_epochs):
        losses = []
        accuracy_list = []
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            target_heatmap = apply_threshold(data, threshold=0.95)
            with torch.no_grad():
                output_classification = model.model(data)
                correct_indx = (output_classification.argmax(dim=1) == target).nonzero()
                if correct_indx.size(0) == 0:
                    print('empty batch -- skipping training data')
                    continue
                pruned_data = data[correct_indx[0]]
                pruned_target = target[correct_indx[0]]
                pruned_target_heatmap = target_heatmap[correct_indx[0]]
            
            pruned_output, pruned_heatmap = model(pruned_data)
            loss, cosine_loss, cross_entropy_loss = criterion(pruned_heatmap, pruned_target_heatmap, pruned_output, pruned_target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            output, heatmap = model(data)
            loss, cosine_loss, cross_entropy_loss = criterion(heatmap, target_heatmap, output, target)
            optimizer.zero_grad()
            cosine_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            classifications = output.argmax(dim=1).squeeze()
            correct = classifications.eq(target).sum().item()
            accuracy = (correct / len(target)) * 100
            
            losses.append(loss.item())
            accuracy_list.append(accuracy)
        print('Epoch: ', x)
        print(f"Train Loss: {np.mean(losses)}, Train Accuracy: {np.mean(accuracy_list)}")
        model.eval()
        test_losses = []
        test_accuracies = []
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                output, heatmap = model(data)
                target_heatmap = apply_threshold(data, threshold=0.95)
                loss, cosine_loss, cross_entropy_loss = criterion(heatmap, target_heatmap, output, target)
            classifications = output.argmax(dim=1).squeeze()
            correct = classifications.eq(target).sum().item()
            accuracy = (correct / len(target)) * 100
            test_losses.append(loss.item())
            test_accuracies.append(accuracy)
        print(f"Test Loss: {np.mean(test_losses)}, Test Accuracy: {np.mean(test_accuracies)}")


        

def test_model(model, criterion, test_loader, device, attention_function):
    model.eval()
    test_loss = 0
    correct = 0
    total_seen = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            target_map = attention_function(data, threshold=0.95)
            output_classification, output = model(data)
            loss, cosine_loss, cross_entropy_loss = criterion(output, target_map, output_classification, target)
            test_loss += loss.item()
            correct += output_classification.argmax(dim=1).eq(target).sum().item()
            total_seen += len(target)
    test_loss /= total_seen
    accuracy = (correct / total_seen) * 100
    print(f"Test Loss: {test_loss}, Accuracy: {accuracy}")
    wandb.log({"accuracy": accuracy})
    return model, test_loss, accuracy    

def plot_heatmap_comparison(model, test_loader, device, attention_function, epoch):
    data, target = next(iter(test_loader))
    target_map = attention_function(data, threshold=0.95)
    output_classification, output = model(data.to(device), target.to(device))
    num = np.random.randint(0, len(target))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(output[num][0].detach().numpy(), cmap='hot')
    axes[0].set_title(f'LRP Output after {epoch} iterations')
    axes[1].imshow(target_map[num][0], cmap='hot')
    axes[1].set_title('Target Heatmap (Ground Truth)')
    axes[2].imshow(data[num][0].detach().numpy(), cmap='gray')
    axes[2].set_title('Input Image (Original)')
    
    # Save the figure to a BytesIO buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    wandb.log({"Example Image": wandb.Image(img)})
    # plt.show()
    
def checkpoint_model(model, output_dir, epoch:int):
    torch.save(model.state_dict(), f"{output_dir}/model_{epoch}.pt")
    
def main(model, optimizer, criterion, train_loader, test_loader, device, apply_threshold):
    max_epochs = wandb.config.max_epochs
    visualise_frequency = wandb.config.visualize_freq
    for x in range(max_epochs):
        # if x % visualise_frequency== 0:
        #     plot_heatmap_comparison(model, test_loader, device, apply_threshold, x)
        # if x % wandb.config.save_frequency == 0:
        #     checkpoint_model(model, wandb.config.output_dir, x)
        print(f'Epoch: {x}')
        model, train_loss = train_model(model, optimizer, criterion, train_loader, device, apply_threshold)
        model, test_classification_loss, accuracy = test_model(model, criterion, test_loader, device, apply_threshold)
        # print(f'Epoch: {x}, Test Classification Loss: {test_classification_loss}, Accuracy: {accuracy}')
        print(f'Epoch: {x}, Training Loss: {train_loss}')
    checkpoint_model(model, wandb.config.output_dir, max_epochs)
    print('completed training.')
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process input arguments for a simulation.')

    # Adding arguments
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility (default: 0).')
    parser.add_argument('--batch_size', type=int, default = 64, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default = 1e-4, help='Learning rate for training.')
    parser.add_argument('--output_dir', type=str, default = "experiments/data", help='Directory to save output results.')
    parser.add_argument('--save_frequency', type=int, default = 25, help='Frequency to save model checkpoints.')
    parser.add_argument('--data_dir', type=str, default = "experiments/data", help='Directory to find training data')
    parser.add_argument('--accuracy_threshold', type=int, default = 50, help='Reward threshold to stop training.')
    parser.add_argument('--_lambda', type=float, default = 0.5, help='balance the loss between cross entropy and cosine distance loss')
    parser.add_argument('--max_epochs', type=int, default = 100, help='Maximum number of epochs to train for.')
    parser.add_argument('--visualize_freq', type=int, default = 5, help='Frequency to visualize the heatmaps')
    parser.add_argument('--tags', nargs='+', default = ["experiment", "mnist_test", "hybrid loss"], help='Tags for wandb runs')

    # Parse the arguments
    args = parser.parse_args()
    
    wandb.init(project="hybrid_loss_mnist", tags=["hybrid_loss", "mnist", "supervised_learning", *args.tags], mode="disabled")
    wandb.config.update(args)
    wandb.config.update({"experiment_class": "pure_loss_mnist"})
    # define device for GPU compatibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and transform datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(wandb.config.data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(wandb.config.data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    # Initialize the network and optimizer for the underlying network
    # torch.set_random_seed(wandb.config.seed)
    # now wrap the network in the LRP class
    # model = WrapperNet(SimpleNet(), hybrid_loss=True)
    # optimizer= optim.Adam(model.parameters(), lr=1e-3)

    # # define the loss functions for each
    # # lambda parameter weights cross entropy loss with CosineDistance. 
    # # The higher the lambda parameter, the more weight is given to the cosine distnace loss
    # criterion = HybridCosineDistanceCrossEntopyLoss(_lambda=wandb.config._lambda)
    # # Move to device
    # model.to(device)
    # main(model, optimizer, criterion, train_loader, test_loader, device, apply_threshold)
    model = WrapperNet(SimpleNet(), hybrid_loss=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, eps=1e-8)
    criterion = HybridCosineDistanceCrossEntopyLoss(_lambda=0.1)
    train_simple_model(model, optimizer, criterion, train_loader, test_loader, device, wandb.config.max_epochs)
