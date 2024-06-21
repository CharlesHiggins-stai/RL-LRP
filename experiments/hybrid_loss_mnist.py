
import sys
sys.path.append('/home/charleshiggins/RL-LRP')
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from experiments import SimpleRNet, apply_threshold, CosineDistanceLoss, ManualCNN, HybridCosineDistanceCrossEntopyLoss
from matplotlib import pyplot as plt
import numpy as np
import argparse
import wandb
from PIL import Image
import io
# comment out when running locally
from experiments import WrapperNet
# comment out when running locally  

def train_model(model, optimizer, criterion, train_loader, device, attention_function):
    total_loss = 0
    model.train()
    for _ in range(10):
        data, target = next(iter(train_loader))
        data, target = data.to(device), target.to(device)
        target_map = attention_function(data, threshold=0.95) # threshold is 0.99
        optimizer.zero_grad()
        output_classification, output = model(data)
        loss = criterion(output, target_map, output_classification, target)
        loss.backward()
        optimizer.step()
        wandb.log({"training loss": loss.item(), "_lambda": criterion._lambda})
        total_loss += loss.item()
    total_loss /= len(train_loader.dataset)
    return total_loss

def test_model(model, criterion, test_loader, device, attention_function):
    model.eval()
    test_loss = 0
    correct = 0
    total_seen = 0
    with torch.no_grad():
        for _ in range(10):
            data, target = next(iter(test_loader))
            data, target = data.to(device), target.to(device)
            target_map = attention_function(data, threshold=0.95)
            output_classification, output = model(data)
            test_loss += criterion(output, target_map, output_classification, target).item()
            correct += output_classification.argmax(dim=1).eq(target).sum().item()
            total_seen += len(target)
    test_loss /= total_seen
    accuracy = (correct / total_seen) * 100
    wandb.log({"accuracy": accuracy})
    return test_loss, accuracy    

def plot_heatmap_comparison(model, test_loader, device, attention_function, epoch):
    data, target = next(iter(test_loader))
    target_map = attention_function(data, threshold=0.95)
    output_classification, output = model(data.to(device), target.to(device))
    num = np.random.randint(0, len(target))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(output[num][0].cpu().detach().numpy(), cmap='hot')
    axes[0].set_title(f'LRP Output after {epoch} iterations')
    axes[1].imshow(target_map[num][0].cpu(), cmap='hot')
    axes[1].set_title('Target Heatmap (Ground Truth)')
    axes[2].imshow(data[num][0].cpu().detach().numpy(), cmap='gray')
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
    parser.add_argument('--loss_mode', type=str, help = "mode for the lambda scheduler to ascend or descend")
    parser.add_argument('--max_steps', type = int, default = 1000000, help='max number of steps, which also acts as a parameter for loss mode schedule')
    parser.add_argument('--tags', nargs='+', default = ["experiment", "mnist_test", "hybrid loss"], help='Tags for wandb runs')

    # Parse the arguments
    args = parser.parse_args()
    
    wandb.init(project="hybrid_loss_mnist", tags=["hybrid_loss", "mnist", "supervised_learning", *args.tags])
    wandb.config.update(args)
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
    model = WrapperNet(SimpleRNet(), hybrid_loss=True)
    optimizer= optim.Adam(model.parameters(), lr=wandb.config.lr)

    # define the loss functions for each
    # lambda parameter weights cross entropy loss with CosineDistance. 
    # The higher the lambda parameter, the more weight is given to the cosine distnace loss
    optim_mode = wandb.config.loss_mode if "loss_mode" in wandb.config.keys() else None
    criterion = HybridCosineDistanceCrossEntopyLoss(_lambda=wandb.config._lambda, mode = optim_mode, max_steps=wandb.config.max_steps)
    # Move to device
    model.to(device)

    max_epochs = wandb.config.max_epochs
    visualise_frequency = wandb.config.visualize_freq
    for x in range(max_epochs):
        if x % visualise_frequency== 0:
            plot_heatmap_comparison(model, test_loader, device, apply_threshold, x)
        if x % wandb.config.save_frequency == 0:
            checkpoint_model(model, wandb.config.output_dir, x)
            print(f'Epoch: {x}')
        test_classification_loss, accuracy = test_model(model, criterion, test_loader, device, apply_threshold)
        if accuracy > wandb.config.accuracy_threshold:
            break
        print(f'Epoch: {x}, Test Classification Loss: {test_classification_loss}, Accuracy: {accuracy}')
        train_loss = train_model(model, optimizer, criterion, train_loader, device, apply_threshold)
        print(f'Epoch: {x}, Training Loss: {train_loss}')
    checkpoint_model(model, wandb.config.output_dir, max_epochs)
    print('completed training.')
    print(f'number of total loss steps: {criterion.step_counter}')