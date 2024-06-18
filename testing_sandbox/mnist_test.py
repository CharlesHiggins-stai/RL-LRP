import sys
sys.path.append('/Users/charleshiggins/Personal/CharlesPhD/CodeRepo/xai_intervention/RL-LRP/')
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import wandb
from experiments import SimpleNet, apply_threshold, CosineDistanceLoss, WrapperNet


def test_inner_net(wrapped_model, device, test_loader):
    net = wrapped_model.model
    new_criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = torch.nn.functional.softmax(net(data), dim=1)
            test_loss += new_criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    wandb.log({"Inner Network Loss": test_loss, "Inner Network Accuracy": accuracy})
    print(f'Inner Network Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return

# Function to test the model
def test(wrapped_model, device, test_loader):
    wrapped_model.eval()
    test_loss = 0
    correct = 0
    # with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        target_map = apply_threshold(data, threshold=0.98)
        output = wrapped_model(data, target)
        test_loss += criterion(output, target_map).item()  # sum up batch loss
            # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    # accuracy = 100. * correct / len(test_loader.dataset)
    wandb.log({"Test Loss": test_loss})
    test_inner_net(wrapped_model, device, test_loader)
    return test_loss

# Training the model with early stopping
def train(wrapped_model, device, train_loader, optimizer, epoch, target_accuracy=99.0):
    wrapped_model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        target_map = apply_threshold(data, threshold=0.98)
        optimizer.zero_grad()
        # print(target.unsqueeze(1))
        output = wrapped_model(data, target)
        loss = criterion(output, target_map)
        loss.backward()
        optimizer.step()
        
        wandb.log({"Train Loss": loss.item()})

        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
            accuracy = test(wrapped_model, device, test_loader)
            # wandb.log({"Test Accuracy": accuracy})
            # if accuracy >= target_accuracy:
            #     print(f"Stopping early: Reached {accuracy:.2f}% accuracy")
            #     return True
    return False

if __name__ == "__main__":
    # Initialize wandb
    wandb.init(project="reverse_LRP_mnist", tags=["diff_lrp", "mnist", "simplernet"], mode="disabled")

    # Load and transform the MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Initialize the network and optimizer for the underlying network
    model = SimpleNet()
    # now wrap the network in the LRP class
    wrapped_model = WrapperNet(model)
    optimizer = optim.Adam(wrapped_model.parameters(), lr=1e-3)

    
    criterion = CosineDistanceLoss()

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapped_model.to(device)

    # Run training
    for epoch in range(1, 11):  # 10
        if train(wrapped_model, device, train_loader, optimizer, epoch):
            break
        
    wandb.finish()
