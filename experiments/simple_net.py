import torch
import torch.nn as nn


# Define the neural network architecture
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Convolution 1
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 2)(x)
        
        # Convolution 2
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 2)(x)

        # Flatten the output for the dense layer
        x = x.view(-1, 64 * 7 * 7)
        
        # Dense layer
        x = self.fc1(x)
        x = nn.ReLU()(x)
        
        # Output layer
        x = self.fc2(x)
        return x

class SimpleRNet(nn.Module):
    def __init__(self):
        super(SimpleRNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(8000, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )

    def forward(self, x):
        x = self.features(x)
        # print("this is the shape that needs to be flattened")
        # print(x.shape)
        x = x.view(-1, 8000)  # Flatten the output for the classifier
        x = self.classifier(x)
        x = torch.nn.functional.log_softmax(x, dim=1)

        return x
