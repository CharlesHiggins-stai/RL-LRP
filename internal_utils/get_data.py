import torch 
import torchvision.transforms as transforms
from torchvision.transforms import Normalize
from torchvision import datasets
import sys
ROOT_DIR = "/home/charleshiggins/RL-LRP"
def get_CIFAR10_dataloader(train:bool = True, batch_size = 64, num_workers = 0, pin_memory = False):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=f'{ROOT_DIR}/baselines/trainVggBaselineForCIFAR10/data', train=train, download=False, transform=transforms.Compose([
                transforms.ToTensor()
            ])),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory)
    return loader