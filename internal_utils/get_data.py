import torch 
import torchvision.transforms as transforms
from torchvision.transforms import Normalize
from torchvision import datasets
import sys
ROOT_DIR = "/Users/charleshiggins/Personal/CharlesPhD/CodeRepo/xai_intervention/RL-LRP"
def get_CIFAR10_dataloader(train:bool = True):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=f'{ROOT_DIR}/baselines/trainVggBaselineForCIFAR10/data', train=train, download=False, transform=transforms.Compose([
                transforms.ToTensor()
            ])),
            batch_size=64, shuffle=False,
            num_workers=0, pin_memory=False)
    return loader