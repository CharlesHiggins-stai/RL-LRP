import torch 
import torchvision.transforms as transforms
from torchvision.transforms import Normalize
from torchvision import datasets
import torchvision.models as models
import sys
ROOT_DIR = "/Users/charleshiggins/Personal/CharlesPhD/CodeRepo/xai_intervention/RL-LRP"

def get_CIFAR10_dataloader(train:bool = True, batch_size = 64, num_workers = 0, pin_memory = False):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=f'{ROOT_DIR}/baselines/trainVggBaselineForCIFAR10/data', train=train, download=False, transform=transforms.Compose([
                transforms.ToTensor(), 
                normalize,
            ])),
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=True,
            multiprocessing_context="forkserver",
            pin_memory=pin_memory)
    return loader

def imagenette_to_imagenet_label_mapping(imagenette_labels):
    mapping = {
        0: 0,    # tench
        1: 217,  # English Springer
        2: 482,  # Cassette Player
        3: 491,  # Chain Saw
        4: 497,  # Church
        5: 566,  # French Horn
        6: 569,  # Garbage Truck
        7: 571,  # Gas Pump
        8: 574,  # Golf Ball
        9: 701   # Parachute
    }
    
    # Assuming imagenette_labels is a list
    output = [mapping[label.item()] for label in imagenette_labels]
    return output

def imagenette_to_imagenet_label_mapping_fast(imagenette_labels):
    # a vectorised version of the mapping above
    mapping = torch.tensor([0, 217, 482, 491, 497, 566, 569, 571, 574, 701])
    
    # Use the imagenette_labels as indices to map to the corresponding imagenet labels
    output = mapping[imagenette_labels]
    return output

def get_vgg16():
    """Download a pretrained vgg16 on Imagenet"""
    vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    vgg16.eval()
    return vgg16
    
def get_vgg19():
    """ Download a pretrained vgg19 on Imagenet"""
    vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    vgg19.eval() 
    return vgg19
    