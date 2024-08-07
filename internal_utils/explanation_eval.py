import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import torch

def transform_batch_of_images(images):
    """Apply standard transformation to the batch of images."""
    # normalise the image to be in the right range
    normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    # convert image to tensor
    to_tensor_transform = transforms.ToTensor()
    return normalize_transform(to_tensor_transform(images))

def get_data_imagenette(path = "/Users/charleshiggins/Personal/CharlesPhD/CodeRepo/xai_intervention/RL-LRP/data/Imagenette", batch_size = 8, shuffle = False):
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    
    imagenette_train = datasets.Imagenette(
        root=path,  # Specify the directory to store the dataset
        split='train',  # Use the validation split
        transform=transform,
        download=False  # Download the dataset if not already present
    )

    # Load Imagenette dataset
    imagenette_val = datasets.Imagenette(
        root=path,  # Specify the directory to store the dataset
        split='val',  # Use the validation split
        transform=transform,
        download=False  # Download the dataset if not already present
    )
    

    train_loader = DataLoader(
        imagenette_train,
        batch_size=batch_size,
        shuffle=shuffle,
        persistent_workers=True,
        multiprocessing_context="forkserver",
        num_workers=4, 
        pin_memory=True
    )

    # Create a DataLoader
    val_loader = DataLoader(
        imagenette_val,
        batch_size=batch_size,
        shuffle=shuffle,
        persistent_workers=True,
        multiprocessing_context="forkserver",
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader



def get_data(path_to_data:str = '/home/charleshiggins/RL-LRP/baselines/trainVggBaselineForCIFAR10/data'):
    """Get Dataloader objects from Cifar10 dataset, with path passed in.

    Args:
        path_to_data (str): path to the data directory
    N.B. Changed to Imagenette dataset from CIFAR210
    Alternative here
    datasets.CIFAR10(root=path_to_data, train=False, 
        batch_size=64, shuffle=False, num_workers=2, 
        pin_memory=True, transforms=transforms.ToTensor()
    )
    """
    val_loader = datasets.CIFAR10(root=path_to_data, train=False, 
        batch_size=64, shuffle=False, num_workers=2, 
        pin_memory=True, transforms=transforms.ToTensor()
    )
    return val_loader

def blur_image_batch(images, kernel_size):
    """Blur the batch of images using a Gaussian kernel.

    Args:
        image (torch.Tensor): batch of images to be blurred
        kernel_size (int): size of the Gaussian kernel
    Returns:
        torch.Tensor: blurred images
    """
    
    blurred_images = torch.stack([TF.gaussian_blur(img, kernel_size=[kernel_size, kernel_size]) for img in images])
    return blurred_images

def add_random_noise_batch(images, noise_level):
    """Add random noise to the batch of images.

    Args:
        images (torch.Tensor): images to have noise added
        noise_level (float): level of noise to be added
    Returns:
        torch.Tensor: images with noise added
    """
    noise = torch.randn_like(images) * noise_level
    noisy_images = images + noise
    return noisy_images

def compute_distance_between_images(images1, images2):
    """Compute the distance between two batches of images.

    Args:
        image1 (torch.Tensor): Tensor of treated images
        image2 (torch.Tensor): Tensor of ground-truth images
    Returns:
        torch.Tensor: Tensor of distances between the two images
    """
    if images1.shape[0] == 0 or images2.shape[0] == 0:
        return None
    # condense images to heatmap
    images1 = condense_to_heatmap(images1)
    images2 = condense_to_heatmap(images2)
    # Flatten the images to compute cosine similarity
    images1_flat = images1.view(images1.size(0), -1)
    images2_flat = images2.view(images2.size(0), -1)
    
    # Compute cosine similarity and convert to cosine distance
    cosine_similarity = F.cosine_similarity(images1_flat, images2_flat)
    cosine_distance = 1 - cosine_similarity  # Convert similarity to distance
    return cosine_distance


def condense_to_heatmap(images):
    """
    Condense a batch of images to a heatmap by taking the maximum activation across channels.
    
    Args:
        images (torch.Tensor): A batch of images with dimensions (batch_size, channels, height, width).
    
    Returns:
        torch.Tensor: A tensor of heatmaps with dimensions (batch_size, height, width).
    """
    # Use torch.max to find the maximum across the channels (dim=1)
    # max function returns values and indices, so we select values using [0]
    if images.dim() == 2:
        # already in heatmap form
        return images
    assert images.dim() == 4, "Input tensor must have 4 dimensions --- assumes a batch of inputs"
    # make sure total sum of outputs is 1
    
    # Normalize each image in the batch independently
    total_sum_per_image = torch.sum(images, dim=(1, 2, 3), keepdim=True)
    normalized_tensor = images / total_sum_per_image
    # print(f"normalized_tensor shape: {normalized_tensor.shape}")
    # print(f"normalized_tensor sum: {torch.sum(normalized_tensor)}")
    # Create heatmap by taking the maximum value across the channel dimension
    heatmaps = torch.max(normalized_tensor, dim=1).values

    return heatmaps

def compute_sparseness_of_heatmap(input_images):
    """Compute the sparseness of the heatmap.

    Args:
        heatmap (torch.Tensor): heatmap to be computed
    Returns:
        float: sparseness of the heatmap
    """
    heatmaps = condense_to_heatmap(input_images)
    threshold = 0.01  # Define near-zero threshold
    near_zero = (heatmaps.abs() < threshold).float()
    sparseness = near_zero.mean(dim=[1, 2])  # Compute mean across spatial dimensions

    # Compute Gini coefficient for each heatmap in the batch
    batch_size, height, width = heatmaps.size()
    gini_indices = torch.empty(batch_size)
    
    for i in range(batch_size):
        values = heatmaps[i].view(-1)
        sorted_values, _ = torch.sort(values)
        n = len(values)
        cumvals = torch.cumsum(sorted_values, dim=0)
        sum_values = cumvals[-1]
        gini_index = (2 * torch.arange(1, n+1).to(heatmaps.device) * sorted_values).sum() / (n * sum_values) - (n + 1) / n
        gini_indices[i] = 1 - gini_index

    return sparseness, gini_indices


def preprocess_images(image_batch):
    """Preprocess the image.

    Args:
        image (torch.Tensor): image to be preprocessed
    Returns:
        torch.Tensor: preprocessed image
    """
    if isinstance(image_batch, torch.Tensor) and image_batch.dim() == 4:
        # normalise the images
        normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        images = normalize_transform(image_batch)
        return images
    else:
        print("something went wrong in preprocessing images")
        print(f"image batch is of shape {image_batch.shape}")
        raise ValueError(f"Input must be a tensor of images -- unknown format {type(image_batch)} and dimension {image_batch.dim()}")
    
    
    
def get_learner_model():
    """Get the learner model."""
    pass

def get_teacher_model():
    """ Load and return a pretrained VGG16 model from TorchVision"""
    # Load the pretrained VGG16 model
    model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
    # model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Set the model to evaluation mode
    model.eval()
    
    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model