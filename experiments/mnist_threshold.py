import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def load_mnist(batch_size=64):
    """ Load MNIST dataset with torchvision. """
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to PyTorch tensors
        transforms.Normalize((0.5,), (0.5,))  # Normalizes the dataset
    ])
    
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    return train_loader

def apply_threshold(images, threshold=0.9):
    """ Apply a threshold to the images, setting all pixels below the threshold to zero.
        Images should retain original dimensions.
    """
    # Thresholding
    thresholded_images = torch.where(images > threshold, images, torch.zeros_like(images))
    sum_of_pixels = thresholded_images.sum(dim=[2, 3], keepdim=True)
    # Normalize such that each image's pixels sum to 1
    thresholded_images = thresholded_images / sum_of_pixels
    return thresholded_images

if __name__ == "__main__":
    # Load data
    train_loader = load_mnist(batch_size=10)

    # Get a single batch of images
    data_iter = iter(train_loader)
    images, labels = next(data_iter)

    # Apply threshold
    thresholded_images = apply_threshold(images, threshold=0.2)  # Using 0.5 as an example threshold
    

    # Plotting
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(images[0][0], cmap='gray')
    axes[0].set_title('Original Image')
    axes[1].imshow(thresholded_images[0][0], cmap='gray')
    axes[1].set_title('Thresholded Image')
    plt.show()
