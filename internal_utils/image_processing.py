import matplotlib.pyplot as plt
import torch 
import seaborn as sns

# Function to visualize an image from the DataLoader without transformations
def visualize_raw_image_from_dataloader(dataloader, index):
    # Access the dataset from the DataLoader
    dataset = dataloader.dataset

    # Get the original image and label (if available) without transformations
    if hasattr(dataset, 'data'):
        # For common datasets like CIFAR-10
        image = dataset.data[index]
        if dataset.targets:
            label = dataset.targets[index]
        else:
            label = None
    else:
        # For custom datasets, you may need to access the image directly
        image, label = dataset[index]
        if hasattr(image, 'permute'):
            image = image.permute(1, 2, 0).cpu().detach().numpy()

    # If the image is in range [0, 1], scale to [0, 255]
    if image.max() <= 1.0:
        image = (image * 255).astype('uint8')

    # Display the image
    plt.imshow(image)
    if label is not None:
        plt.title(f'Label: {label}')
    plt.axis('off')  # Optional: Remove axis
    plt.show()

# Example usage:
# dataloader = your DataLoader object
# visualize_raw_image_from_dataloader(dataloader, 5)  # Visualize the 6th image in the dataset

def plot_top_10_percent_heatmap(model_output_image, percentile_cutoff=0.75):
    """
    Plot the heatmap of the top 10% relevance scores from the model output image.
    Assume the image comes in in shape (C, H, W) where C is the number of channels 
    --- for single image only --- 
    """
    output_tensor = model_output_image.max(dim=0)[0]
    # Get the top 10% threshold
    top_10_percent_threshold = torch.quantile(output_tensor, percentile_cutoff)

    # Mask out values below the top 10% threshold
    mask = output_tensor >= top_10_percent_threshold

    # Apply the mask to the output tensor
    filtered_output = torch.zeros_like(output_tensor)
    filtered_output[mask] = output_tensor[mask]

    # Convert to numpy for plotting
    output_final = filtered_output.cpu().detach().numpy()

    # Plot the heatmap
    plt.figure(figsize=(10, 8))  # Set the size of the figure (optional)
    sns.heatmap(output_final, cmap='viridis', annot=False)
    plt.title(f'Top {1-percentile_cutoff}% Relevance Map')
    plt.show()
    

    
def filter_top_percent_pixels_over_channels(batch_tensor, percentile_cutoff):
    """
    Filters the top x percent of pixels in a batch of images by aggregating over channels.
    Arguments:
        batch_tensor: tensor of shape [b, c, h, w]
        percentile_cutoff: the top x percent threshold (e.g., 0.1 for top 10%)
    Returns:
        filtered_batch: tensor of the same shape as batch_tensor with only the top x percent pixels retained
    """
    b, c, h, w = batch_tensor.shape

    # Compute the maximum value across channels for each pixel
    max_over_channels = batch_tensor.max(dim=1)[0]

    # Flatten the spatial dimensions (h, w) to find the percentile across them
    flattened = max_over_channels.view(b, -1)

    # Calculate the threshold for the top x percent for each image in the batch
    threshold_values = torch.quantile(flattened, percentile_cutoff, dim=-1, keepdim=True)

    # Expand the threshold values to match the max_over_channels tensor shape
    thresholds = threshold_values.view(b, 1, 1).expand(-1, h, w)

    # Create a mask for values greater than or equal to the threshold
    mask = max_over_channels >= thresholds

    # Expand the mask to match the original tensor shape
    mask = mask.unsqueeze(1).expand(-1, c, -1, -1)

    # Apply the mask to retain only the top x percent values
    filtered_batch = torch.zeros_like(batch_tensor)
    filtered_batch[mask] = batch_tensor[mask]

    return filtered_batch