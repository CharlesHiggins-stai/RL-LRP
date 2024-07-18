import torch
import sys
sys.path.append('/Users/charleshiggins/Personal/CharlesPhD/CodeRepo/xai_intervention/RL-LRP')
from experiments import WrapperNet
from captum.attr import GuidedGradCam, LRP

def perform_lrp_plain(image, label, model):
    """Perform LRP on the image.

    Args:
        image (torch.Tensor): Tensor of images to be explained
        labels (torch.Tensor): labels of the image (i.e. the class)
        model (torch.nn.Module): model to be visualised
    Returns:
        torch.Tensor: heatmaps of the image
    """
    assert isinstance(model, WrapperNet), "Model must be a WrapperNet for LRP"
    class_idx, output = model(image, label)
    return output

def perform_loss_lrp(image, label, model):
    """Perform LRP on the image using the loss.

    Args:
        image (torch.Tensor): Tensor of images to be explained
        labels (torch.Tensor): labels of the image (i.e. the class)
        model (torch.nn.Module): model to be visualised
    Returns:
        torch.Tensor: heatmaps of the image
    """
    assert isinstance(model, WrapperNet), "Model must be a WrapperNet for LossLRP"
    class_idx, output = model(image, label)
    return output

def perform_lrp_captum(images, labels, model):
    """Perform LRP on the image using Captum.
    
    Args:
        image (torch.Tensor): Tensor of images to be explained
        labels (torch.Tensor): labels of the image (i.e. the class)
        model (torch.nn.Module): model to be visualised
    Returns:
        torch.Tensor: heatmaps of the image
    """
    lrp = LRP(model)
     # Example target class
    attributions = lrp.attribute(images, target=labels)
    
    return attributions
    

def get_input_output_layers(model):
    """
    Gets the first and last convolutional layers of the model for GradCam
    
    Args:
    - model: The PyTorch model
    
    Returns:
    - input_layer: The first convolutional layer
    - output_layer: The last convolutional layer
    """
    layers = list(model.modules())
    conv_layers = [layer for layer in layers if isinstance(layer, torch.nn.Conv2d)]
    
    if not conv_layers:
        raise ValueError("The model does not contain any Conv2d layers.")
    
    input_layer = conv_layers[0]
    output_layer = conv_layers[-1]
    
    return input_layer, output_layer

def perform_gradcam(images, labels, model):
    """Perform GradCAM on the image.

    Args:
        image (torch.Tensor): Tensor of images to be explained
        labels (torch.Tensor): labels of the image (i.e. the class)
        model (torch.nn.Module): model to be visualised
    Returns:
        torch.Tensor: heatmaps of the image
    """
    # Ensure the model is in evaluation mode
    model.eval()
    
    # Get the input and output layers
    input_layer, output_layer = get_input_output_layers(model)
    
    # Create a LayerGradCam object
    layer_gc = GuidedGradCam(model, output_layer)
    
    # Compute GradCAM attributions
    attributions = layer_gc.attribute(images, target=labels)
    
    return attributions