import torch
import torch.nn as nn
import torch.nn.functional as F

def reverse_layer(activations_at_start:torch.Tensor, layer:torch.nn.Module, relevance:torch.Tensor, layer_type:str=None):
    # print(f"reversing layer of type {layer_type}")
    if isinstance(layer, torch.nn.Linear):
        R = lrp_linear(layer, activations_at_start, relevance)
    elif isinstance(layer, torch.nn.Conv2d):
        R = lrp_conv2d(layer, activations_at_start, relevance)
    elif isinstance(layer, torch.nn.ReLU) or layer_type == 'ReLU':
        # print('encounterted a ReLu Layer')
        R = relevance * (activations_at_start > 0).float()
    elif isinstance(layer, torch.nn.LogSoftmax) or layer_type == 'LogSoftmax':
        R = reverse_log_softmax(activations_at_start, relevance)
    else:
        print(f"Layer type {type(layer)} not supported")
    return R

def lrp_linear(layer, activation, R, eps=1e-6):
    """
    LRP for a linear layer.
    Arguments:
        layer: the linear layer (nn.Linear)
        R: relevance scores from the previous layer (Tensor)
        eps: small value to avoid division by zero (float)
    Returns:
        relevance scores for the input of this layer (Tensor)
    """
    W = layer.weight
    # Z = W @ activation.t() + layer.bias[:, None] + eps
    Z = layer.forward(activation)
    S = R / Z
    C = W.t() @ S.t()
    R_new = activation * C.t()
    return R_new

def lrp_conv2d(layer, activation, R, eps=1e-6):
    """
    LRP for a convolutional layer.
    Arguments:
        layer: the convolutional layer (nn.Conv2d)
        R: relevance scores from the previous layer (Tensor)
        eps: small value to avoid division by zero (float)
    Returns:
        relevance scores for the input of this layer (Tensor)
    """
    W = layer.weight
    X = activation
    Z = F.conv2d(X, W, bias=layer.bias, stride=layer.stride, padding=layer.padding) + eps
    S = R / Z
    C = F.conv_transpose2d(S, W, stride=layer.stride, padding=layer.padding)
    R_new = X * C
    return R_new

def reverse_log_softmax(activation, R):
    """
    Reverse the log_softmax operation.
    Arguments:
        layer: the log_softmax layer (nn.LogSoftmax)
        activation: the input activation to the log_softmax layer (Tensor)
        R: relevance scores from the previous layer (Tensor)
    Returns:
        relevance scores for the input of this layer (Tensor)
    """
    # Assuming activation are log probabilities
    probs = torch.exp(activation)
    probs_sum = probs.sum(dim=1, keepdim=True)
    return R * probs / probs_sum

def diff_softmax(input, temperature=1.0):
    """
    A function to generate a differential alternative to the softmax function for the arg-max operation
    It's a smooth function, which essentially focuses on the lagest value (i.e. the output of the softmax function)

    Args:
        input (torch.Tensor): Input to be transformed.
        temperature (float, optional): Lower values make softmax harder/more explicit. Defaults to 1.0.

    Returns:
        torch.Tensor: A soft-softmax transformed tensor.
    """
    soft_mask = F.softmax(input/temperature, dim=1)
    return (input * soft_mask)
