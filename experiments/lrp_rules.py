import torch
import torch.nn as nn
import torch.nn.functional as F

def reverse_layer(activations_at_start:torch.Tensor, layer:torch.nn.Module, relevance:torch.Tensor, layer_type:str=None, extra=None):
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
    elif isinstance(layer, torch.nn.MaxPool2d) or layer_type == 'MaxPool2d':
        R = reverse_max_pool2d(relevance, extra["indices"], activations_at_start, extra["stride"])
    elif isinstance(layer, torch.nn.Dropout or layer_type == 'Dropout'):
        # don't bother reversing --- assume that dropout has no effect...
        # cannot perform gradient op, and 
        R = relevance
    elif isinstance(layer, torch.nn.AdaptiveAvgPool2d) or layer_type == 'AdaptiveAvgPool2d':
        R = reverse_adaptive_avg_pool2d(relevance, activations_at_start)
    else:
        raise ValueError(f"Layer type {type(layer)} not supported")
        
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
    S = R / (Z + eps)
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
    Z = F.conv2d(X, W, bias=layer.bias, stride=layer.stride, padding=layer.padding) 
    S = torch.divide(R,(Z + eps))
    C = F.conv_transpose2d(S, W, stride=layer.stride, padding=layer.padding)
    R_new = X * C
    R_new_sum = R_new.sum(dim=[1, 2, 3], keepdim=True)
    R_sum = R.sum(dim=[1, 2, 3], keepdim=True)
    R_new = R_new * (R_sum / (R_new_sum + eps))
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

def diff_softmax(input, temperature=1):
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

# def reverse_max_pool2d(relevance, indices, input_shape):
#     # Create an empty tensor with the same shape as the input
#     reversed_input = torch.zeros_like(input_shape)
    
#     # Flatten the output and indices for easier manipulation
#     relevance_flat = relevance.view(relevance.size(0), -1)
#     indices_flat = indices.view(indices.size(0), -1)
    
#     # Scatter the relevance scores from the output back to the input positions
#     reversed_input.view(reversed_input.size(0), -1).scatter_(1, indices_flat, relevance_flat)
    
#     return reversed_input

def reverse_max_pool2d(relevance, indices, input_shape, kernel_size, stride=None, padding=0):
    # Calculate the appropriate output size
    if stride is None:
        stride = kernel_size
    output_size = (input_shape.size(2), input_shape.size(3))
    
    # Use max_unpool2d to propagate relevance scores back to the input positions
    reversed_input = F.max_unpool2d(relevance, indices, kernel_size=kernel_size, stride=stride, padding=padding, output_size=output_size)
    
    return reversed_input

def reverse_adaptive_avg_pool2d(relevance, input_activation, eta = 1e-9):
    """
    Make sure to use a differentiable interpolation method if gradients need to pass through this function.
    """
    output_shape = relevance.shape
    # Calculate stride and kernel size
    stride_height = input_activation.shape[2] // output_shape[2]
    stride_width = input_activation.shape[3] // output_shape[3]
    kernel_height = input_activation.shape[2] - (output_shape[2] - 1) * stride_height
    kernel_width = input_activation.shape[3] - (output_shape[3] - 1) * stride_width
    
    # Differentiable interpolation
    upsampled_relevance = F.interpolate(relevance, size=(input_activation.shape[2], input_activation.shape[3]), mode='bilinear', align_corners=True)

    # Continue with padding and average pooling
    input_activation_padded = F.pad(input_activation, (0, kernel_width - 1, 0, kernel_height - 1))
    Z = F.avg_pool2d(input_activation_padded, kernel_size=(kernel_height, kernel_width), stride=(stride_height, stride_width))
    Z_upsampled = F.interpolate(Z, size=(input_activation.shape[2], input_activation.shape[3]), mode='bilinear', align_corners=True)
    
    return upsampled_relevance * (input_activation / (Z_upsampled + eta))

