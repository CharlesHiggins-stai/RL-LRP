import torch
import torch.nn as nn
import torch.nn.functional as F

def reverse_layer(activations_at_start:torch.Tensor, layer:torch.nn.Module, relevance:torch.Tensor, layer_type:str=None, extra=None):
    if isinstance(layer, torch.nn.Linear):
        R = lrp_linear_alpha_beta(layer, activations_at_start, relevance)
    elif isinstance(layer, torch.nn.Conv2d):
        R = lrp_conv2d_alpha_beta(layer, activations_at_start, relevance)
    elif isinstance(layer, torch.nn.ReLU) or layer_type == 'ReLU':
        R = relevance * (activations_at_start > 0).float()
    elif isinstance(layer, torch.nn.LogSoftmax) or layer_type == 'LogSoftmax':
        R = reverse_log_softmax(activations_at_start, relevance)
    elif isinstance(layer, torch.nn.MaxPool2d) or layer_type == 'MaxPool2d':
        R = reverse_max_pool2d(relevance, extra["indices"], activations_at_start, extra["stride"])
    elif isinstance(layer, torch.nn.BatchNorm2d) or layer_type == 'BatchNorm2d':
        R = reverse_batch_norm2d(layer, activations_at_start, relevance)
    elif isinstance(layer, torch.nn.Dropout or layer_type == 'Dropout'):
        # don't bother reversing --- assume that dropout has no effect...
        # cannot perform gradient op, and 
        R = relevance
    elif isinstance(layer, torch.nn.AdaptiveAvgPool2d) or layer_type == 'AdaptiveAvgPool2d':
        R = reverse_adaptive_avg_pool2d(relevance, activations_at_start)
    else:
        raise ValueError(f"Layer type {type(layer)} not supported")
        
    return R

def lrp_linear(layer, activation, R, eps=1e-2):
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

def lrp_linear_alpha_beta(layer, activation, R, alpha=1, beta=1, eps=1e-2):
    """
    LRP-αβ for a linear layer.
    Arguments:
        layer: the linear layer (nn.Linear)
        R: relevance scores from the previous layer (Tensor)
        alpha: coefficient for positive contributions
        beta: coefficient for negative contributions
        eps: small value to avoid division by zero (float)
    Returns:
        relevance scores for the input of this layer (Tensor)
    """
    W = layer.weight
    bias = layer.bias[:, None] if layer.bias is not None else 0
    bias_plus = torch.clamp(bias, min=0)
    bias_minus = torch.clamp(bias, max=0)

    # Separate positive and negative contributions
    W_plus = torch.clamp(W, min=0)
    W_minus = torch.clamp(W, max=0)
    
    activation_pos = torch.clamp(activation, min=0)
    activation_neg = torch.clamp(activation, max=0)
    
    # perform the positive and negative values for the foeward pass
    Z_plus_plus = ((W_plus @ activation_pos.t()) + bias_plus).t()
    Z_minus_minus = ((W_minus @ activation_neg.t()) + bias_plus).t()
    
    Z_plus_minus = ((W_minus @ activation_pos.t()) + bias_minus).t()
    Z_minus_plus = ((W_plus @ activation_neg.t()) + bias_minus).t()
    
    Z_plus = Z_plus_plus + Z_minus_minus
    Z_minus = Z_minus_plus + Z_plus_minus

    # Calculate contributions for R
    S_plus = safe_divide(R, Z_plus, eps)
    S_minus = safe_divide(R, Z_minus, eps)
    
    # Propagate the relevance scores back
    
    C_plus_plus = W_plus.t() @ S_plus.t()
    C_minus_minus = W_minus.t() @ S_plus.t()
    
    C_plus_minus = W_plus.t() @ S_minus.t()
    C_minus_plus = W_minus.t() @ S_minus.t()
    
    # Combine contributions to get the final relevance scores
    R_new_plus = (activation_pos * C_plus_plus.t()) + (activation_neg * C_minus_minus.t())
    R_new_minus = (activation_neg * C_plus_minus.t()) + (activation_pos + C_minus_plus.t())
    
    # compute alpha and beta values now
    R_new = alpha * R_new_plus + beta * R_new_minus
    if torch.isnan(R_new).any():
        print('Nan crept in here --- time to debug this mother fucker')
    # Normalise the relevance scores to match the input
    # R_new_sum = R_new.sum(dim=1, keepdim=True)
    # R_sum = R.sum(dim=1, keepdim=True)
    # R_new = R_new * (R_sum / (R_new_sum + eps))
    return R_new


def lrp_conv2d(layer, activation, R, eps=1e-2):
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
    S = R / (Z + eps)
    C = F.conv_transpose2d(S, W, stride=layer.stride, padding=layer.padding)
    R_new = X * C
    R_new_sum = R_new.sum(dim=[1, 2, 3], keepdim=True)
    R_sum = R.sum(dim=[1, 2, 3], keepdim=True)
    R_new = R_new * (R_sum / (R_new_sum + eps))
    return R_new

def lrp_conv2d_alpha_beta(layer, activation, R, alpha=1, beta=0, eps=1e-2):
    """
    LRP-αβ for a convolutional layer.
    Arguments:
        layer: the convolutional layer (nn.Conv2d)
        activation: activation maps from the previous layer (Tensor)
        R: relevance scores from the previous layer (Tensor)
        alpha: coefficient for positive contributions
        beta: coefficient for negative contributions
        eps: small value to avoid division by zero (float)
    Returns:
        relevance scores for the input of this layer (Tensor)
    """
    W = layer.weight
    bias = layer.bias if layer.bias is not None else 0
    # bias = None

    # Compute positive and negative parts of the weight
    W_plus = torch.clamp(W, min=0)
    W_minus = torch.clamp(W, max=0)
    # Compute positive and negeative parts of the activation
    activation_pos = torch.clamp(activation, min=0)
    activation_neg = torch.clamp(activation, max=0)

    bias_pos = torch.clamp(bias, min=0)
    bias_neg = torch.clamp(bias, max=0)
    # Forward pass for positive and negative contributions
    Z_plus_plus = F.conv2d(activation_pos, W_plus, bias=bias_pos, stride=layer.stride, padding=layer.padding) 
    Z_minus_minus = F.conv2d(activation_neg, W_minus, bias= bias_pos, stride=layer.stride, padding=layer.padding) 
    Z_plus_minus = F.conv2d(activation_neg, W_plus, bias= bias_neg, stride=layer.stride, padding=layer.padding) 
    Z_minus_plus = F.conv2d(activation_pos, W_minus, bias= bias_neg, stride=layer.stride, padding=layer.padding)

    Z_plus = Z_plus_plus + Z_minus_minus
    Z_minus = Z_minus_plus + Z_plus_minus

    # Calculate the positive and negative contributions
    S_plus = safe_divide(R, Z_plus, eps)
    S_minus = safe_divide(R, Z_minus, eps)

    # Propagate the relevance scores back
    # S = (S_plus * alpha) + (S_minus * beta)
    # C = F.conv_transpose2d(S, W, stride=layer.stride, padding=layer.padding)
    C_plus_plus = F.conv_transpose2d(S_plus, W_plus, stride=layer.stride, padding=layer.padding)
    C_minus_minus = F.conv_transpose2d(S_plus, W_minus, stride=layer.stride, padding=layer.padding)
    
    C_plus_minus = F.conv_transpose2d(S_minus, W_plus, stride=layer.stride, padding=layer.padding)
    C_minus_plus = F.conv_transpose2d(S_minus, W_minus, stride=layer.stride, padding=layer.padding)
    
    
    # Combine contributions to get the final relevance scores
    if activation_pos.shape != C_plus_plus.shape:
        print('shapes are fucked, so adding extra padding')
    
    R_new_plus = (activation_pos * C_plus_plus) + (activation_neg * C_minus_minus)
    R_new_minus = (activation_neg * C_plus_minus) + (activation_pos + C_minus_plus)
    
    R_new = alpha * R_new_plus + beta * R_new_minus
    
    if torch.isnan(R_new).any():
        print('Nan crept in here --- time to debug this mother fucker')
    # Normalize the relevance scores if necessar
    # R_new_sum = R_new.sum(dim=[1, 2, 3], keepdim=True)
    # R_sum = R.sum(dim=[1, 2, 3], keepdim=True)
    # R_new = R_new * (R_sum / (R_new_sum + eps))

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

def reverse_batch_norm2d(layer, activation, relevance, eps=1e-2):
    """Reverse the batch normalisation operation.

    Args:
        relevance (torch.Tensor): Relevance scores from the previous layer.
        layer (torch.Tensor): Layer
        activation (torch.Tensor): activation incoming to the layer
        eps (error, optional): Prevents div by zero errors. Defaults to 1e-2.
    """
    # Get parameters of the batch normalization layer
    weight = layer.weight
    bias = layer.bias
    mean = layer.running_mean
    var = layer.running_var
    eps = layer.eps

    # Calculate the normalized input
    normalized_input = (activation - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + eps)

    # Calculate the relevance score for each feature map
    weight_expanded = weight[None, :, None, None]
    relevance_input = relevance * weight_expanded
    bias_expanded = bias[None, :, None, None]

    # Normalize the relevance
    relevance_sum = relevance_input.sum(dim=(2, 3), keepdim=True) + eps
    relevance_normed = relevance_input / relevance_sum

    # Propagate the relevance through the layer
    r_new = relevance_normed * normalized_input + bias_expanded

    return r_new
    

# def safe_divide(numerator, denominator, eps):
#     # Where denominator is not zero, perform the division, otherwise, return zero
#     return torch.where(denominator != 0, numerator / denominator + eps , torch.zeros_like(numerator))

def safe_divide(numerator, denominator, eps):
    # Where denominator is not zero, perform the division, otherwise, return zero
    safe_denom = denominator + eps * torch.sign(denominator).detach() + eps * 0.1
    return torch.div(numerator, safe_denom)
