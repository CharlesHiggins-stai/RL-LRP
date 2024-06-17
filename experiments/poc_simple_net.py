import torch
import torch.nn as nn
import torch.nn.functional as F


class ManualCNN(nn.Module):
    def __init__(self):
        super(ManualCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)  # Input: (b, 1, 28, 28)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) # Input: (b, 10, 24, 24)
        self.fc1 = nn.Linear(20*20*20, 50)            # Input: (b, 8000)
        self.fc2 = nn.Linear(50, 10)                  # Input: (b, 50)
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, target_specific = None):
        # first layer and activation
        conv_0_out = self.conv1(x) # 0
        relu_0_out = self.relu(conv_0_out)  # 1
        # Output: (b, 10, 24, 24)
        # second layer and activation
        conv_1_out = self.conv2(relu_0_out) # 2
        relu_1_out = self.relu(conv_1_out) # 3
        # Output: (b, 20, 20, 20)
        # third layer and activation
        flat_relu_1_out = relu_1_out.view(-1, 20*20*20)    
        # Output: (b, 8000)
        fc1_out = self.fc1(flat_relu_1_out) # 4
        relu_2_out = self.relu(fc1_out) # 5 
        # relu layer 2  
        # Output: (b, 50)
        fc2_out = self.fc2(relu_2_out) # 6
        relu_3_out = self.relu(fc2_out) # 7          
        # Output: (b, 10)
        output_layer = relu_3_out
        # now apply a mask to select only the target class from the output layer
        if target_specific != None:
            mask = torch.zeros_like(output_layer)
            mask[torch.arange(mask.size(0)), target_specific.squeeze()] = 1  # Ensure target_class is either a scalar or has the same batch size as x
            # Apply the mask to propogate relenace forwards
            relevance = output_layer * mask
        else:
            # Softmax to create a differentiable mask
            # Adjust temperature to control the softness
            # assumes that the largest value is the target class --- desired output
            relevance = self.diff_softmax(output_layer, temperature=0.05)
            # relevance = output_layer
        # print(relevance[0])
        
        # R = self._reverse_layer_(relu_3_out, self.log_softmax, relevance) # 8
        R = self._reverse_layer_(fc2_out, self.relu, relevance) # 7
        R = self._reverse_layer_(relu_2_out, self.fc2, R) # 6
        R = self._reverse_layer_(fc1_out, self.relu, R) # 5
        R = self._reverse_layer_(flat_relu_1_out, self.fc1, R) # 4
        # now we need to reshape the R tensor to the shape of the relu_1_out tensor
        R = R.view(relu_1_out.shape)
        R = self._reverse_layer_(conv_1_out, self.relu, R) # 3
        R = self._reverse_layer_(relu_0_out, self.conv2, R) # 2
        R = self._reverse_layer_(conv_0_out, self.relu, R) # 1
        R = self._reverse_layer_(x, self.conv1, R) # 0
        return R
        
        
    def _reverse_layer_(self, activations_at_start:torch.Tensor, layer:torch.nn.Module, relevance:torch.Tensor):
        # print(f"reversing layer of type {type(layer)}")
        if isinstance(layer, torch.nn.Linear):
            R = self.lrp_linear(layer, activations_at_start, relevance)
        elif isinstance(layer, torch.nn.Conv2d):
            R = self.lrp_conv2d(layer, activations_at_start, relevance)
        elif isinstance(layer, torch.nn.ReLU):
            # print('encounterted a ReLu Layer')
            R = relevance * (activations_at_start > 0).float()
        elif isinstance(layer, torch.nn.LogSoftmax):
            R = self.reverse_log_softmax(layer, activations_at_start, relevance)
        else:
            print(f"Layer type {type(layer)} not supported")
        return R
    
    def lrp_linear(self, layer, activation, R, eps=1e-6):
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

    def lrp_conv2d(self, layer, activation, R, eps=1e-6):
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
    
    def reverse_log_softmax(self, layer, activation, R):
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
    
    def diff_softmax(self, input, temperature=1.0):
        soft_mask = F.softmax(input/temperature, dim=1)
        return (input * soft_mask)
