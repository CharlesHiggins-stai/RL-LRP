import torch
import torch.nn as nn
import torch.nn.functional as F
from .lrp_rules import reverse_layer, diff_softmax


class ManualCNN(nn.Module):
    def __init__(self, hybrid_loss = False):
        super(ManualCNN, self).__init__()
        self.hybrid_loss = hybrid_loss
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
            relevance = diff_softmax(output_layer, temperature=0.05)
            # relevance = output_layer
        # print(relevance[0])
        
        # R = self._reverse_layer_(relu_3_out, self.log_softmax, relevance) # 8
        R = reverse_layer(fc2_out, self.relu, relevance) # 7
        R = reverse_layer(relu_2_out, self.fc2, R) # 6
        R = reverse_layer(fc1_out, self.relu, R) # 5
        R = reverse_layer(flat_relu_1_out, self.fc1, R) # 4
        # now we need to reshape the R tensor to the shape of the relu_1_out tensor
        R = R.view(relu_1_out.shape)
        R = reverse_layer(conv_1_out, self.relu, R) # 3
        R = reverse_layer(relu_0_out, self.conv2, R) # 2
        R = reverse_layer(conv_0_out, self.relu, R) # 1
        R = reverse_layer(x, self.conv1, R) # 0
        if self.hybrid_loss == True:
            return diff_softmax(output_layer), R
        else: 
            return R
        