import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from .lrp_rules import reverse_layer, diff_softmax

@contextmanager
def track_activations(wrapper):
    original_relu = F.relu
    original_max_pool2d = F.max_pool2d
    original_log_softmax = F.log_softmax
    
    def wrapped_relu(input, *args, **kwargs):
        output = original_relu(input, *args, **kwargs)
        wrapper.record_activation('ReLU', input, output)
        return output
    
    def wrapped_max_pool2d(input, *args, **kwargs):
        output = original_max_pool2d(input, *args, **kwargs)
        wrapper.record_activation('MaxPool2d', input,output)
        return output
    
    def wrapped_log_softmax(input, *args, **kwargs):
        output = original_log_softmax(input, *args, **kwargs)
        wrapper.record_activation('LogSoftmax', input, output)
        return output
    
    F.relu = wrapped_relu
    F.max_pool2d = wrapped_max_pool2d
    F.log_softmax = wrapped_log_softmax
    
    try:
        yield
    finally:
        F.relu = original_relu
        F.max_pool2d = original_max_pool2d
        F.log_softmax = original_log_softmax


# Wrapper class to track layers and activations
class WrapperNet(nn.Module):
    def __init__(self, model):
        super(WrapperNet, self).__init__()
        self.model = model
        self.executed_layers = []
        self.activations_inputs = []
        self.activation_outputs = []
        
        # Register hooks for the layers
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Sequential) and not isinstance(module, WrapperNet) and not len(list(module.children())) > 0:
                module.register_forward_hook(self.forward_hook)
    
    def forward_hook(self, module, input, output):
        self.executed_layers.append((module.__class__.__name__, module))
        self.activations_inputs.append(input)
        self.activation_outputs.append(output)
    
    def record_activation(self, name, input, output):
        self.executed_layers.append(name)
        self.activations_inputs.append(input)
        self.activation_outputs.append(output)

    def forward(self, x):
        self.executed_layers = []
        self.activations_inputs = []
        self.activation_outputs = []
        with track_activations(self):
            y =  self.model(x)
        relevance = diff_softmax(y)
        for index, layer in enumerate(zip(reversed(self.executed_layers), reversed(self.activations_inputs), reversed(self.activation_outputs))):
            if relevance.shape != layer[2].shape:
                    relevance = relevance.view(layer[2].shape)
            if isinstance(layer[0], tuple):
                # if there is a reshaping operation, we need to reshape the relevance tensor
                relevance = reverse_layer(layer[1][0], layer[0][1], relevance)
            else:
                relevance = reverse_layer(layer[1], None, relevance, layer_type=layer[0])
        return relevance

    def get_layers_and_activation_lists(self):
        return self.executed_layers, self. activation_inputs, self.activation_outputs
