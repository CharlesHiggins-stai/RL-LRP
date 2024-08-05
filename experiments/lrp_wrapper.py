import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import torch
import torch.nn as nn
from contextlib import contextmanager
from .lrp_rules import reverse_layer, diff_softmax

@contextmanager
def track_activations(wrapper):
    original_relu = F.relu
    original_max_pool2d = F.max_pool2d
    original_log_softmax = F.log_softmax
    original_adaptive_avg_pool2d = F.adaptive_avg_pool2d
    
    def wrapped_relu(input, *args, **kwargs):
        output = original_relu(input, *args, **kwargs)
        wrapper.record_activation('ReLU', input, output)
        return output
    
    def wrapped_max_pool2d(input, stride, *args, **kwargs):
        output, indices = original_max_pool2d(input, stride, return_indices=True)
        wrapper.record_activation('MaxPool2d', input, output, extra={"indices": indices, "stride": stride})
        return output

    def wrapped_adaptive_avg_pool2d(input, output_size, *args, **kwargs):
        output = original_adaptive_avg_pool2d(input, output_size)
        wrapper.record_activation('AdaptiveAvgPool2d', input, output)
        return output
    
    def wrapped_log_softmax(input, *args, **kwargs):
        # don't do anything -- try to ignore the log_softmax layer
        # under the assumption this is the last layer
        # output = original_log_softmax(input, *args, **kwargs)
        # wrapper.record_activation('LogSoftmax', input, output)
        return input
    
    F.relu = wrapped_relu
    F.max_pool2d = wrapped_max_pool2d
    F.log_softmax = wrapped_log_softmax
    F.adaptive_avg_pool2d = wrapped_adaptive_avg_pool2d
    
    try:
        yield
    finally:
        F.relu = original_relu
        F.max_pool2d = original_max_pool2d
        F.log_softmax = original_log_softmax
        F.adaptive_avg_pool2d = original_adaptive_avg_pool2d


# Wrapper class to track layers and activations
class WrapperNet(nn.Module):
    def __init__(self, model:nn.Module, hybrid_loss:bool = False):
        super(WrapperNet, self).__init__()
        self.model = model
        self.hybrid_loss = hybrid_loss
        self.executed_layers = []
        self.activations_inputs = []
        self.activation_outputs = []
        self.info = []
        self.handles = []
        ######################################################################## 
        # Register hooks for the layers
        ######################################################################## 
        # need to add in a bit of logic to ensure layers aren't double counted
        # under the hood, some layers use functions which have been
        # overridden with the context manager. 
        # We discard these layers, and instead use the context manager for these
        # and instead we simply record all other layers
        ######################################################################## 
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Sequential) \
            and not isinstance(module, WrapperNet) \
            and not len(list(module.children())) > 0 \
                and type(module) not in [nn.ReLU, nn.MaxPool2d, nn.AdaptiveAvgPool2d, nn.LogSoftmax, nn.Dropout]:
                self.handles.append(module.register_forward_hook(self.forward_hook))
    
    def forward_hook(self, module, input, output):
        self.executed_layers.append((module.__class__.__name__, module))
        self.activations_inputs.append(input)
        self.activation_outputs.append(output)
        self.info.append({}) # placeholder for extra information
    
    def record_activation(self, name, input, output, extra = {}):
        self.executed_layers.append(name)
        self.activations_inputs.append(input)
        self.activation_outputs.append(output)
        self.info.append(extra)

    def forward(self, x, target_class = None):
        self.executed_layers = []
        self.activations_inputs = []
        self.activation_outputs = []
        self.info = []
        with track_activations(self):
            y =  self.model(x)
        if target_class != None:
            # relevance = y.gather(1, target_class.unsqueeze(1))
            mask = torch.zeros_like(y)
            mask.scatter_(1, target_class.unsqueeze(1), 1)
            relevance = y * mask
        else:
            # relevance = F.log_softmax(y) 
            # relevance = diff_softmax(y)
            relevance = y
        for index, layer in enumerate(zip(reversed(self.executed_layers), reversed(self.activations_inputs), reversed(self.activation_outputs), reversed(self.info))):
            # print('index:', index, '\tlayer:', layer[0])
            if index != 0 and relevance.shape != layer[2].shape:
            # if there is a reshaping/view operation, we need to reshape the relevance tensor
            # in the backwards pass to match the shape of the input tensor
                    relevance = relevance.view(layer[2].shape)
            if isinstance(layer[0], tuple):
            # If layer is layer or simply activation function, we need to treat differently 
                relevance = reverse_layer(layer[1][0], layer[0][1], relevance)
                # print(f'index: {index} layer:', layer[0][1])
            else:
                relevance = reverse_layer(layer[1], None, relevance, layer_type=layer[0], extra=layer[3])
                # print(f'index: {index} layer:', layer[0])
            # print('total relevance: ', relevance.flatten().sum().item())
            # print('relevance mean value:', relevance.flatten().mean().item())
            # print('relevance shape:', relevance.shape)
            if torch.isnan(relevance).any():
                print("Nan found here --- need to work out where this occured")
                nan_mask = torch.isnan(relevance)
                relevance = relevance.masked_fill(nan_mask, 0.0)
        if self.hybrid_loss == True:
            # return torch.nn.functional.log_softmax(y, dim=1), relevance
            return y, relevance
        else:
            return relevance

    def get_layers_and_activation_lists(self):
        return self.executed_layers, self. activation_inputs, self.activation_outputs
    
    def remove_hooks(self):
        # ensure that we delete hooks after every epoch to avoid unnecessary memory usage
        for handle in self.handles:
            handle.remove()
        self.handles = []
        
    def reapply_hooks(self):
        # reapply hooks after we have removed them --- make sure we still capture the layers output and activations
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Sequential) \
            and not isinstance(module, WrapperNet) \
            and not len(list(module.children())) > 0 \
                and type(module) not in [nn.ReLU, nn.MaxPool2d, nn.AdaptiveAvgPool2d, nn.LogSoftmax, nn.Dropout]:
                self.handles.append(module.register_forward_hook(self.forward_hook))
