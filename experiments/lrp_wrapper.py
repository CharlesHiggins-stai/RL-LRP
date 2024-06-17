import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class DiffLrpWrapper(nn.Module):
    def __init__(self, net):
        super().__init__()
        assert isinstance(net, nn.Module), f"Expected net to be an instance of nn.Module, got {type(net)}"
        self.net = net
        self.activations = {}
        self.outputs = {}
        self._register_hooks()

    def _register_hooks(self):
        # Register a forward hook on each module
        for name, module in self.net.named_modules():
            # Avoid registering hooks on containers
            if len(list(module.children())) == 0:
                module.register_forward_hook(self._save_activation(name))

    def _save_activation(self, name):
        # This method returns a hook function
        def hook(module, input, output):
            # saving activation input so that we can calculate the relevance later...
            self.activations[name] = input[0]
            self.outputs[name] = output.detach()
        return hook

    def forward(self, x, target_class:torch.Tensor= None):
        if target_class != None: 
            assert x.shape[0] == target_class.shape[0], f"Expected x and target_class to have the same batch size, got {x.shape[0]} and {target_class.shape[0]}"
        # Forward pass through the network
        initial_out = self.net(x)
        # Create a mask that zeros out all elements except for the target class
        if target_class != None:
            mask = torch.zeros_like(initial_out)
            mask[torch.arange(mask.size(0)), target_class.squeeze()] = 1  # Ensure target_class is either a scalar or has the same batch size as x
            # Apply the mask to propogate relenace forwards
            relevance = initial_out * mask
        else:
            relevance = initial_out
        # loop backwards from output to input layer
        for name, actual_module in list(self.net.named_modules())[::-1]:
            if len(list(actual_module.children())) == 0:
                # if the module is a leaf module, apply LRP
                relevance = self._apply_lrp(name, actual_module, relevance.detach())
        # normalise output so that it sums to 1 in total --- all outputs are also in same range
        # sum_of_pixels = relevance.sum(dim=[2, 3], keepdim=True)
        # relevance = relevance / sum_of_pixels
        # print(f"size of hooks dict: {len(self.activations)}")
        return relevance
    

    def _apply_lrp(self, name:str, layer:torch.nn.Module, relevance_to_be_propagaed:torch.Tensor):
        # Get the activation of the module
        layer_activation_values = self.activations[name]
        # check datatypes coming through
        assert isinstance(layer_activation_values, torch.Tensor)
        assert isinstance(layer, nn.Module)
        assert isinstance(relevance_to_be_propagaed, torch.Tensor)
        # Check that the shape of the layer outputs and relevance are the same
        if not self.outputs[name].shape == relevance_to_be_propagaed.shape:
            # print(f"shapes didn't match for layer {name}")
            relevance_to_be_propagaed = relevance_to_be_propagaed.view(self.outputs[name].shape)
        # Get the relevance of the output & apply LRP
        relevance = self._reverse_layer_(layer_activation_values, layer, relevance_to_be_propagaed)
        return relevance
    
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
        
    def get_activations(self):
        return self.activations
    
    def DEPRECIATED_reverse_layer_(self, activations_at_start:torch.Tensor, actual_module:torch.nn.Module, relevance:torch.Tensor, epsilon=1e-9):
        # make sure corret data is coming in
        assert isinstance(activations_at_start, torch.Tensor), f"Expected activations_at_start to be a torch.Tensor, got {type(activations_at_start)}"
        assert isinstance(actual_module, nn.Module), f"Expected actual_module to be an nn.Module, got {type(actual_module)}"
        assert isinstance(relevance, torch.Tensor), f"Expected relevance to be a torch.Tensor, got {type(relevance)}"
        # print(f"activations_at_start shape: {activations_at_start.shape}")
        activations_at_start.requires_grad_()
        activations_at_start.retain_grad()
        # perform a modified forward pass (alpha beta rule applied here apparently)
        z = epsilon + actual_module.forward(activations_at_start)
        # divide the outputs by the relevance
        s = torch.div(relevance, z)
        # multiply with weights matrix and perform a backwards pass to get the unit relevance
        torch.multiply(z, s.data).sum().backward(retain_graph=True)
        # multiple activations with gradients to get the final relevance
        c = activations_at_start * activations_at_start.grad
        return c