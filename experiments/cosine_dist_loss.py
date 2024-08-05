import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM


class SSIMLoss(torch.nn.Module):
    def __init__(self, data_range=1.0, size_average=True, channel=3):
        super(SSIMLoss, self).__init__()
        self.ssim_module = SSIM(data_range=data_range, size_average=size_average, channel=channel)
    
    def forward(self, img1, img2):
        ssim_value = self.ssim_module(img1, img2)
        return 1 - ssim_value

class CosineDistanceLoss(nn.Module):
    def __init__(self):
        super(CosineDistanceLoss, self).__init__()

    def forward(self, input1, input2):
        # Flatten the images: shape from [b, 1, 28, 28] to [b, 784]
        input1_flat = input1.view(input1.size(0), -1)
        input2_flat = input2.view(input2.size(0), -1)
        
        # Compute cosine similarity, then convert to cosine distance
        cosine_sim = F.cosine_similarity(input1_flat, input2_flat)
        cosine_dist = 1 - cosine_sim
        
        # Calculate the mean of the cosine distances
        loss = cosine_dist.mean()
        return loss

class HybridCosineDistanceCrossEntopyLoss(nn.Module):
    def __init__(self, _lambda=0.5, mode = None, step_size = 1e-5, max_lambda = 0.5, min_lambda = 0.0):
        super(HybridCosineDistanceCrossEntopyLoss, self).__init__()
        self._lambda = _lambda
        self.cosine_loss = CosineDistanceLoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        # Extra params for dynamic updating of _lambda
        self.mode = mode
        self.step_size = step_size
        self.max_lambda = max_lambda
        self.min_lambda = min_lambda

    def forward(self, explanation_output, explanation_target, classification_output, classification_target):
        # Compute cosine distance loss
        cosine_loss = self.cosine_loss(explanation_output, explanation_target)
        
        # Compute cross-entropy loss
        cross_entropy_loss = self.cross_entropy_loss(classification_output, classification_target)
        
        # Combine the losses
        loss = self._lambda * cosine_loss + (1 - self._lambda) * cross_entropy_loss
        if torch.isnan(loss).any() or torch.isinf(loss).any():            
            print(f"cosine_loss: {cosine_loss.item()}, cross_entropy_loss: {cross_entropy_loss.item()}")
            print("returning 0 loss due to nan or inf")
            return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
        return loss, cosine_loss, cross_entropy_loss
    
    def step_lambda(self):
        assert self.mode != None, "Cosine loss step mode not set"
        if self.mode == "ascending":
            self._lambda = min(self.max_lambda, self._lambda + self.step_size)
        elif self.mode == "descending":
            self._lambda = max(self.min_lambda, self._lambda - self.step_size)
        else:
            raise ValueError("Invalid step mode for cosine loss stepping")
        
class HybridSSIMDistanceCrossEntopyLoss(nn.Module):
    def __init__(self, _lambda=0.5, mode = None, step_size = 1e-5, max_lambda = 0.5, min_lambda = 0.0):
        super(HybridSSIMDistanceCrossEntopyLoss, self).__init__()
        self._lambda = _lambda
        self.ssim_loss = SSIMLoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        # Extra params for dynamic updating of _lambda
        self.mode = mode
        self.step_size = step_size
        self.max_lambda = max_lambda
        self.min_lambda = min_lambda

    def forward(self, explanation_output, explanation_target, classification_output, classification_target):
        # Compute cosine distance loss
        ssim_loss = self.ssim_loss(explanation_output, explanation_target)
        
        # Compute cross-entropy loss
        cross_entropy_loss = self.cross_entropy_loss(classification_output, classification_target)
        
        # Combine the losses
        loss = self._lambda * ssim_loss + (1 - self._lambda) * cross_entropy_loss
        if torch.isnan(loss).any() or torch.isinf(loss).any():            
            print(f"ssim_loss: {ssim_loss.item()}, cross_entropy_loss: {cross_entropy_loss.item()}")
            print("returning 0 loss due to nan or inf")
            return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
        return loss, ssim_loss, cross_entropy_loss
    
    def step_lambda(self):
        assert self.mode != None, "SSIM loss step mode not set"
        if self.mode == "ascending":
            self._lambda = min(self.max_lambda, self._lambda + self.step_size)
        elif self.mode == "descending":
            self._lambda = max(self.min_lambda, self._lambda - self.step_size)
        else:
            raise ValueError("Invalid step mode for ssim loss stepping")
    
# Example usage
if __name__ == "__main__":
    # Random tensors simulating image batches
    img1 = torch.randn(10, 1, 28, 28)  # Batch of 10 images
    img2 = torch.randn(10, 1, 28, 28)  # Batch of 10 images
    
    # Initialize the loss function
    loss_func = CosineDistanceLoss()
    
    # Calculate loss
    loss = loss_func(img1, img2)
    print("Cosine Distance Loss:", loss.item())
