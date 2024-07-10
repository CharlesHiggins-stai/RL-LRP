import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, _lambda=0.5):
        super(HybridCosineDistanceCrossEntopyLoss, self).__init__()
        self._lambda = _lambda
        self.cosine_loss = CosineDistanceLoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

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
