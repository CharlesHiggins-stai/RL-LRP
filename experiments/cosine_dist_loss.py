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
    def __init__(self, _lambda=0.5, mode=None, max_steps = 100000):
        super(HybridCosineDistanceCrossEntopyLoss, self).__init__()
        self._lambda = _lambda
        self.cosine_loss = CosineDistanceLoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.step_counter = 0
        self.num_steps = max_steps
        self.mode = mode 
        if mode != None:
            print(f"mode specified here as {mode}")
            if mode == "increasing":
                self.start_value = _lambda
                self.end_value = 1
            elif mode == "decreasing":
                self.start_value = _lambda
                self.end_value = 0
            else:
                raise ValueError(f"Unsupported _lambda schedule mode specified: {mode}")

    def forward(self, explanation_output, explanation_target, classification_output, classification_target):
        # Compute cosine distance loss
        cosine_loss = self.cosine_loss(explanation_output, explanation_target)
        
        # Compute cross-entropy loss
        cross_entropy_loss = self.cross_entropy_loss(classification_output, classification_target)
        
        # Combine the losses
        loss = self._lambda * cosine_loss + (1 - self._lambda) * cross_entropy_loss
        self.step()
        return loss
    
    def step(self):
        # Calculate the current value of _lambda based on the progress
        if self.mode != None:
            if self.mode == 'decreasing':
                new_value = self.start_value - (self.start_value - self.end_value) * (self.step_counter / self.num_steps)
            elif self.mode == 'increasing':
                new_value = self.start_value + (self.end_value - self.start_value) * (self.step_counter / self.num_steps)
            else:
                raise ValueError("Mode should be either 'decreasing' or 'increasing'")
                
            # Ensure _lambda stays within the bounds
            self._lambda = max(0.0, min(new_value, 1.0))
            
        # Increment the step count
        self.step_counter += 1
        
        
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
