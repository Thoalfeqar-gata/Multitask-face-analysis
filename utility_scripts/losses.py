import scipy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


#################################################

#   Age Estimation deep label distribution loss

#################################################

def label_to_gaussian(label, min_age, max_age, sigma):

    ages = torch.arange(min_age, max_age + 1, device=label.device).float().view(1, -1)
    label = label.view(-1, 1).float()
    
    # Calculate Gaussian
    squared_diff = torch.pow(ages - label, 2)
    distribution = torch.exp(-squared_diff / (2 * sigma ** 2))
    
    # Normalize with epsilon for safety
    distribution = distribution / (torch.sum(distribution, dim=1, keepdim=True) + 1e-10)
    
    return distribution

def dldl_loss(logits, target, min_age=0, max_age=101, sigma=2.0, l1_weight=1.0):
    # 1. Calc Probabilities
    log_probs = F.log_softmax(logits, dim=1) 
    probs = torch.exp(log_probs)
    
    # 2. Target Dist
    target_distribution = label_to_gaussian(target, min_age, max_age, sigma)
    
    # 3. KL Loss
    loss_kl = F.kl_div(log_probs, target_distribution, reduction='batchmean')
    
    # 4. Expectation L1
    # Create ages vector consistent with logits dimension
    ages = torch.arange(min_age, max_age + 1, device=logits.device).float()
    
    # Verify shape consistency (Optional debugging)
    if logits.size(1) != len(ages):
        raise ValueError(f"Logits classes {logits.size(1)} != Age range {len(ages)}")

    expectation = torch.sum(probs * ages, dim=1)
    loss_l1 = F.l1_loss(expectation, target.float())
    
    return loss_kl + (l1_weight * loss_l1)


#################################################

#   Pose estimation geodesic loss

#################################################

#matrices batch*3*3
#both matrix are orthogonal rotation matrices
#out theta between 0 to 180 degree batch
class GeodesicLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, m1, m2):
        m = torch.bmm(m1, m2.transpose(1,2)) #batch*3*3
        
        trace = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2])
        cos = (trace - 1) / 2
        theta_loss = torch.acos(torch.clamp(cos, -1+self.eps, 1-self.eps))
         
        return torch.mean(theta_loss)
