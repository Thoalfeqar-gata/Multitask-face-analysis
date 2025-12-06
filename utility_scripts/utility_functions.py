import scipy
import numpy as np
import torch
import torch.nn.functional as F



##################################

#   Age Estimation Utilities

##################################

def label_to_gaussian(label, min_age=0, max_age=100, sigma=2.0):
    """
    Generates a Gaussian distribution for a batch of labels.
    
    Args:
        label (torch.Tensor): Tensor of shape [Batch_Size] containing true ages.
    """
    # 1. Create ages vector [min, max] and reshape to [1, Classes] for broadcasting
    # We create it on the same device as the label
    ages = torch.arange(min_age, max_age + 1, device=label.device).float().view(1, -1)
    
    # 2. Reshape labels to [Batch, 1] so we can subtract: (1, C) - (B, 1) = (B, C)
    label = label.view(-1, 1).float()
    
    # 3. Calculate Gaussian
    distribution = torch.exp(-torch.pow(ages - label, 2) / (2 * sigma ** 2))
    
    # 4. Normalize so each distribution sums to 1 across the age dimension (dim=1)
    distribution = distribution / torch.sum(distribution, dim=1, keepdim=True)
    
    return distribution


def dldl_loss(logits, target, min_age=0, max_age=100, sigma=2.0, l1_weight=1.0):
    """
    Computes DLDL-v2 loss (KL Divergence + L1 Expectation Loss).
    """
    # 1. Calculate Probabilities
    log_probs = F.log_softmax(logits, dim=1) # [Batch, Classes], log_probs are needed for F.kl_div to work.
    probs = torch.exp(log_probs)             # [Batch, Classes]
    
    # 2. Generate Target Distributions
    target_distribution = label_to_gaussian(target, min_age, max_age, sigma)
    
    # 3. KL Divergence Loss
    # reduction='batchmean' divides by batch size, which is mathematically correct for KL
    loss_kl = F.kl_div(log_probs, target_distribution, reduction='batchmean')
    
    # 4. Expectation L1 Loss (Correction: use dim=1)
    ages = torch.arange(min_age, max_age + 1, device=logits.device).float()
    
    # Calculate expected age for EACH sample: sum(p_i * age_i)
    # Result shape: [Batch]
    expectation = torch.sum(probs * ages, dim=1)
    
    loss_l1 = F.l1_loss(expectation, target.float())
    
    # Combine losses (Papers often weight them, but 1:1 is a standard start)
    return loss_kl + (l1_weight * loss_l1)



##################################

#   Pose Estimation Utilities

##################################

def mat_to_rotation_matrix(mat_file: str):
    """
    (Incomplete) Intended to extract a rotation matrix from a .mat file.

    Args:
        mat_file (str): Path to the .mat file.

    Returns:
        str: A placeholder string.
    """
    # TODO: Implement the logic to extract and return the actual rotation matrix.
    content = scipy.io.loadmat(mat_file)
    print(content)

    return "Cookie!"