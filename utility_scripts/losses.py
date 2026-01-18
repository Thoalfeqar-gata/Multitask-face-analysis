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
def geodesic_loss(m1, m2, eps = 1e-5):
    R_diffs = m1 @ m2.permute(0, 2, 1)
    traces = R_diffs.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    dists = torch.acos(torch.clamp((traces - 1) / 2, -1 + eps, 1 - eps))
    return dists.mean()



# Adopted from: https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py
class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=6, gamma_pos=1, clip=0.1, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum() / x.size(0)

