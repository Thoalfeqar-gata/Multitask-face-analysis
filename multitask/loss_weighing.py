import numpy as np
import torch
import torch.nn as nn

"""

    Weighting by learning speed.

"""

class DynamicWeightAverage:
    def __init__(self, num_tasks, Temperature = 1.0):
        self.num_tasks = num_tasks
        self.T = Temperature
        # Store the average loss of each task from the previous epoch
        self.avg_losses_previous_epoch = None

    def calculate_weights(self, avg_losses_current_epoch):
        """
            Calculates the dynamic weights for each task based on the current and previous epoch's average losses.
            Args:
                avg_losses_current_epoch: A numpy array of the average loss for each task in the current epoch.
            Returns:
                weights: A numpy array of weights for each task.
        """
        if avg_losses_current_epoch is None:
            return np.ones(self.num_tasks, dtype=np.float32)

        # On the first epoch, use equal weights and store the current losses for the next epoch.
        if self.avg_losses_previous_epoch is None:
            weights = np.ones(self.num_tasks, dtype=np.float32)
            self.avg_losses_previous_epoch = avg_losses_current_epoch.copy()
            return weights
        
        else:
            W = np.zeros(self.num_tasks, dtype=np.float32)
            W = np.divide(avg_losses_current_epoch, self.avg_losses_previous_epoch)
            weights = np.exp(W / self.T)
            weights = weights / np.sum(weights)
            weights *= self.num_tasks

            self.avg_losses_previous_epoch = avg_losses_current_epoch.copy()

            return weights


class UncertaintyLossWrapper(nn.Module):
    def __init__(self, num_tasks=6):
        super(UncertaintyLossWrapper, self).__init__()
        # We define log_var (log(sigma^2)) instead of sigma for numerical stability
        # Initializing to 0.0 means sigma=1, equivalent to standard sum
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):
        """
        losses: list of standard calculated losses [L_fr, L_em, L_age, ...]
        """
        total_loss = 0
        
        for i, loss in enumerate(losses):
            # Precision = 1 / (2 * sigma^2) = 0.5 * exp(-log_var)
            precision = 0.5 * torch.exp(-self.log_vars[i])
            
            total_loss += (precision * loss) + (0.5 * self.log_vars[i])
            
        return total_loss

    def get_weights(self):
        # Returns 1 / (2 * sigma^2)
        return 0.5 * torch.exp(-self.log_vars.detach())