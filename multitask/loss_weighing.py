import numpy as np


"""

    Weighting by learning speed.

"""

class DynamicWeightAverage:
    def __init__(self, num_tasks, Temperature=2.0):
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
        

