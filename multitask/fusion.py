import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleFusion(nn.Module):
    """
        This module fuses the multiscale features output by a swin or a davit backbone.
        The features come from the four stages in the backbone.
        The feature sizes are as follows:
            transformer_embedding_dim = 96 for tiny and small model, 128 for base model
            stage_0: 56x56, transformer_embedding_dim channels
            stage_1: 28x28, transformer_embedding_dim * 2 channels
            stage_2: 14x14, transformer_embedding_dim * 4 channels
            stage_3: 7x7, tranformer_embedding_dim * 8 channels

        The output shape after the fusion is:
            fused_output: 7x7, 512 channels
    """
    def __init__(self, out_channels=[47, 93, 186, 186], transformer_embedding_dim = 96):
        super(MultiScaleFusion, self).__init__()
        
        """
        
            We will follow a similar setup to swinface initially, and tune it later.
            The contribution of each stage to the final 512 channel size is as follows:
            stage_0: 47
            stage_1: 93
            stage_2: 186
            stage_3: 186
        
        """
        
        # stage_0
        self.conv11 = nn.Conv2d(in_channels = transformer_embedding_dim, out_channels = out_channels[0],
                                kernel_size = 3, stride = 2, padding = 1, bias = True) # 56x56 -> 28 x 28
        self.conv12 = nn.Conv2d(in_channels = out_channels[0], out_channels = out_channels[0],
                                kernel_size = 3, stride = 2, padding = 1, bias = True) # 28x28 -> 14 x 14
        self.conv13 = nn.Conv2d(in_channels = out_channels[0], out_channels = out_channels[0],
                                kernel_size = 3, stride = 2, padding = 1, bias = True) # 14x14 -> 7 x 7

        # stage_1
        self.conv21 = nn.Conv2d(in_channels = transformer_embedding_dim * 2, out_channels = out_channels[1],
                                kernel_size = 3, stride = 2, padding = 1, bias = True) # 28 x 28 -> 14 x 14
        self.conv22 = nn.Conv2d(in_channels = out_channels[1], out_channels = out_channels[1],
                                kernel_size = 3, stride = 2, padding = 1, bias = True) # 14 x 14 -> 7 x 7

        # stage_2
        self.conv31 = nn.Conv2d(in_channels = transformer_embedding_dim * 4, out_channels = out_channels[2],
                                kernel_size = 3, stride = 2, padding = 1, bias = True) # 14 x 14 -> 7 x 7
        
        # stage_3
        self.conv41 = nn.Conv2d(in_channels = transformer_embedding_dim * 8, out_channels = out_channels[3],
                                kernel_size = 3, stride = 1, padding = 1, bias = True) # 7 x 7 -> 7 x 7, but reduced the number of channels
        
        self.relu = nn.ReLU()
        
    def forward(self, multiscale_features):
        # Input features are expected to be in (N, L, C) format.
        # Reshape them to (N, C, H, W) for 2D convolutions.
        
        assert len(multiscale_features) == 4, "The number of multiscale features must be 4."
        
        stage_0, stage_1, stage_2, stage_3 = multiscale_features
        
        stage_0 = stage_0.permute(0, 2, 1).reshape(-1, stage_0.shape[2], 56, 56)
        stage_1 = stage_1.permute(0, 2, 1).reshape(-1, stage_1.shape[2], 28, 28)
        stage_2 = stage_2.permute(0, 2, 1).reshape(-1, stage_2.shape[2], 14, 14)
        stage_3 = stage_3.permute(0, 2, 1).reshape(-1, stage_3.shape[2], 7, 7)

        stage_0 = self.relu(self.conv11(stage_0))
        stage_0 = self.relu(self.conv12(stage_0))
        stage_0 = self.relu(self.conv13(stage_0))

        stage_1 = self.relu(self.conv21(stage_1))
        stage_1 = self.relu(self.conv22(stage_1))

        stage_2 = self.relu(self.conv31(stage_2))

        stage_3 = self.relu(self.conv41(stage_3))

        fused_output = torch.cat([stage_0, stage_1, stage_2, stage_3], dim = 1)
        
        return fused_output
    

