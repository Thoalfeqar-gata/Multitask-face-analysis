import torch
import torch.nn as nn
import torch.nn.functional as F
from multitask.framework1.cbam import CBAM



class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, 
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, 
            stride=1, padding=0, bias=False
        )
        self.bn_depthwise = nn.BatchNorm2d(in_channels)
        self.bn_pointwise = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)



    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn_depthwise(x)
        x = self.relu(x)
        
        x = self.pointwise(x)
        x = self.bn_pointwise(x)
        return x


class SmartMultiScaleFusion(nn.Module):
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
            fused_output: 7x7x512 channels
    """
    def __init__(self, stage_out_channels=[50, 100, 150, 212], transformer_embedding_dim = 96):
        super(SmartMultiScaleFusion, self).__init__()
        
        """
        
            The contribution of each stage to the final 512 channel size is as follows:
            stage_0: stage_out_channels[0]
            stage_1: stage_out_channels[1]
            stage_2: stage_out_channels[2]
            stage_3: stage_out_channels[3]
        
        """
        
        # stage_0
        self.conv11 = DepthwiseSeparableConv(in_channels = transformer_embedding_dim, out_channels = stage_out_channels[0],
                                kernel_size = 3, stride = 2, padding = 1) # 56 x 56 x transformer_embedding_dim -> 28 x 28 x stage_out_channels[0]
        self.conv12 = DepthwiseSeparableConv(in_channels = stage_out_channels[0], out_channels = stage_out_channels[0],
                                kernel_size = 3, stride = 2, padding = 1) # 28 x 28 x stage_out_channels[0] -> 14 x 14 x stage_out_channels[0]
        self.conv13 = DepthwiseSeparableConv(in_channels = stage_out_channels[0], out_channels = stage_out_channels[0],
                                kernel_size = 3, stride = 2, padding = 1) # 14 x 14 x stage_out_channels[0] -> 7 x 7 x stage_out_channels[0]

        # stage_1
        self.conv21 = DepthwiseSeparableConv(in_channels = transformer_embedding_dim * 2, out_channels = stage_out_channels[1],
                                kernel_size = 3, stride = 2, padding = 1) # 28 x 28 x transformer_embedding_dim * 2 -> 14 x 14 x stage_out_channels[1]
        self.conv22 = DepthwiseSeparableConv(in_channels = stage_out_channels[1], out_channels = stage_out_channels[1],
                                kernel_size = 3, stride = 2, padding = 1) # 14 x 14 x stage_out_channels[1] -> 7 x 7 x stage_out_channels[1]

        # stage_2
        self.conv31 = DepthwiseSeparableConv(in_channels = transformer_embedding_dim * 4, out_channels = stage_out_channels[2],
                                kernel_size = 3, stride = 2, padding = 1) # 14 x 14 x transformer_embedding_dim * 4 -> 7 x 7 x stage_out_channels[2]
        
        # stage_3
        self.conv41 = DepthwiseSeparableConv(in_channels = transformer_embedding_dim * 8, out_channels = stage_out_channels[3],
                                kernel_size = 3, stride = 1, padding = 1) # 7 x 7 x transformer_embedding_dim * 8 -> 7 x 7 x stage_out_channels[3]
        
        
        # CBAM Blocks used to apply attention before downsampling. Don't use skip connection to filter unrelated features.
        self.cbam0 = CBAM(channels = transformer_embedding_dim, reduction = 4, skip_connection=False)
        self.cbam1 = CBAM(channels = transformer_embedding_dim * 2, reduction = 4, skip_connection=False)
        self.cbam2 = CBAM(channels = transformer_embedding_dim * 4, reduction = 4, skip_connection=False)
        self.cbam3 = CBAM(channels = transformer_embedding_dim * 8, reduction = 4, skip_connection=False)

        # Final CBAM block to apply attention after the concatenation. Use skip connection to refine the features instead of filtering them.
        self.cbam_final = CBAM(channels = 512, reduction = 4, skip_connection=True)
        
        self.relu = nn.ReLU(inplace = True)


    def forward(self, multiscale_features):
        # Input features are expected to be in (N, L, C) format.
        # Reshape them to (N, C, H, W) for 2D convolutions.
        
        assert len(multiscale_features) == 4, "The number of multiscale features must be 4."
        
        stage_0, stage_1, stage_2, stage_3 = multiscale_features
        
        stage_0 = stage_0.permute(0, 2, 1).reshape(-1, stage_0.shape[2], 56, 56)
        stage_1 = stage_1.permute(0, 2, 1).reshape(-1, stage_1.shape[2], 28, 28)
        stage_2 = stage_2.permute(0, 2, 1).reshape(-1, stage_2.shape[2], 14, 14)
        stage_3 = stage_3.permute(0, 2, 1).reshape(-1, stage_3.shape[2], 7, 7)

        stage_0 = self.cbam0(stage_0)
        stage_0 = self.relu(self.conv11(stage_0))
        stage_0 = self.relu(self.conv12(stage_0))
        stage_0 = self.relu(self.conv13(stage_0))

        stage_1 = self.cbam1(stage_1)
        stage_1 = self.relu(self.conv21(stage_1))
        stage_1 = self.relu(self.conv22(stage_1))

        stage_2 = self.cbam2(stage_2)
        stage_2 = self.relu(self.conv31(stage_2))

        stage_3 = self.cbam3(stage_3)
        stage_3 = self.relu(self.conv41(stage_3))

        fused_output = torch.cat([stage_0, stage_1, stage_2, stage_3], dim = 1)
        
        
        return self.cbam_final(fused_output)
    

class StandardMultiScaleFusion(nn.Module):
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
            fused_output: 7x7x512 channels
    """
    def __init__(self, stage_out_channels=[50, 100, 150, 212], transformer_embedding_dim = 96):
        super(StandardMultiScaleFusion, self).__init__()
        
        """
        
            The contribution of each stage to the final 512 channel size is as follows:
            stage_0: stage_out_channels[0]
            stage_1: stage_out_channels[1]
            stage_2: stage_out_channels[2]
            stage_3: stage_out_channels[3]
        
        """
        
        # stage_0
        self.conv11 = nn.Conv2d(in_channels = transformer_embedding_dim, out_channels = stage_out_channels[0],
                                kernel_size = 3, stride = 2, padding = 1) # 56 x 56 x transformer_embedding_dim -> 28 x 28 x stage_out_channels[0]
        self.conv12 = nn.Conv2d(in_channels = stage_out_channels[0], out_channels = stage_out_channels[0],
                                kernel_size = 3, stride = 2, padding = 1) # 28 x 28 x stage_out_channels[0] -> 14 x 14 x stage_out_channels[0]
        self.conv13 = nn.Conv2d(in_channels = stage_out_channels[0], out_channels = stage_out_channels[0],
                                kernel_size = 3, stride = 2, padding = 1) # 14 x 14 x stage_out_channels[0] -> 7 x 7 x stage_out_channels[0]

        # stage_1
        self.conv21 = nn.Conv2d(in_channels = transformer_embedding_dim * 2, out_channels = stage_out_channels[1],
                                kernel_size = 3, stride = 2, padding = 1) # 28 x 28 x transformer_embedding_dim * 2 -> 14 x 14 x stage_out_channels[1]
        self.conv22 = nn.Conv2d(in_channels = stage_out_channels[1], out_channels = stage_out_channels[1],
                                kernel_size = 3, stride = 2, padding = 1) # 14 x 14 x stage_out_channels[1] -> 7 x 7 x stage_out_channels[1]

        # stage_2
        self.conv31 = nn.Conv2d(in_channels = transformer_embedding_dim * 4, out_channels = stage_out_channels[2],
                                kernel_size = 3, stride = 2, padding = 1) # 14 x 14 x transformer_embedding_dim * 4 -> 7 x 7 x stage_out_channels[2]
        
        # stage_3
        self.conv41 = nn.Conv2d(in_channels = transformer_embedding_dim * 8, out_channels = stage_out_channels[3],
                                kernel_size = 3, stride = 1, padding = 1) # 7 x 7 x transformer_embedding_dim * 8 -> 7 x 7 x stage_out_channels[3]
        

        self.cbam = CBAM(channels = 512, reduction = 4, skip_connection=True)
        self.relu = nn.ReLU(inplace = True)


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
        
        
        return self.cbam(fused_output)