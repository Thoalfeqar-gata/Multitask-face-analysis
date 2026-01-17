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


class LightMultiScaleFusion(nn.Module):
    """
        This module fuses the multiscale features output by a swin or a davit backbone.
        The features come from the four stages in the backbone.
        The input feature sizes are as follows:
            transformer_embedding_dim = 96 for tiny and small model, 128 for base model
            stage_0: h0xw0, transformer_embedding_dim channels
            stage_1: h1xw1, transformer_embedding_dim * 2 channels
            stage_2: h2xw2, transformer_embedding_dim * 4 channels
            stage_3: h3xw3, transformer_embedding_dim * 8 channels

        The output shape after the fusion is:
            fused_output: 7x7x512 out_channel_dim
    """
    def __init__(self, out_channel_dim = 512, transformer_embedding_dim = 96):
        super(LightMultiScaleFusion, self).__init__()
        
        self.activation = nn.SiLU(inplace = True)
        self.total_channels = transformer_embedding_dim * (8 + 4 + 2 + 1)

        # Spatial downsampling layers.
        # stage_0
        self.down0 = nn.Sequential(
            DepthwiseSeparableConv(in_channels = transformer_embedding_dim, out_channels = transformer_embedding_dim,
                                kernel_size = 3, stride = 2, padding = 1), # 56 x 56 x transformer_embedding_dim -> 28 x 28 x transformer_embedding_dim
            self.activation,
            DepthwiseSeparableConv(in_channels = transformer_embedding_dim, out_channels = transformer_embedding_dim,
                                kernel_size = 3, stride = 2, padding = 1), # 28 x 28 x transformer_embedding_dim -> 14 x 14 x transformer_embedding_dim
            self.activation,
            DepthwiseSeparableConv(in_channels = transformer_embedding_dim, out_channels = transformer_embedding_dim,
                                kernel_size = 3, stride = 2, padding = 1), # 14 x 14 x transformer_embedding_dim -> 7 x 7 x transformer_embedding_dim
        )

        # stage_1
        self.down1 = nn.Sequential(
            DepthwiseSeparableConv(in_channels = transformer_embedding_dim * 2, out_channels = transformer_embedding_dim * 2,
                                kernel_size = 3, stride = 2, padding = 1), # 28 x 28 x transformer_embedding_dim * 2 -> 14 x 14 x transformer_embedding_dim * 2
            self.activation,
            DepthwiseSeparableConv(in_channels = transformer_embedding_dim * 2, out_channels = transformer_embedding_dim * 2,
                                kernel_size = 3, stride = 2, padding = 1), # 14 x 14 x transformer_embedding_dim * 2 -> 7 x 7 x transformer_embedding_dim * 2
        )


        # stage_2
        self.down2 = DepthwiseSeparableConv(in_channels = transformer_embedding_dim * 4, out_channels = transformer_embedding_dim * 4,
                                kernel_size = 3, stride = 2, padding = 1) # 14 x 14 x transformer_embedding_dim * 4 -> 7 x 7 x transformer_embedding_dim * 4
        
        # No downsampling for stage 3
        
        
        
        # CBAM Blocks used to apply attention before spatial downsampling. Don't use skip connection to filter unrelated features.
        self.cbam0 = CBAM(channels = transformer_embedding_dim, reduction = 16, skip_connection=False)
        self.cbam1 = CBAM(channels = transformer_embedding_dim * 2, reduction = 16, skip_connection=False)
        self.cbam2 = CBAM(channels = transformer_embedding_dim * 4, reduction = 16, skip_connection=False)


        # Final CBAM block to apply attention after the concatenation. Use skip connection to refine the features instead of filtering them.
        self.cbam_final = CBAM(channels = transformer_embedding_dim * (8 + 4 + 2 + 1), reduction = 16, skip_connection=True)

        # 1x1 conv to reduce the number of channels after the concatenation + CBAM.
        self.compressor = nn.Sequential(
            nn.Conv2d(self.total_channels, out_channel_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel_dim),
            self.activation
        )

    def forward(self, multiscale_features):
        # Input features are expected to be in (N, L, C) format.
        # Reshape them to (N, C, H, W) for 2D convolutions.
        
        assert len(multiscale_features) == 4, "The number of multiscale features must be 4."
        
        stage_0, stage_1, stage_2, stage_3 = multiscale_features
        
        B, L0, C0 = stage_0.shape
        h0 = w0 = int(L0 ** 0.5)
        stage_0 = stage_0.permute(0, 2, 1).reshape(-1, C0, h0, w0)

        B, L1, C1 = stage_1.shape
        h1 = w1 = int(L1 ** 0.5)
        stage_1 = stage_1.permute(0, 2, 1).reshape(-1, C1, h1, w1)
        
        B, L2, C2 = stage_2.shape
        h2 = w2 = int(L2 ** 0.5)
        stage_2 = stage_2.permute(0, 2, 1).reshape(-1, C2, h2, w2)
        
        B, L3, C3 = stage_3.shape
        h3 = w3 = int(L3 ** 0.5)
        stage_3 = stage_3.permute(0, 2, 1).reshape(-1, C3, h3, w3)


        # attention before spatial downsampling
        stage_0 = self.cbam0(stage_0) 
        stage_1 = self.cbam1(stage_1)
        stage_2 = self.cbam2(stage_2)
    

        # spatial downsampling
        stage_0 = self.down0(stage_0)
        stage_1 = self.down1(stage_1)
        stage_2 = self.down2(stage_2)
        # no spatial downsampling for stage 3


        # concatenate the features to generate a giant feature vector
        fused_features = torch.cat([stage_0, stage_1, stage_2, stage_3], dim = 1)

        # apply CBAM before channel compression
        fused_features = self.cbam_final(fused_features)

        # perform channel compression
        fused_features = self.compressor(fused_features)
        
        return fused_features
    


class StandardMultiScaleFusion(nn.Module):
    """
        This module fuses the multiscale features output by a swin or a davit backbone.
        The features come from the four stages in the backbone.
        The input feature sizes are as follows:
            transformer_embedding_dim = 96 for tiny and small model, 128 for base model
            stage_0: h0xw0, transformer_embedding_dim channels
            stage_1: h1xw1, transformer_embedding_dim * 2 channels
            stage_2: h2xw2, transformer_embedding_dim * 4 channels
            stage_3: h3xw3, transformer_embedding_dim * 8 channels

        The output shape after the fusion is:
            fused_output: 7x7xout_channel_dim 
    """
    def __init__(self, out_channel_dim = 512, transformer_embedding_dim = 96):
        super(StandardMultiScaleFusion, self).__init__()
        
        self.activation = nn.SiLU(inplace = True)
        self.total_channels = transformer_embedding_dim * (8 + 4 + 2 + 1)

        # The following depthwise convolutional layers are used to downsample the spatial resolution and keeping the channel size fixed.
        # stage_0
        self.down0 = nn.Sequential(
            nn.Conv2d(in_channels = transformer_embedding_dim, out_channels = transformer_embedding_dim, 
                                kernel_size = 3, stride = 2, padding = 1, bias = False), # 56 x 56 x transformer_embedding_dim -> 28 x 28 x transformer_embedding_dim
            self.activation,
            nn.Conv2d(in_channels = transformer_embedding_dim, out_channels = transformer_embedding_dim,
                                kernel_size = 3, stride = 2, padding = 1, bias = False), # 28 x 28 x transformer_embedding_dim -> 14 x 14 x transformer_embedding_dim
            self.activation,
            nn.Conv2d(in_channels = transformer_embedding_dim, out_channels = transformer_embedding_dim,
                                kernel_size = 3, stride = 2, padding = 1, bias = False) # 14 x 14 x transformer_embedding_dim -> 7 x 7 x transformer_embedding_dim
        )

        # stage_1
        self.down1 =  nn.Sequential(
            nn.Conv2d(in_channels = transformer_embedding_dim * 2, out_channels = transformer_embedding_dim * 2,
                                kernel_size = 3, stride = 2, padding = 1, bias = False), # 28 x 28 x transformer_embedding_dim * 2 -> 14 x 14 x transformer_embedding_dim 2
            self.activation,
            nn.Conv2d(in_channels = transformer_embedding_dim * 2, out_channels = transformer_embedding_dim * 2,
                                kernel_size = 3, stride = 2, padding = 1, bias = False) # 14 x 14 x transformer_embedding_dim * 2 -> 7 x 7 x transformer_embedding_dim * 2
        )

        # stage_2
        self.down2 = nn.Conv2d(
            in_channels = transformer_embedding_dim * 4, out_channels = transformer_embedding_dim * 4,
            kernel_size = 3, stride = 2, padding = 1
        )

        # no downsampling for stage 3
        
        
        # CBAM Blocks used to apply attention before spatial downsampling (stages 1, 2, and 3). Don't use skip connection to filter unrelated features.
        self.cbam0 = CBAM(channels = transformer_embedding_dim, reduction = 8, skip_connection=False)
        self.cbam1 = CBAM(channels = transformer_embedding_dim * 2, reduction = 8, skip_connection=False)
        self.cbam2 = CBAM(channels = transformer_embedding_dim * 4, reduction = 8, skip_connection=False)
  

        # Final CBAM block to apply attention after the concatenation. Use skip connection to refine the features instead of filtering them.
        self.cbam_final = CBAM(channels = transformer_embedding_dim * (8 + 4 + 2 + 1), reduction = 8, skip_connection=True)

        # 1x1 conv to reduce the number of channels after the concatenation + CBAM.
        self.compressor = nn.Sequential(
            nn.Conv2d(self.total_channels, out_channel_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel_dim),
            self.activation
        )


    def forward(self, multiscale_features):
        # Input features are expected to be in (N, L, C) format.
        # Reshape them to (N, C, H, W) for 2D convolutions.
        
        assert len(multiscale_features) == 4, "The number of multiscale features must be 4."
        
        stage_0, stage_1, stage_2, stage_3 = multiscale_features
        
        B, L0, C0 = stage_0.shape
        h0 = w0 = int(L0 ** 0.5)
        stage_0 = stage_0.permute(0, 2, 1).reshape(-1, C0, h0, w0)

        B, L1, C1 = stage_1.shape
        h1 = w1 = int(L1 ** 0.5)
        stage_1 = stage_1.permute(0, 2, 1).reshape(-1, C1, h1, w1)
        
        B, L2, C2 = stage_2.shape
        h2 = w2 = int(L2 ** 0.5)
        stage_2 = stage_2.permute(0, 2, 1).reshape(-1, C2, h2, w2)
        
        B, L3, C3 = stage_3.shape
        h3 = w3 = int(L3 ** 0.5)
        stage_3 = stage_3.permute(0, 2, 1).reshape(-1, C3, h3, w3)


        # attention before spatial downsampling
        stage_0 = self.cbam0(stage_0) 
        stage_1 = self.cbam1(stage_1)
        stage_2 = self.cbam2(stage_2)
    

        # spatial downsampling
        stage_0 = self.down0(stage_0)
        stage_1 = self.down1(stage_1)
        stage_2 = self.down2(stage_2)
        # no spatial downsampling for stage 3


        # concatenate the features to generate a giant feature vector
        fused_features = torch.cat([stage_0, stage_1, stage_2, stage_3], dim = 1)

        # apply CBAM before channel compression
        fused_features = self.cbam_final(fused_features)

        # perform channel compression
        fused_features = self.compressor(fused_features)
        
        return fused_features
