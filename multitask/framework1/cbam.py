import torch 
import torch.nn as nn
import torch.nn.functional as F

class SAM(nn.Module):
    def __init__(self, bias=False):
        super(SAM, self).__init__()
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=2, out_channels=1, 
                              kernel_size=7, stride=1, padding=3, 
                              dilation=1, bias=self.bias)

    def forward(self, x):
        max_pool = torch.max(x, 1)[0].unsqueeze(1)
        avg_pool = torch.mean(x, 1).unsqueeze(1)
        
        concat = torch.cat((max_pool, avg_pool), dim=1)
        
        mask = self.conv(concat)
        mask = torch.sigmoid(mask)
        
        return x * mask 

class CAM(nn.Module):
    def __init__(self, channels, reduction):
        super(CAM, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.SiLU(inplace=True), # SiLU Is smoother than ReLU
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        # Global Pooling
        max_pool = F.adaptive_max_pool2d(x, output_size=1)
        avg_pool = F.adaptive_avg_pool2d(x, output_size=1)
        
        linear_max = self.mlp(max_pool)
        linear_avg = self.mlp(avg_pool)
        
        mask = linear_max + linear_avg
        mask = torch.sigmoid(mask)
        
        return x * mask
    
class CBAM(nn.Module):
    def __init__(self, channels, reduction=8, skip_connection = True):
        super(CBAM, self).__init__()
        self.cam = CAM(channels=channels, reduction=reduction)
        self.sam = SAM(bias=False)
        self.skip_connection = skip_connection

    def forward(self, x):
        output = self.cam(x)
        output = self.sam(x)

        if self.skip_connection:
            return x + output
        else:
            return output