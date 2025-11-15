import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from multitask.face_recognition_heads import build_head
from multitask.fusion import MultiScaleFusion
from multitask.cbam import CBAM


class FaceRecognitionSubnet(nn.Module):
    def __init__(
            self,
            classnum,
            head_type = 'adaface',
            feature_embedding_dim = 768, 
            embedding_dim = 512,
            m = 0.4, 
            t_alpha = 1.0, 
            h = 0.333, 
            s = 64.0,
        ):
        
        super(FaceRecognitionSubnet, self).__init__()

        self.norm_layer = nn.LayerNorm(feature_embedding_dim)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.feature_head = nn.Sequential(
            nn.Linear(in_features = feature_embedding_dim, out_features=feature_embedding_dim, bias = False),
            nn.BatchNorm1d(num_features = feature_embedding_dim, eps = 2e-5),
            nn.Linear(in_features = feature_embedding_dim, out_features = embedding_dim, bias = False),
            nn.BatchNorm1d(num_features = embedding_dim, eps = 2e-5)
        )

        self.head = build_head(
            head_type = head_type,
            embedding_size = embedding_dim,
            classnum = classnum,
            m = m,
            t_alpha = t_alpha,
            h = h,
            s = s,
        )
            
    def forward(self, multiscale_features, labels):
        x = multiscale_features[-1] #obtain the feature vector from the last stage, which has a shape of 49x768
        x = self.norm_layer(x) # B, 49, 768 
        x = self.avgpool(x.transpose(1, 2)) # B, 768, 1
        x = torch.flatten(x, 1) # B, 768
        embedding = self.feature_head(x)
        embedding_norm = torch.norm(embedding, p=2, dim=1, keepdim = True)
        normalized_embedding = torch.div(embedding, embedding_norm)
        
        logits = self.head(normalized_embedding, embedding_norm, labels)

        return logits


class AgeEstimationSubnet(nn.Module):
    def __init__(self):
        super(AgeEstimationSubnet, self).__init__()

        self.fusion = MultiScaleFusion(out_channels=[47, 93, 186, 186])
        self.cbam = CBAM(channels=512)
        self.head = nn.Sequential( # A simple regression head
            nn.AdaptiveMaxPool2d((1, 1)), # 7x7x512 ->1x1x512
            nn.Flatten(), # 1x1x512 -> 512
            nn.ReLU(),
            nn.Linear(in_features = 512, out_features = 256, bias = True),
            nn.ReLU(),
            nn.Linear(in_features = 256, out_features = 1, bias = True),
        )

    def forward(self, multiscale_features):
        x = self.fusion(multiscale_features)
        x = self.cbam(x)
        return self.head(x)


class GenderEstimationSubnet(nn.Module):
    def __init__(self):
        super(GenderEstimationSubnet, self).__init__()
        
        self.fusion = MultiScaleFusion(out_channels=[47, 93, 186, 186])
        self.cbam = CBAM(channels=512)
        self.head = nn.Sequential( # A simple binary classification head
            nn.AdaptiveMaxPool2d((1, 1)), # 7x7x512 -> 1x1x512
            nn.Flatten(), # 1x1x512 -> 512
            nn.ReLU(),
            nn.Linear(in_features = 512, out_features = 256, bias = True),
            nn.ReLU(),
            nn.Linear(in_features = 256, out_features = 1, bias = True),
        )
    
    def forward(self, multiscale_features):
        x = self.fusion(multiscale_features)
        x = self.cbam(x)
        return self.head(x)



class EmotionRecognitionSubnet(nn.Module):
    def __init__(self, num_classes = 7):
        super(EmotionRecognitionSubnet, self).__init__()

        self.head = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)), # 7x7x512 -> 1x1x512
            nn.Flatten(), # 1x1x512 -> 512
            nn.ReLU(),
            nn.Linear(in_features = 512, out_features = 256, bias = True),
            nn.ReLU(),
            nn.Linear(in_features = 256, out_features = num_classes, bias = True),

        )


    def forward(self, x): # x represents the fused features
        return self.head(x)