import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from multitask.face_recognition_heads import build_head
from multitask.fusion import MultiScaleFusion
from multitask.cbam import CBAM


class FaceRecognitionEmbeddingSubnet(nn.Module):
    def __init__(
            self,
            feature_embedding_dim = 768, 
            embedding_dim = 512,
        ):
        
        super(FaceRecognitionEmbeddingSubnet, self).__init__()

        self.norm_layer = nn.LayerNorm(feature_embedding_dim)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.feature_head = nn.Sequential(
            nn.Linear(in_features = feature_embedding_dim, out_features=feature_embedding_dim, bias = False),
            nn.BatchNorm1d(num_features = feature_embedding_dim, eps = 2e-5),
            nn.Linear(in_features = feature_embedding_dim, out_features = embedding_dim, bias = False),
            nn.BatchNorm1d(num_features = embedding_dim, eps = 2e-5)
        )

            
    def forward(self, multiscale_features):
        x = multiscale_features[-1] #obtain the feature vector from the last stage, which has a shape of 49x768 or 49x1024
        x = self.norm_layer(x) # B, 49, 768 
        x = self.avgpool(x.transpose(1, 2)) # B, 768, 1
        x = torch.flatten(x, 1) # B, 768
        embedding = self.feature_head(x)
        embedding_norm = torch.norm(embedding, p=2, dim=1, keepdim = True)
        normalized_embedding = torch.div(embedding, embedding_norm)

        return normalized_embedding, embedding_norm


class AgeEstimationSubnet(nn.Module):
    def __init__(self, transformer_embedding_dim = 96, min_age = 0, max_age = 101):
        super(AgeEstimationSubnet, self).__init__()
        self.num_classes = max_age - min_age + 1

        self.fusion = MultiScaleFusion(out_channels=[47, 93, 186, 186], transformer_embedding_dim=transformer_embedding_dim)
        self.cbam = CBAM(channels=512)
        self.head = nn.Sequential( # A simple distribution prediction head
            nn.AdaptiveAvgPool2d((1, 1)), # 7x7x512 ->1x1x512
            nn.Flatten(), # 1x1x512 -> 512
            nn.Linear(in_features = 512, out_features = 256, bias = True),
            nn.BatchNorm1d(num_features = 256, eps = 2e-5),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features = 256, out_features = self.num_classes, bias = True),
        )

    def forward(self, multiscale_features):
        x = self.fusion(multiscale_features)
        x = self.cbam(x)
        return self.head(x)


class GenderRecognitionSubnet(nn.Module):
    def __init__(self, transformer_embedding_dim = 96):
        super(GenderRecognitionSubnet, self).__init__()
        
        self.fusion = MultiScaleFusion(out_channels=[47, 93, 186, 186], transformer_embedding_dim=transformer_embedding_dim)
        self.cbam = CBAM(channels=512)
        self.head = nn.Sequential( # A simple binary classification head
            nn.AdaptiveAvgPool2d((1, 1)), # 7x7x512 -> 1x1x512
            nn.Flatten(), # 1x1x512 -> 512
            nn.Linear(in_features = 512, out_features = 256, bias = True),
            nn.BatchNorm1d(num_features = 256, eps = 2e-5),
            nn.ReLU(),
            nn.Linear(in_features = 256, out_features = 1, bias = True),
        )
    
    def forward(self, multiscale_features):
        x = self.fusion(multiscale_features)
        x = self.cbam(x)
        return self.head(x)



class EmotionRecognitionSubnet(nn.Module):
    def __init__(self, num_classes = 7, transformer_embedding_dim = 96):
        super(EmotionRecognitionSubnet, self).__init__()

        self.fusion = MultiScaleFusion(out_channels=[47, 93, 186, 186], transformer_embedding_dim=transformer_embedding_dim)
        self.cbam = CBAM(channels=512)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), # 7x7x512 -> 1x1x512
            nn.Flatten(), # 1x1x512 -> 512
            nn.Linear(in_features = 512, out_features = 256, bias = True),
            nn.BatchNorm1d(num_features = 256, eps = 2e-5),
            nn.ReLU(),
            nn.Linear(in_features = 256, out_features = num_classes, bias = True),
        )

    def forward(self, multiscale_features):
        x = self.fusion(multiscale_features)
        x = self.cbam(x)
        return self.head(x)


class RaceRecognitionSubnet(nn.Module):
    def __init__(self, num_classes = 7, transformer_embedding_dim = 96):
        super(RaceRecognitionSubnet, self).__init__()
        self.fusion = MultiScaleFusion(out_channels = [47, 93, 186, 186], transformer_embedding_dim=transformer_embedding_dim)
        self.cbam = CBAM(channels = 512)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), # 7x7x512 -> 1x1x512
            nn.Flatten(), # 1x1x512 -> 512
            nn.Linear(in_features = 512, out_features = 256, bias = True),
            nn.BatchNorm1d(num_features = 256, eps = 2e-5),
            nn.ReLU(),
            nn.Linear(in_features = 256, out_features = num_classes, bias = True),
        )
    
    def forward(self, multiscale_features):
        x = self.fusion(multiscale_features)
        x = self.cbam(x)
        return self.head(x)