import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FaceRecognitionSubnet(nn.Module):
    def __init__(
            self,
            feature_embedding_dim = 768, 
            embedding_dim = 512
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
    

    def forward(self, multiscale_features):
        x = multiscale_features[-1] #obtain the feature vector from the last stage, which has a shape of 49x768
        x = self.norm_layer(x) # B, 49, 768 
        x = self.avgpool(x.transpose(1, 2)) # B, 768, 1
        x = torch.flatten(x, 1) # B, 768
        embedding = self.feature_head(x)
        embedding_norm = torch.norm(embedding, p=2, dim=1, keepdim = True)
        normalized_embedding = torch.div(embedding, embedding_norm)
        
        return normalized_embedding, embedding_norm

    

class EmotionRecognitionSubnet(nn.Module):
    def __init__(self, num_classes = 7, dropout_rate = 0.25, embedding_dim = 512):
        super(EmotionRecognitionSubnet, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(in_features = 4*4*512, out_features = embedding_dim, bias = True)
        self.fc2 = nn.Linear(in_features = embedding_dim, out_features = num_classes, bias = True)


    def forward(self, x): # x represents the fused features
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x