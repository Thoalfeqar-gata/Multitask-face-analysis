import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from multitask.framework1.cbam import CBAM
from backbones import backbones
from multitask import face_recognition_heads

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
            fused_output: 7x7x512 channels
    """
    def __init__(self, stage_out_channels=[50, 100, 150, 212], transformer_embedding_dim = 96):
        super(MultiScaleFusion, self).__init__()
        
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



##########################

#      Subnets

##########################

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
            nn.Linear(in_features = feature_embedding_dim, out_features=feature_embedding_dim, bias = False), # bias = False since it is followed by a BatchNorm1d, which removes the bias
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
    def __init__(self, min_age = 0, max_age = 101):
        super(AgeEstimationSubnet, self).__init__()
        self.num_classes = max_age - min_age + 1

        self.feature_fusion = MultiScaleFusion()
        self.head = nn.Sequential( # A simple distribution prediction head
            nn.AdaptiveAvgPool2d((1, 1)), # 7x7x512 ->1x1x512
            nn.Flatten(), # 1x1x512 -> 512
            nn.Linear(in_features = 512, out_features = 256, bias = False),
            nn.BatchNorm1d(num_features = 256, eps = 2e-5),
            nn.ReLU(inplace = True),
            nn.Dropout(0.2),
            nn.Linear(in_features = 256, out_features = self.num_classes, bias = True),
        )

    def forward(self, multiscale_features):
        x = self.feature_fusion(multiscale_features)
        return self.head(x)


class GenderRecognitionSubnet(nn.Module):
    def __init__(self):
        super(GenderRecognitionSubnet, self).__init__()
        
        self.feature_fusion = MultiScaleFusion()
        self.head = nn.Sequential( # A simple binary classification head
            nn.AdaptiveAvgPool2d((1, 1)), # 7x7x512 -> 1x1x512
            nn.Flatten(), # 1x1x512 -> 512
            nn.Linear(in_features = 512, out_features = 256, bias = False),
            nn.BatchNorm1d(num_features = 256, eps = 2e-5),
            nn.ReLU(inplace = True),
            nn.Dropout(0.2),
            nn.Linear(in_features = 256, out_features = 1, bias = True),
        )
    
    def forward(self, multiscale_features):
        x = self.feature_fusion(multiscale_features)
        return self.head(x)



class EmotionRecognitionSubnet(nn.Module):
    def __init__(self, num_classes = 7):
        super(EmotionRecognitionSubnet, self).__init__()

        self.feature_fusion = MultiScaleFusion()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), # 7x7x512 -> 1x1x512
            nn.Flatten(), # 1x1x512 -> 512
            nn.Linear(in_features = 512, out_features = 256, bias = False),
            nn.BatchNorm1d(num_features = 256, eps = 2e-5),
            nn.ReLU(inplace = True),
            nn.Dropout(0.2),
            nn.Linear(in_features = 256, out_features = num_classes, bias = True),
        )

    def forward(self, multiscale_features):
        x = self.feature_fusion(multiscale_features)
        return self.head(x)


class RaceRecognitionSubnet(nn.Module):
    def __init__(self, num_classes = 5):
        super(RaceRecognitionSubnet, self).__init__()

        self.feature_fusion = MultiScaleFusion()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), # 7x7x512 -> 1x1x512
            nn.Flatten(), # 1x1x512 -> 512
            nn.Linear(in_features = 512, out_features = 256, bias = False),
            nn.BatchNorm1d(num_features = 256, eps = 2e-5),
            nn.ReLU(inplace = True),
            nn.Dropout(0.2),
            nn.Linear(in_features = 256, out_features = num_classes, bias = True),
        )
    
    def forward(self, multiscale_features):
        x = self.feature_fusion(multiscale_features)
        return self.head(x)


class AttributeRecognitionSubnet(nn.Module):
    def __init__(self, num_classes = 40):
        super(AttributeRecognitionSubnet, self).__init__()

        self.feature_fusion = MultiScaleFusion()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), # 7x7x512 -> 1x1x512
            nn.Flatten(), # 1x1x512 -> 512
            nn.Linear(in_features = 512, out_features = 256, bias = False),
            nn.BatchNorm1d(num_features = 256, eps = 2e-5),
            nn.ReLU(inplace = True),
            nn.Dropout(0.2),
            nn.Linear(in_features = 256, out_features = num_classes, bias = True),
        )
    
    def forward(self, multiscale_features):
        x = self.feature_fusion(multiscale_features)
        return self.head(x)

class PoseEstimationSubnet(nn.Module):
    def __init__(self):
        super(PoseEstimationSubnet, self).__init__()
        self.feature_fusion = MultiScaleFusion()
        
        # Output 6 values for the Rotation Matrix (Gram-Schmidt representation)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 6) # <--- 6 Neurons for 6D rotation rep
        )

    def forward(self, multiscale_features):
        x = self.feature_fusion(multiscale_features)
        return self.head(x)


##########################

#      Complete model

##########################

class MultiTaskFaceAnalysisModel(nn.Module):
    def __init__(self, num_classes, **kwargs):
        """
            Args:
            kwargs: contains all the hyperparameters.
        """
        super().__init__()
        # Backbone args
        self.backbone_name = kwargs.get('backbone_name')
        self.pretrained_backbone_path = kwargs.get('pretrained_backbone_path')
        self.pretrained_face_recognition_path = kwargs.get('pretrained_face_recognition_subnet_path')
        self.pretrained_margin_head_path = kwargs.get('pretrained_margin_head_path')
        self.imagenet_pretrained = kwargs.get('pretrained')

        # Face recognition args
        self.head_type = kwargs.get('head_type')
        self.embedding_dim = kwargs.get('embedding_dim')
        self.num_classes = num_classes
        self.margin = kwargs.get('margin')
        self.scale = kwargs.get('scale')
        self.h = kwargs.get('h')
        self.t_alpha = kwargs.get('t_alpha')
        

        ##########
        # Backbone
        ##########

        # Obtain the backbone and load the weights if specified
        self.backbone = backbones.get_backbone(
            backbone_name = self.backbone_name, 
            imagenet_pretrained=self.imagenet_pretrained, 
            embedding_dim=self.embedding_dim
        )

        print(self.pretrained_backbone_path)
        if self.pretrained_backbone_path is not None:
            self.backbone.load_state_dict(torch.load(self.pretrained_backbone_path))
            print(f'Loaded pretrained backbone from {self.pretrained_backbone_path}.')
        

        ##########
        # Subnets
        ##########

        # Face Recognition

        if self.backbone_name in ['swin_b', 'swin_v2_b', 'davit_b']:
            self.feature_embedding_dim = 1024
            self.transformer_embedding_dim = 128
        else:
            self.feature_embedding_dim = 768
            self.transformer_embedding_dim = 96

        self.face_recognition_embedding_subnet = FaceRecognitionEmbeddingSubnet(
            feature_embedding_dim=self.feature_embedding_dim,
        )

        if self.pretrained_face_recognition_path:
            self.face_recognition_embedding_subnet.load_state_dict(torch.load(self.pretrained_face_recognition_path))
            print(f'Loaded pretrained face recognition subnet from {self.pretrained_face_recognition_path}')

        self.margin_head = face_recognition_heads.build_head(
            head_type = self.head_type,
            embedding_size = self.embedding_dim,
            classnum = self.num_classes,
            m = self.margin,
            s = self.scale,
            h = self.h,
            t_alpha = self.t_alpha
        )

        if self.pretrained_margin_head_path:
            self.margin_head.load_state_dict(torch.load(self.pretrained_margin_head_path))
            print(f'Loaded pretrained margin head from {self.pretrained_margin_head_path}\n')

        # Emotion Recognition
        self.emotion_recognition_subnet = EmotionRecognitionSubnet()

        # Age Estimation
        self.age_estimation_subnet = AgeEstimationSubnet()

        # Gender Recognition
        self.gender_recognition_subnet = GenderRecognitionSubnet()

        # Race Recognition
        self.race_recognition_subnet = RaceRecognitionSubnet()

        # Attribute Recognition
        self.attribute_recognition_subnet = AttributeRecognitionSubnet()

        # Pose estimation
        self.pose_estimation_subnet = PoseEstimationSubnet()
        

    def forward(self, x):
        
        multiscale_features = self.backbone(x)
        
        # Face recognition
        normalized_embedding, embedding_norm = self.face_recognition_embedding_subnet(multiscale_features)
        
        # Emotion Recognition
        emotion_output = self.emotion_recognition_subnet(multiscale_features)
        
        # Age Estimation
        age_output = self.age_estimation_subnet(multiscale_features)
        
        # Gender Recognition
        gender_output = self.gender_recognition_subnet(multiscale_features)

        # Race Recognition
        race_output = self.race_recognition_subnet(multiscale_features)

        # Attribute Recognition
        attribute_output = self.attribute_recognition_subnet(multiscale_features)

        # Pose estimation
        pose_output = self.pose_estimation_subnet(multiscale_features)
        
        return (normalized_embedding, embedding_norm), emotion_output, age_output, gender_output, race_output, attribute_output, pose_output

    def get_face_recognition_logits(self, normalized_embedding, embedding_norm, labels):
        return self.margin_head(normalized_embedding, embedding_norm, labels)