import torch
import torch.nn as nn
import torch.nn.functional as F
import backbones.backbones as backbones
import multitask.face_recognition_heads as face_recognition_heads
from multitask.subnets import FaceRecognitionEmbeddingSubnet, GenderRecognitionSubnet, AgeEstimationSubnet, \
                              EmotionRecognitionSubnet


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
        self.pretrained_face_recognition_path = kwargs.get('pretrained_face_recognition_path')
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

        # Emotion Recognition
        self.emotion_recognition_subnet = EmotionRecognitionSubnet(
            transformer_embedding_dim = self.transformer_embedding_dim
        )

        # Age Estimation
        self.age_estimation_subnet = AgeEstimationSubnet(
            transformer_embedding_dim=self.transformer_embedding_dim
        )

        # Gender Recognition
        self.gender_recognition_subnet = GenderRecognitionSubnet(
            transformer_embedding_dim=self.transformer_embedding_dim
        )
    

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
        
        return (normalized_embedding, embedding_norm), emotion_output, age_output, gender_output

    def get_face_recognition_logits(self, normalized_embedding, embedding_norm, labels):
        return self.margin_head(normalized_embedding, embedding_norm, labels)