import torch
import torch.nn as nn
import torch.nn.functional as F
from multitask.framework1.multiscale_fusion import StandardMultiScaleFusion, LightMultiScaleFusion


##########################

#      Subnets

##########################

class FaceRecognitionEmbeddingSubnet(nn.Module):
    def __init__(self, feature_embedding_dim=768, embedding_dim=512):
        super(FaceRecognitionEmbeddingSubnet, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.feature_head = nn.Sequential(
            nn.Linear(feature_embedding_dim, feature_embedding_dim, bias=False),
            nn.BatchNorm1d(feature_embedding_dim, eps=2e-5),
            nn.Linear(feature_embedding_dim, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim, eps=2e-5) 
        )

    def forward(self, multiscale_features):
        x = multiscale_features[-1]  # Shape: (B, 49, 768)
        
        x = self.avgpool(x.permute(0, 2, 1)) # Shape: (B, 768, 1)
        x = self.flatten(x)  # Shape: (B, 768)
        embedding = self.feature_head(x)
        
        embedding_norm = torch.norm(embedding, p=2, dim=1, keepdim=True)
        normalized_embedding = torch.div(embedding, embedding_norm)
        
        return normalized_embedding, embedding_norm


class BasicHead(nn.Module):
    """
        This module defines the basic head for each subnet. 
        The subnet's job will be to only turn the features extracted by this module into
        usable logits for prediction.
    """
    def __init__(self, multiscale_fusion_type = 'standard', out_channel_dim = 512):
        super(BasicHead, self).__init__()

        if multiscale_fusion_type == 'standard':
            self.feature_fusion = StandardMultiScaleFusion(out_channel_dim = out_channel_dim)
        else:
            self.feature_fusion = LightMultiScaleFusion(out_channel_dim = out_channel_dim)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), # 7x7xout_channel_dim -> 1x1xout_channel_dim
            nn.Flatten(), # 1x1xout_channel_dim -> out_channel_dim
            
            nn.Linear(in_features = out_channel_dim, out_features = out_channel_dim, bias = False),
            nn.BatchNorm1d(num_features = out_channel_dim, eps = 2e-5),
            nn.SiLU(inplace = True),
            nn.Dropout(0.2),

            nn.Linear(in_features = out_channel_dim, out_features = out_channel_dim, bias = False),
            nn.BatchNorm1d(num_features = out_channel_dim, eps = 2e-5),
            nn.SiLU(inplace = True),
            nn.Dropout(0.2),
        )
    
    def forward(self, multiscale_features):
        return self.head(self.feature_fusion(multiscale_features))



class AgeEstimationSubnet(nn.Module):
    def __init__(self, min_age = 0, max_age = 101, multiscale_fusion_type = 'standard'):
        super(AgeEstimationSubnet, self).__init__()
        self.num_classes = max_age - min_age + 1

        self.basic_head = BasicHead(multiscale_fusion_type = multiscale_fusion_type)

        self.head = nn.Sequential(
            self.basic_head,
            nn.Linear(in_features = 512, out_features = self.num_classes, bias = True),
        )

    def forward(self, multiscale_features):
        return self.head(multiscale_features)


class GenderRecognitionSubnet(nn.Module):
    def __init__(self, multiscale_fusion_type = 'standard'):
        super(GenderRecognitionSubnet, self).__init__()
        
        self.basic_head = BasicHead(multiscale_fusion_type = multiscale_fusion_type)

        self.head = nn.Sequential( # A simple binary classification head
            self.basic_head,
            nn.Linear(in_features = 512, out_features = 1, bias = True),
        )
    
    def forward(self, multiscale_features):
        return self.head(multiscale_features)



class EmotionRecognitionSubnet(nn.Module):
    def __init__(self, num_classes = 7, multiscale_fusion_type = 'standard'):
        super(EmotionRecognitionSubnet, self).__init__()

        self.basic_head = BasicHead(multiscale_fusion_type = multiscale_fusion_type)

        self.head = nn.Sequential(
            self.basic_head,
            nn.Linear(in_features = 512, out_features = num_classes, bias = True),
        )

    def forward(self, multiscale_features):
        return self.head(multiscale_features)


class RaceRecognitionSubnet(nn.Module):
    def __init__(self, num_classes = 5, multiscale_fusion_type = 'standard'):
        super(RaceRecognitionSubnet, self).__init__()

        self.basic_head = BasicHead(multiscale_fusion_type = multiscale_fusion_type)

        self.head = nn.Sequential(
            self.basic_head,
            nn.Linear(in_features = 512, out_features = num_classes, bias = True),
        )
    
    def forward(self, multiscale_features):
        return self.head(multiscale_features)



class AttributeRecognitionSubnet(nn.Module):
    def __init__(self, multiscale_fusion_type='light'):
        super(AttributeRecognitionSubnet, self).__init__()

        # Define Groups
        self.attribute_groups = {
            'mouth' : ['5_o_Clock_Shadow', 'Big_Lips', 'Mouth_Slightly_Open', 'Mustache', 'Wearing_Lipstick', 'No_Beard'],
            'ear' : ['Wearing_Earrings'],
            'lower_face': ['Double_Chin', 'Goatee', 'Wearing_Necklace', 'Wearing_Necktie'],
            'cheeks' : ['High_Cheekbones', 'Rosy_Cheeks', 'Sideburns'],
            'nose' : ['Big_Nose', 'Pointy_Nose'],
            'eyes' : ['Arched_Eyebrows', 'Bags_Under_Eyes', 'Bushy_Eyebrows', 'Narrow_Eyes', 'Eyeglasses'],
            'hair' : ['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair', 'Receding_Hairline', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Hat'],
            'object' : ['Attractive', 'Blurry', 'Chubby', 'Heavy_Makeup', 'Male', 'Oval_Face', 'Pale_Skin', 'Smiling', 'Young']
        }
        
        self.channel_dim = 512
        self.heads = nn.ModuleList()
        self.classifiers = nn.ModuleList()
        
        for group_name in self.attribute_groups.keys():
            attributes = self.attribute_groups[group_name]
            num_classes = len(attributes) 
            
            subnet_head = BasicHead(multiscale_fusion_type=multiscale_fusion_type, out_channel_dim=self.channel_dim)
            self.heads.append(subnet_head)
            
            classifier = nn.Linear(self.channel_dim, num_classes)
            self.classifiers.append(classifier)

    def forward(self, multiscale_features):
        outputs = []
        
        for head, classifier in zip(self.heads, self.classifiers):
            group_embedding = head(multiscale_features)
            logits = classifier(group_embedding)
            outputs.append(logits)
        
        return torch.cat(outputs, dim=1)




class PoseEstimationSubnet(nn.Module):
    def __init__(self, multiscale_fusion_type = 'standard'):
        super(PoseEstimationSubnet, self).__init__()

        self.basic_head = BasicHead(multiscale_fusion_type = multiscale_fusion_type)

        # Output 6 values for the Rotation Matrix (Gram-Schmidt representation)
        self.head = nn.Sequential(
            self.basic_head,
            nn.Linear(in_features = 512, out_features = 6) # <--- 6 Neurons for 6D rotation rep
        )

    def forward(self, multiscale_features):
        return self.head(multiscale_features)