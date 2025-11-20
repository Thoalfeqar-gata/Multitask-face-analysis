import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import backbones.backbones as backbones
import datasets
import multitask.face_recognition_heads as face_recognition_heads
import eval, os
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import lr_scheduler
from augmenter import Augmenter
from multitask.subnets import FaceRecognitionEmbeddingSubnet, GenderRecognitionSubnet, AgeEstimationSubnet, \
                              EmotionRecognitionSubnet
from datasets import MultiTaskDataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


torch.set_float32_matmul_precision('medium')


class MultiTaskFaceAnalysisModel(nn.Module):
    def __init__(self, **kwargs):
        """
            Args:
            kwargs: contains all the hyperparameters.
        """
        super().__init__()
        # Backbone args
        self.backbone_name = kwargs.get('backbone_name')
        self.pretrained_backbone_path = kwargs.get('pretrained_backbone_path')
        self.imagenet_pretrained = kwargs.get('pretrained')

        # Face recognition args
        self.head_type = kwargs.get('head_type')
        self.embedding_dim = kwargs.get('embedding_dim')
        self.num_classes = kwargs.get('num_classes')
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

        if self.pretrained_backbone_path is not None:
            self.backbone.load_state_dict(torch.load(self.pretrained_backbone_path))
        

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

def main(**kwargs):

    num_of_tasks = 3

    # data preparation
    face_recognition_dataset = datasets.CasiaWebFace_Dataset()
    face_recognition_dataset.discard_classes(kwargs.get('min_num_images_per_class'))


    # RAF Dataset
    raf_dataset = datasets.RAF_Dataset()
    raf_train_size = int(0.8 * len(raf_dataset))
    raf_test_size = len(raf_dataset) - raf_train_size
    raf_train_dataset, raf_test_dataset = torch.utils.data.random_split(raf_dataset, [raf_train_size, raf_test_size])


    # AgeDB Dataset
    agedb_dataset = datasets.AgeDB_Dataset()
    agedb_train_size = int(0.8 * len(agedb_dataset))
    agedb_test_size = len(agedb_dataset) - agedb_train_size
    agedb_train_dataset, agedb_test_dataset = torch.utils.data.random_split(agedb_dataset, [agedb_train_size, agedb_test_size])



    train_dataset = datasets.ZippedDataset([
        face_recognition_dataset,
        raf_dataset,
        agedb_dataset
    ])

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = kwargs.get('batch_size') // num_of_tasks,
        pin_memory = True,
        num_workers = kwargs.get('num_workers')
    )

    


    





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Multitask training test')

    # Model Hyperparameters
    parser.add_argument('--backbone_name', type=str, default='swin_t', help='Backbone model name')
    parser.add_argument('--pretrained', type = int, default = 0, help='Use pretrained weights 0 = False, 1 = True')
    parser.add_argument('--head_type', type=str, default='adaface', help='Head name (e.g., arcface, cosface, adaface)')
    parser.add_argument('--embedding_dim', type=int, default=512, help='Embedding dimension')
    parser.add_argument('--optimizer', type=str, default='adamw', help='Optimizer name (can be either adamw or sgd)')
    parser.add_argument('--weight_decay', type = float, default = 5e-4, help = 'Weight decay for the optimizer')
    parser.add_argument('--scheduler', type=str, default='cosine', help='Learning rate scheduler name (can be either cosine or multistep)')
    parser.add_argument('--scheduler_milestones', type=int, nargs='+', default=[20, 30], help='Milestones for the multistep scheduler (not needed for cosine)')
    parser.add_argument('--start_factor', type=float, default=0.1, help='Start factor for the linear warmup learning rate scheduler')
    parser.add_argument('--min_lr', type=float, default=5e-6, help='Minimum learning rate for the cosine scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of warmup epochs')
    parser.add_argument('--margin', type=float, default=0.5, help='Margin for the loss function')
    parser.add_argument('--scale', type=float, default=64.0, help='Scale for the loss function')
    parser.add_argument('--h', type=float, default=0.333, help='h parameter for AdaFace loss')
    parser.add_argument('--t_alpha', type=float, default=1.0, help='t_alpha parameter for AdaFace loss')

    # Training Hyperparameters
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=40, help='Number of training epochs')
    parser.add_argument('--precision', type=str, default = '16-mixed', help='Use mixed precision training')

    # Data
    parser.add_argument('--min_num_images_per_class', type = int, default = 20, help = "The minimum number of images per class, classes less than this number are discarded.")
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')

    # System
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Path to a checkpoint to resume training from')

    args = parser.parse_args()
    main(**vars(args))
