import argparse
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import backbones.backbones as backbones
import datasets
import multitask.face_recognition_heads as face_recognition_heads
import eval, os
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.optim import lr_scheduler
from lightning.pytorch import loggers 
from augmenter import Augmenter
from multitask.subnets import FaceRecognitionSubnet
from multitask.fusion import MultiScaleFusion


torch.set_float32_matmul_precision('medium')


"""
    To-Do: Complete the implementation below
"""

class MultiTaskFaceAnalysisModel(pl.LightningModule):
    def __init__(self, **kwargs):
        """
            Args:
            kwargs: contains all the hyperparameters.
        """
        super().__init__()
        self.backbone_name = kwargs.get('backbone_name')
        self.pretrained_backbone_path = kwargs.get('pretrained_backbone_path')
        self.save_hyperparameters()
