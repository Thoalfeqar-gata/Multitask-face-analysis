import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import pandas as pd
from torchvision.io import decode_image



class BaseDatasetClass(torch.utils.data.Dataset):
    """
        The base dataset class for all datasets
    """

    def __init__(self, dataset_dir):
        super().__init__()

        self.images_dir = os.path.join(dataset_dir, 'Images')
        self.labels_df = pd.read_csv(os.path.join(dataset_dir, 'labels.csv'))
    
    def __len__(self):
        return len(self.labels_df)


###################################

#       Face Recognition

###################################

class BaseFaceRecognitionClass(BaseDatasetClass):
    """
        The base class for all face recognition datasets.
    """

    def __init__(self, dataset_dir):
        super().__init__(dataset_dir)
    
    def number_of_classes(self):
        return len(self.labels_df['label'].unique())
    
    def __getitem__(self, idx):
        filename = self.labels_df['filename'][idx]
        image_path = os.path.join(self.images_dir, filename)
        image = decode_image(image_path, mode = torchvision.io.image.ImageReadMode.RGB)
        label = self.labels_df['label'][idx]

        return image, label

class Glint360k(BaseFaceRecognitionClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'face recognition', 'glint360k')):
        super().__init__(dataset_dir)

class MS1MV2(BaseFaceRecognitionClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'face recognition', 'ms1mv2')):
        super().__init__(dataset_dir)

class VGGFace(BaseFaceRecognitionClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'face recognition', 'VGG-Face')):
        super().__init__(dataset_dir)

class CasiabWebFace(BaseFaceRecognitionClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'face recognition', 'Casia webface')):
        super().__init__(dataset_dir)



###################################

#       Face Verification

###################################

class BaseFaceVerificationClass(BaseDatasetClass):
    """
        The base class for all face verification datasets.
    """
    def __init__(self, dataset_dir):
        super().__init__(dataset_dir)
    
    def __getitem__(self, idx):
        image1 = self.labels_df['image1'][idx]
        image2 = self.labels_df['image2'][idx]
        same = self.labels_df['same'][idx]

        image1 = decode_image(os.path.join(self.images_dir, image1), mode = torchvision.io.image.ImageReadMode.RGB)
        image2 = decode_image(os.path.join(self.images_dir, image2), mode = torchvision.io.image.ImageReadMode.RGB)

        return (image1, image2), same

class CFPFP(BaseFaceVerificationClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'face recognition', 'cfp_fp')):
        super().__init__(dataset_dir)

class CFPFF(BaseFaceVerificationClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'face recognition', 'cfp_ff')):
        super().__init__(dataset_dir)

class CALFW(BaseFaceVerificationClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'face recognition', 'calfw')):
        super().__init__(dataset_dir)

class CPLFW(BaseFaceVerificationClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'face recognition', 'cplfw')):
        super().__init__(dataset_dir)

class LFW(BaseFaceVerificationClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'face recognition', 'lfw')):
        super().__init__(dataset_dir)