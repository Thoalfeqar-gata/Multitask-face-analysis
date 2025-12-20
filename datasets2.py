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

class FaceRecognitionClass(BaseDatasetClass):
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
        label = {
            'face recognition': label
        }
        return image, label

class Glint360k(FaceRecognitionClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'face recognition', 'glint360k')):
        super().__init__(dataset_dir)

class MS1MV2(FaceRecognitionClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'face recognition', 'ms1mv2')):
        super().__init__(dataset_dir)

class VGGFace(FaceRecognitionClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'face recognition', 'VGG-Face')):
        super().__init__(dataset_dir)

class CasiabWebFace(FaceRecognitionClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'face recognition', 'Casia webface')):
        super().__init__(dataset_dir)



###################################

#       Face Verification

###################################

class FaceVerificationClass(BaseDatasetClass):
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

class CFPFP(FaceVerificationClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'face recognition', 'cfp_fp')):
        super().__init__(dataset_dir)

class CFPFF(FaceVerificationClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'face recognition', 'cfp_ff')):
        super().__init__(dataset_dir)

class CALFW(FaceVerificationClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'face recognition', 'calfw')):
        super().__init__(dataset_dir)

class CPLFW(FaceVerificationClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'face recognition', 'cplfw')):
        super().__init__(dataset_dir)

class LFW(FaceVerificationClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'face recognition', 'lfw')):
        super().__init__(dataset_dir)


###################################

#       Emotion Recognition

###################################

class EmotionRecognitionClass(BaseDatasetClass):
    """
        The base class for all emotion recognition datasets.
    """
    def __init__(self, dataset_dir, subset = None):
        super().__init__(dataset_dir)
        if 'split' in self.labels_df.columns and subset != None: # will be used mainly with RAFDB
            self.labels_df = self.labels_df[self.labels_df['split'] == subset]
            self.labels_df.reset_index(drop = True, inplace = True)
    
    
    def __getitem__(self, idx):
        filename = self.labels_df['filename'][idx]
        image_path = os.path.join(self.images_dir, filename)
        image = decode_image(image_path, mode = torchvision.io.image.ImageReadMode.RGB)
        label = self.labels_df['label'][idx]
        label = {
            'emotion': label
        }
        return image, label

class AffectNet(EmotionRecognitionClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'emotion recognition', 'AffectNet')):
        super().__init__(dataset_dir)

class ExpressionInTheWild(EmotionRecognitionClass): # This dataset might be removed later since it is low quality and has many label mistakes
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'emotion recognition', 'Expression in the wild')):
        super().__init__(dataset_dir)

class RAFDB(EmotionRecognitionClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'emotion recognition', 'RAF_DB'), subset = 'train'):
        super().__init__(dataset_dir, subset = subset)


###################################

#       Age, Gender, and Race Recognition

###################################

class AgeDB(BaseDatasetClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'age gender and race estimation', 'AgeDB')):
        super().__init__(dataset_dir)
    
    def __getitem__(self, idx):
        filename = self.labels_df['filename'][idx]
        image = decode_image(os.path.join(self.images_dir, filename), mode = torchvision.io.image.ImageReadMode.RGB)
        age = self.labels_df['age'][idx]
        gender = self.labels_df['gender'][idx]
        label = {
            'age': age,
            'gender': gender
        }
        return image, label

class MORPH(BaseDatasetClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'age gender and race estimation', 'MORPH')):
        super().__init__(dataset_dir)

    def __getitem__(self, index):
        filename = self.labels_df['filename'][index]
        image = decode_image(os.path.join(self.images_dir, filename), mode = torchvision.io.image.ImageReadMode.RGB)
        age = self.labels_df['age'][index]
        gender = self.labels_df['gender'][index]
        label = {
            'age': age,
            'gender': gender
        }
        return image, label


class UTKFace(BaseDatasetClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'age gender and race estimation', 'UTKFace')):
        super().__init__(dataset_dir)
    
    def __getitem__(self, idx):
        filename = self.labels_df['filename'][idx]
        image = decode_image(os.path.join(self.images_dir, filename), mode = torchvision.io.image.ImageReadMode.RGB)
        age = self.labels_df['age'][idx]
        gender = self.labels_df['gender'][idx]
        race = self.labels_df['race'][idx]
        label = {
            'age': age,
            'gender': gender,
            'race': race
        }
        return image, label


class FairFace(BaseDatasetClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'age gender and race estimation', 'FairFace')):
        super().__init__(dataset_dir)
    
    def __getitem__(self, idx):
        filename = self.labels_df['filename'][idx]
        image = decode_image(os.path.join(self.images_dir, filename), mode = torchvision.io.image.ImageReadMode.RGB)
        gender = self.labels_df['gender'][idx]
        race = self.labels_df['race'][idx]

        label = {
            'gender': gender,
            'race': race
        }
        return image, label