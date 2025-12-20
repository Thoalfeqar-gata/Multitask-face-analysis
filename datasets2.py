import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import pandas as pd
from torchvision.io import decode_image
from torch.utils.data import ConcatDataset
def get_default_labels():
    return {
        'face_recognition' : -1,
        'emotion' : -1,
        'age' : -1,
        'gender' : -1,
        'race' : -1
    }

def create_balanced_dataset(datasets_list):
    """
    Repeats smaller datasets so that their size roughly matches 
    the size of the largest dataset in the list.
    """
    max_size = max([len(ds) for ds in datasets_list])
    print(f"Largest dataset size: {max_size}")

    balanced_list = []
    
    for ds in datasets_list:
        ds_len = len(ds)
        
        repeat_factor = int(max_size / ds_len)
        if repeat_factor < 1: 
            repeat_factor = 1
            
        print(f"Dataset {type(ds).__name__}: Original={ds_len}, Repeating {repeat_factor} times.")
        
        balanced_list.extend([ds] * repeat_factor)
        
    return ConcatDataset(balanced_list)


class BaseDatasetClass(torch.utils.data.Dataset):
    """
        The base dataset class for all datasets
    """

    def __init__(self, dataset_dir, transform = None):
        super().__init__()

        self.images_dir = os.path.join(dataset_dir, 'Images')
        self.labels_df = pd.read_csv(os.path.join(dataset_dir, 'labels.csv'))
        self.transform = transform
    
    def __len__(self):
        return len(self.labels_df)


###################################

#       Face Recognition

###################################

class FaceRecognitionClass(BaseDatasetClass):
    """
        The base class for all face recognition datasets.
    """

    def __init__(self, dataset_dir, transform = None):
        super().__init__(dataset_dir, transform)
    
    def number_of_classes(self):
        return len(self.labels_df['label'].unique())
    
    def __getitem__(self, idx):
        filename = self.labels_df['filename'][idx]
        image_path = os.path.join(self.images_dir, filename)
        image = decode_image(image_path, mode = torchvision.io.image.ImageReadMode.RGB)
        
        if self.transform:
            image = self.transform(image)

        label = get_default_labels()
        label['face_recognition'] = self.labels_df['label'][idx]

        return image, label

class Glint360k(FaceRecognitionClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'face recognition', 'glint360k'), transform = None):
        super().__init__(dataset_dir, transform)

class MS1MV2(FaceRecognitionClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'face recognition', 'ms1mv2'), transform = None):
        super().__init__(dataset_dir, transform)

class VGGFace(FaceRecognitionClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'face recognition', 'VGG-Face'), transform = None):
        super().__init__(dataset_dir, transform)

class CasiabWebFace(FaceRecognitionClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'face recognition', 'Casia webface'), transform = None):
        super().__init__(dataset_dir, transform)



###################################

#       Face Verification

###################################

class FaceVerificationClass(BaseDatasetClass):
    """
        The base class for all face verification datasets.
    """
    def __init__(self, dataset_dir, transform = None):
        super().__init__(dataset_dir, transform)
    
    def __getitem__(self, idx):
        image1 = self.labels_df['image1'][idx]
        image2 = self.labels_df['image2'][idx]
        same = self.labels_df['same'][idx]

        image1 = decode_image(os.path.join(self.images_dir, image1), mode = torchvision.io.image.ImageReadMode.RGB)
        image2 = decode_image(os.path.join(self.images_dir, image2), mode = torchvision.io.image.ImageReadMode.RGB)

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return (image1, image2), same

class CFPFP(FaceVerificationClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'face recognition', 'cfp_fp'), transform = None):
        super().__init__(dataset_dir, transform)

class CFPFF(FaceVerificationClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'face recognition', 'cfp_ff'), transform = None):
        super().__init__(dataset_dir, transform)

class CALFW(FaceVerificationClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'face recognition', 'calfw'), transform = None):
        super().__init__(dataset_dir, transform)

class CPLFW(FaceVerificationClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'face recognition', 'cplfw'), transform = None):
        super().__init__(dataset_dir, transform)

class LFW(FaceVerificationClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'face recognition', 'lfw'), transform = None):
        super().__init__(dataset_dir, transform)

class AgeDB30(FaceVerificationClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'face recognition', 'agedb30'), transform = None):
        super().__init__(dataset_dir, transform)

class VGG2FP(FaceVerificationClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'face recognition', 'vgg2_fp'), transform = None):
        super().__init__(dataset_dir, transform)



###################################

#       Emotion Recognition

###################################

class EmotionRecognitionClass(BaseDatasetClass):
    """
        The base class for all emotion recognition datasets.
    """
    def __init__(self, dataset_dir, subset = None, transform = None):
        super().__init__(dataset_dir, transform)
        if 'split' in self.labels_df.columns and subset != None: # will be used mainly with RAFDB
            self.labels_df = self.labels_df[self.labels_df['split'] == subset]
            self.labels_df.reset_index(drop = True, inplace = True)
    
    
    def __getitem__(self, idx):
        filename = self.labels_df['filename'][idx]
        image_path = os.path.join(self.images_dir, filename)
        image = decode_image(image_path, mode = torchvision.io.image.ImageReadMode.RGB)

        if self.transform:
            image = self.transform(image)

        label = get_default_labels()
        label['emotion'] = self.labels_df['label'][idx]

        return image, label

class AffectNet(EmotionRecognitionClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'emotion recognition', 'AffectNet'), transform = None):
        super().__init__(dataset_dir, transform)

class ExpressionInTheWild(EmotionRecognitionClass): # This dataset might be removed later since it is low quality and has many label mistakes
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'emotion recognition', 'Expression in the wild'), transform = None):
        super().__init__(dataset_dir, transform)

class RAFDB(EmotionRecognitionClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'emotion recognition', 'RAF_DB'), subset = 'train', transform = None):
        super().__init__(dataset_dir, subset = subset, transform = transform)


###################################

#       Age, Gender, and Race Recognition

###################################

class AgeDB(BaseDatasetClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'age gender and race estimation', 'AgeDB'), transform = None):
        super().__init__(dataset_dir, transform)
    
    def __getitem__(self, idx):
        filename = self.labels_df['filename'][idx]
        image = decode_image(os.path.join(self.images_dir, filename), mode = torchvision.io.image.ImageReadMode.RGB)

        if self.transform:
            image = self.transform(image)
        
        age = self.labels_df['age'][idx]
        gender = self.labels_df['gender'][idx]

        label = get_default_labels()
        label['age'] = age
        label['gender'] = gender

        return image, label

class MORPH(BaseDatasetClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'age gender and race estimation', 'MORPH'), transform = None):
        super().__init__(dataset_dir, transform)

    def __getitem__(self, index):
        filename = self.labels_df['filename'][index]
        image = decode_image(os.path.join(self.images_dir, filename), mode = torchvision.io.image.ImageReadMode.RGB)

        if self.transform:
            image = self.transform(image)

        age = self.labels_df['age'][index]
        gender = self.labels_df['gender'][index]
        label = get_default_labels()
        label['age'] = age
        label['gender'] = gender

        return image, label


class UTKFace(BaseDatasetClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'age gender and race estimation', 'UTKFace'), transform = None):
        super().__init__(dataset_dir, transform)
    
    def __getitem__(self, idx):
        filename = self.labels_df['filename'][idx]
        image = decode_image(os.path.join(self.images_dir, filename), mode = torchvision.io.image.ImageReadMode.RGB)
        
        if self.transform:
            image = self.transform(image)

        age = self.labels_df['age'][idx]
        gender = self.labels_df['gender'][idx]
        race = self.labels_df['race'][idx]
        label = get_default_labels()
        label['age'] = age
        label['gender'] = gender
        label['race'] = race

        return image, label



class FairFace(BaseDatasetClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'age gender and race estimation', 'FairFace'), transform = None):
        super().__init__(dataset_dir, transform)
    
    def __getitem__(self, idx):
        filename = self.labels_df['filename'][idx]
        image = decode_image(os.path.join(self.images_dir, filename), mode = torchvision.io.image.ImageReadMode.RGB)

        if self.transform:
            image = self.transform(image)

        gender = self.labels_df['gender'][idx]
        race = self.labels_df['race'][idx]
        label = get_default_labels()
        label['gender'] = gender
        label['race'] = race

        return image, label 