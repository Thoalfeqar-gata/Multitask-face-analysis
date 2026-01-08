import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import pandas as pd
from torchvision.io import decode_image
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler


def get_balanced_loader(datasets_list, batch_size, num_workers, epoch_size=None):
    """
    Args:
        datasets_list (list): A list of the datasets to be combined and turned into a dataloader.
        epoch_size (int): How many images to sample per epoch. 
                          
    """
    # 1. Concatenate naturally (NO repetition)
    unified_dataset = ConcatDataset(datasets_list)
    
    # 2. Calculate weights for each dataset
    dataset_counts = [len(ds) for ds in datasets_list]
    total_count = sum(dataset_counts)
    
    # Calculate weight per sample: 1.0 / count
    # This ensures that (count * weight) is constant for all datasets
    dataset_weights = [1.0 / count for count in dataset_counts]
    
    print("Dataset weights:", dataset_weights)
    
    # 3. Assign a weight to EVERY sample in the unified dataset
    # This creates a long vector of weights matching the unified_dataset length
    sample_weights = []
    for i, count in enumerate(dataset_counts):
        # Extend the list by the number of items in that dataset
        sample_weights.extend([dataset_weights[i]] * count)
        
    sample_weights = torch.DoubleTensor(sample_weights)
    
    # 4. Define Epoch Size
    # If we don't set this, an 'epoch' is just the sum of all raw lengths (~6.5M)
    # But we likely want to define a 'Virtual Epoch' to check validation often.
    if epoch_size is None:
        epoch_size = len(unified_dataset)
        
    # 5. Create the Sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=epoch_size,
        replacement=True # Crucial: Allows picking samples multiple times
    )
    
    # 6. Create Loader
    # Note: shuffle must be False when using a sampler! It already shuffles everything nicely.
    loader = DataLoader(
        unified_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler, 
        pin_memory=True
    )
    
    return loader



class BaseDatasetClass(torch.utils.data.Dataset):
    """
        The base dataset class for all datasets
    """

    def __init__(self, dataset_dir, transform = None, return_name = False):
        super().__init__()

        self.images_dir = os.path.join(dataset_dir, 'Images')
        self.labels_df = pd.read_csv(os.path.join(dataset_dir, 'labels.csv'))
        self.transform = transform
        self.return_name = return_name

    def __getitem__(self, idx):
        raise NotImplementedError
    
    def __len__(self):
        return len(self.labels_df)

    # used for getting the default labels for each task and, optionally, the name of the database
    def get_default_labels(self):
        label = { # the default label is -1 because putting someting like None won't work in the torch.utils.data.DataLoader, and using something like float('nan') won't work with integer labels.
            'face_recognition' : -1,
            'emotion' : -1,
            'age' : -1,
            'gender' : -1,
            'race' : -1,
            'attributes' : torch.ones(40, dtype = torch.int32) * -1, # place -1 for all the 40 attributes
            'pose' : torch.ones(3, dtype = torch.float32) * -999, # place -999 for all the 3 angles. We can't place -1 because it could be a valid angle.
        }
        if self.return_name:
            label['dataset_name'] = self.__class__.__name__
        return label



###################################

#       Face Recognition

###################################

class FaceRecognitionClass(BaseDatasetClass):
    """
        The base class for all face recognition datasets.
    """

    def __init__(self, dataset_dir, transform = None, **kwargs):
        super().__init__(dataset_dir, transform, **kwargs)
    
    def number_of_classes(self):
        return len(self.labels_df['label'].unique())
    
    def __getitem__(self, idx):
        filename = self.labels_df['filename'][idx]
        image_path = os.path.join(self.images_dir, filename)
        image = decode_image(image_path, mode = torchvision.io.image.ImageReadMode.RGB)
        
        if self.transform:
            image = self.transform(image)

        label = self.get_default_labels()
        label['face_recognition'] = self.labels_df['label'][idx]

        return image, label

class Glint360k(FaceRecognitionClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'face recognition', 'glint360k'), transform = None, **kwargs):
        super().__init__(dataset_dir, transform, **kwargs)

class MS1MV2(FaceRecognitionClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'face recognition', 'ms1mv2'), transform = None, **kwargs):
        super().__init__(dataset_dir, transform, **kwargs)

class VGGFace(FaceRecognitionClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'face recognition', 'VGG-Face'), transform = None, **kwargs):
        super().__init__(dataset_dir, transform, **kwargs)

class CasiaWebFace(FaceRecognitionClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'face recognition', 'Casia webface'), transform = None, **kwargs):
        super().__init__(dataset_dir, transform, **kwargs)



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
    def __init__(self, dataset_dir, subset = None, transform = None, **kwargs):
        super().__init__(dataset_dir, transform, **kwargs)
        if 'split' in self.labels_df.columns and subset != None: # will be used mainly with RAFDB
            self.labels_df = self.labels_df[self.labels_df['split'] == subset]
            self.labels_df.reset_index(drop = True, inplace = True)
    
    
    def __getitem__(self, idx):
        filename = self.labels_df['filename'][idx]
        image_path = os.path.join(self.images_dir, filename)
        image = decode_image(image_path, mode = torchvision.io.image.ImageReadMode.RGB)

        if self.transform:
            image = self.transform(image)

        label = self.get_default_labels()
        label['emotion'] = self.labels_df['label'][idx]

        return image, label

class AffectNet(EmotionRecognitionClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'emotion recognition', 'AffectNet'), subset = 'train', transform = None, **kwargs):
        super().__init__(dataset_dir, subset = subset, transform = transform, **kwargs)

class RAFDB(EmotionRecognitionClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'emotion recognition', 'RAF_DB'), subset = 'train', transform = None, **kwargs):
        super().__init__(dataset_dir, subset = subset, transform = transform, **kwargs)


###################################

#       Age, Gender, and Race Recognition

###################################

class AgeDB(BaseDatasetClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'age gender and race estimation', 'AgeDB'), transform = None, **kwargs):
        super().__init__(dataset_dir, transform, **kwargs)
    
    def __getitem__(self, idx):
        filename = self.labels_df['filename'][idx]
        image = decode_image(os.path.join(self.images_dir, filename), mode = torchvision.io.image.ImageReadMode.RGB)

        if self.transform:
            image = self.transform(image)
        
        age = self.labels_df['age'][idx]
        gender = self.labels_df['gender'][idx]

        label = self.get_default_labels()
        label['age'] = age
        label['gender'] = gender

        return image, label

class MORPH(BaseDatasetClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'age gender and race estimation', 'MORPH'), transform = None, subset = 'train', **kwargs):
        super().__init__(dataset_dir, transform, **kwargs)
        if subset != None: # Can be 'train', 'test' or 'validation'
            self.labels_df = self.labels_df[self.labels_df['split'] == subset]
            self.labels_df.reset_index(drop = True, inplace = True)

    def __getitem__(self, index):
        filename = self.labels_df['filename'][index]
        image = decode_image(os.path.join(self.images_dir, filename), mode = torchvision.io.image.ImageReadMode.RGB)

        if self.transform:
            image = self.transform(image)

        age = self.labels_df['age'][index]
        gender = self.labels_df['gender'][index]
        label = self.get_default_labels()
        label['age'] = age
        label['gender'] = gender

        return image, label


class UTKFace(BaseDatasetClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'age gender and race estimation', 'UTKFace'), transform = None, subset = None, **kwargs):
        super().__init__(dataset_dir, transform, **kwargs)
        if subset != None: # Can be either 'train' or 'test'
            self.labels_df = self.labels_df[self.labels_df['split'] == subset]
            self.labels_df.reset_index(drop = True, inplace = True)
    
    def __getitem__(self, idx):
        filename = self.labels_df['filename'][idx]
        image = decode_image(os.path.join(self.images_dir, filename), mode = torchvision.io.image.ImageReadMode.RGB)
        
        if self.transform:
            image = self.transform(image)

        age = self.labels_df['age'][idx]
        gender = self.labels_df['gender'][idx]
        race = self.labels_df['race'][idx]
        label = self.get_default_labels()
        label['age'] = age
        label['gender'] = gender
        label['race'] = race

        return image, label



class FairFace(BaseDatasetClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'age gender and race estimation', 'FairFace'), transform = None, subset = None, **kwargs):
        super().__init__(dataset_dir, transform, **kwargs)
        if subset != None: # Can be either 'train' or 'test'
            self.labels_df = self.labels_df[self.labels_df['split'] == subset]
            self.labels_df.reset_index(drop = True, inplace = True)

    def __getitem__(self, idx):
        filename = self.labels_df['filename'][idx]
        image = decode_image(os.path.join(self.images_dir, filename), mode = torchvision.io.image.ImageReadMode.RGB)

        if self.transform:
            image = self.transform(image)

        gender = self.labels_df['gender'][idx]
        race = self.labels_df['race'][idx]
        label = self.get_default_labels()
        label['gender'] = gender
        label['race'] = race

        return image, label 
    

###################################

#       Attribute recognition

###################################

class CelebA(BaseDatasetClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'attribute recognition', 'CelebA'), transform = None, subset = None, **kwargs):
        super().__init__(dataset_dir, transform, **kwargs)
        if subset != None: # Can be 'train', 'test', or 'validation'
            self.labels_df = self.labels_df[self.labels_df['split'] == subset]
            self.labels_df.reset_index(drop = True, inplace = True)

    
    def __getitem__(self, idx):
        filename = self.labels_df['filename'][idx]
        image = decode_image(os.path.join(self.images_dir, filename), mode = torchvision.io.image.ImageReadMode.RGB)

        if self.transform:
            image = self.transform(image)

        label = self.get_default_labels()
        label['attributes'] = torch.tensor(np.array(self.labels_df.iloc[idx, 1:-1].values, dtype = np.int8), dtype = torch.int8)

        return image, label



###################################

#       Attribute recognition

###################################

class W300LP(BaseDatasetClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'head pose estimation', '300W_LP'), transform = None, **kwargs):
        super().__init__(dataset_dir, transform, **kwargs)
    
    def __getitem__(self, idx):
        filename = self.labels_df['filename'][idx]
        image = decode_image(os.path.join(self.images_dir, filename), mode = torchvision.io.image.ImageReadMode.RGB)

        if self.transform:
            image = self.transform(image)

        label = self.get_default_labels()
        label['pose'] = torch.tensor(np.array(self.labels_df.iloc[idx, 1:].values, dtype = np.float32), dtype = torch.float32)


        return image, label


class BIWI(BaseDatasetClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'head pose estimation', 'BIWI'), transform = None, **kwargs):
        super().__init__(dataset_dir, transform, **kwargs)
    
    def __getitem__(self, idx):
        filename = self.labels_df['filename'][idx]
        image = decode_image(os.path.join(self.images_dir, filename), mode=torchvision.io.image.ImageReadMode.RGB)

        if self.transform:
            image = self.transform(image)

        label = self.get_default_labels()
        label['pose'] = torch.tensor(np.array(self.labels_df.iloc[idx, 1:].values, dtype = np.float32), dtype = torch.float32)

        return image, label


