import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import pandas as pd
from torchvision.io import decode_image
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler


def get_balanced_loader(datasets_dict: dict[str, torch.utils.data.Dataset], batch_size: int, num_workers: int, epoch_size: int = None):
    """

    This function returns a dataloader such that each task will have an equal probability of being chosen. 

    Args:
        datasets (dict): A dictionary of datasets for each task. 
                              Each dataset must have a get_sample_weight method defined.
        batch_size (int): How many samples per batch to load.
        num_workers (int): How many subprocesses to use for data loading.
        epoch_size (int): How many images to sample per epoch. 
    
    Returns:
        dataloader (torch.utils.data.DataLoader): A dataloader with balanced loading of the datasets.                          
    """
    # 1. get the sample weights per task and concatenate the datasets of each task.
    task_weight = 1 / len(datasets_dict) # assign equal weight for each task.
    task_sample_weights = [] # accumulate sample weights per task here.
    task_datasets = [] # accumulate concatenated datasets per task here.
    for task_name in datasets_dict.keys():
        if len(datasets_dict[task_name]) == 1: # for tasks with one dataset
            task_datasets.append(datasets_dict[task_name][0])
            task_sample_weights.append(datasets_dict[task_name][0].get_sample_weights() * task_weight)
        else: # for tasks with more than one dataset
            task_datasets.append(ConcatDataset(datasets_dict[task_name])) # concatenate the task's datasets and add them
            
            # concatenate the sample weights of each dataset within this task, then multiply by the task_weight / the number of datasets in this task 
            task_sample_weights.append(np.concatenate([dataset.get_sample_weights() for dataset in datasets_dict[task_name]]) \
                                             * (task_weight / len(datasets_dict[task_name]))) # distribute the task weight equally among the datasets of this task.


    # 2. concatenate the datasets of each task and the sample weights
    unified_dataset = ConcatDataset(task_datasets)
    sample_weights = np.concatenate(list(task_sample_weights))
    
    # 3. Define Epoch Size
    # If we don't set this, an 'epoch' is just the sum of all raw lengths
    # But we likely want to define a 'Virtual Epoch' to check validation often.
    if epoch_size is None:
        epoch_size = len(unified_dataset)
        
    # 4. Create the Sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=epoch_size,
        replacement=True # Crucial: Allows picking samples multiple times
    )
    
    # 5. Create Loader
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
    
    def get_sample_weights(self):
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

    def get_sample_weights(self):
        """
            Face recognition datasets will not be balanced.
            This method returns an equal weight for each sample to be combined in the get_balanced_loader method.
        """
        sample_weights = np.ones(len(self.labels_df))
        return sample_weights / sample_weights.sum()

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

    def get_sample_weights(self):
        """
            Returns the weight of each sample in this dataset such that it balances the classes.
            Classes with low samples get a higher weight.
            The final weight must sum to 1.
        """

        label_counts = self.labels_df['label'].value_counts().sort_index()
        class_weights = len(self.labels_df) / label_counts
        sample_weights = self.labels_df['label'].map(class_weights).to_numpy()
        return sample_weights / sample_weights.sum()


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
    
    def get_sample_weights(self):
        """
            Returns the weight of each sample such that it balances the males and females. Age is not balanced
            The final weight must sum to 1.
        """

        label_counts = self.labels_df['gender'].value_counts().sort_index()
        class_weights = len(self.labels_df) / label_counts
        sample_weights = self.labels_df['gender'].map(class_weights).to_numpy()
        return sample_weights / sample_weights.sum()



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

    def get_sample_weights(self):
        """
            Returns the weight of each sample such that it balances the males and females.
            The final weight must sum to 1.
        """

        label_counts = self.labels_df['gender'].value_counts().sort_index()
        class_weights = len(self.labels_df) / label_counts
        sample_weights = self.labels_df['gender'].map(class_weights).to_numpy()
        return sample_weights / sample_weights.sum()


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
    
    def get_sample_weights(self):
        """
            Returns the weight of each sample such that it balanced both the race and the gender.
            The final weight must sum to 1.
        """

        gender_label_counts = self.labels_df['gender'].value_counts().sort_index()
        gender_class_weights = len(self.labels_df) / gender_label_counts
        gender_sample_weights = self.labels_df['gender'].map(gender_class_weights).to_numpy()

        race_label_counts = self.labels_df['race'].value_counts().sort_index()
        race_class_weights = len(self.labels_df) / race_label_counts
        race_sample_weights = self.labels_df['race'].map(race_class_weights).to_numpy()

        sample_weights = gender_sample_weights * race_sample_weights
        return sample_weights / sample_weights.sum()




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

    def get_sample_weights(self):
        """
            Returns the weight of each sample such that it balanced both the race and the gender.
            The final weight must sum to 1.
        """

        gender_label_counts = self.labels_df['gender'].value_counts().sort_index()
        gender_class_weights = len(self.labels_df) / gender_label_counts
        gender_sample_weights = self.labels_df['gender'].map(gender_class_weights).to_numpy()

        race_label_counts = self.labels_df['race'].value_counts().sort_index()
        race_class_weights = len(self.labels_df) / race_label_counts
        race_sample_weights = self.labels_df['race'].map(race_class_weights).to_numpy()

        sample_weights = gender_sample_weights * race_sample_weights
        return sample_weights / sample_weights.sum()
    

###################################

#       Attribute recognition

###################################

class CelebA(BaseDatasetClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'attribute recognition', 'CelebA'), transform = None, subset = None, **kwargs):
        super().__init__(dataset_dir, transform, **kwargs)
        if subset != None: # Can be 'train', 'test', or 'validation'
            self.labels_df = self.labels_df[self.labels_df['split'] == subset]
            self.labels_df.reset_index(drop = True, inplace = True)

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
    
    def get_cut_indices(self):
        cut_indices = [0]
        for key, value in self.attribute_groups.items():
            cut_indices.append(len(value) + cut_indices[-1])
        
        return cut_indices

    def __getitem__(self, idx):
        filename = self.labels_df['filename'][idx]
        image = decode_image(os.path.join(self.images_dir, filename), mode = torchvision.io.image.ImageReadMode.RGB)

        if self.transform:
            image = self.transform(image)

        label = self.get_default_labels()

        # Convert -1 to 0
        raw_attrs = self.labels_df.iloc[idx, 1:-1].values
        label['attributes'] = torch.tensor((raw_attrs > 0).astype(np.uint8), dtype=torch.uint8)

        return image, label
    
    def get_sample_weights(self):
        """
            Balancing this dataset is handled using the loss function.
            This method returns an equal weight for each sample to be combined in the get_balanced_loader method.
        """
        sample_weights = np.ones(len(self.labels_df))
        return sample_weights / sample_weights.sum()
    

    def get_attribute_weights(self):
            """
            Returns the pos_weight for BCEWithLogitsLoss.
            Formula: Negative_Count / Positive_Count
            """
            # 1. Count positives. Replace -1 with 0 for the negative.
            labels = self.labels_df.iloc[:, 1:-1].replace(-1, 0)
            
            pos_counts = labels.sum()
            total_counts = len(self.labels_df)
            neg_counts = total_counts - pos_counts
            
            # 2. Calculate Ratio (Neg / Pos)
            # Add epsilon to avoid division by zero
            pos_weights = neg_counts / (pos_counts + 1e-6)
            
            # Return as tensor
            return torch.tensor(pos_weights.values, dtype=torch.float32)



###################################

#       Head Pose Estimation

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
    
    def get_sample_weights(self):
        """
            This dataset will not be balanced since it represents a continuous range of angles for the head pose.
            This method returns an equal weight for each sample to be combined in the get_balanced_loader method.
        """
        sample_weights = np.ones(len(self.labels_df))
        return sample_weights / sample_weights.sum()


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
    
    def get_sample_weights(self):
        """
            This dataset will not be balanced since it represents a continuous range of angles for the head pose.
            This method returns an equal weight for each sample to be combined in the get_balanced_loader method.
        """
        sample_weights = np.ones(len(self.labels_df))
        return sample_weights / sample_weights.sum()


