import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
from torchvision.io import decode_image

#############################################

# Classification Datasets

#############################################

class ClassificationDataset(nn.Module):
    """
        A general classification dataset that can be inherited. 
    """

    def __init__(self, dataset_dir, image_paths, labels, image_transform = None, target_transform = None, subset = None, train_split = 0.7, seed = 100):
        '''
        Args:
            dataset_dir: (str) The directory of the dataset. Used to obtain the full image paths.

            image_paths: (list) A list of image paths (has to be sorted according to the labels)

            labels: (list) A list of integers for the labels (has to be sorted in ascending order)

            image_transform: (transform) The transform to be applied to the image.

            target_transform: (transform) The transform to be applied to the target (labels).

            subset: (int) specifies a subset of the dataset in terms of the number of classes to load.
                    (train) specifies a random training subset.
                    (test) specifies a random testing subset.
                    (None) Loads the entire dataset
            
            train_split: (float) specifies the portion of the dataset to use for training when subset=train. The remainder is used for testing.

            seed: A seed to initialize the random number generator.
        '''
        super().__init__()
        np.random.seed(seed)
        
        assert len(image_paths) == len(labels), "Please make sure the image_paths and labels are of equal length and correspond correctly!"

        self.dataset_dir = dataset_dir
        self.image_paths = np.array(image_paths, dtype = str)
        self.labels = np.array(labels, dtype = int)
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.subset = subset
        self.train_split = train_split


        if isinstance(self.subset, int):  # Subset by number of classes
            unique_classes = np.unique(self.labels)
            assert self.subset <= len(unique_classes), "Subset must be less than or equal to the number of classes!"
            
            chosen_classes = np.random.choice(unique_classes, size=self.subset, replace=False)
            mask = np.isin(self.labels, chosen_classes)
            self.image_paths = self.image_paths[mask]
            self.labels = self.labels[mask]
        
            #Remap the labels to be from 0 to number_of_classes-1
            unique_labels = np.unique(self.labels)
            label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
            self.labels = np.array([label_mapping[label] for label in self.labels], dtype=int)
        
        elif self.subset in ['train', 'test']: # Subset by train/test split
            assert 0 < self.train_split < 1, "Train split must be between 0 and 1!"

            indices = np.arange(0, len(self.labels))
            np.random.shuffle(indices)
            
            split_index = int(len(indices) * self.train_split)
            
            if self.subset == 'train':
                selected = indices[:split_index]
            else:
                selected = indices[split_index:]
            
            self.image_paths = self.image_paths[selected]
            self.labels = self.labels[selected]
            
    
    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        image_path = os.path.join(self.dataset_dir, self.image_paths[idx]) # Full path
        image = decode_image(image_path)
        label = self.labels[idx]

        if self.image_transform:
            image = self.image_transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label


class SimpleFaceRecognitionDataset(ClassificationDataset):
    """
        A simple face recognition dataset class that can be inherited. 
        Assumes the dataset is organized in the following structure:
        dataset_dir/
            0/                    # Class 0
                1.jpg
                2.jpg
                ...
            1/                    # Class 1
                1.jpg
                2.jpg
                ...
            ...
    """
    def __init__(self, dataset_dir, image_transform = None, target_transform = None, subset = None, train_split = 0.7, seed = 100):
        self.image_paths = []
        self.labels = []
        self.dataset_dir = dataset_dir

        # Sort the classes numerically if they are digits
        classes = os.listdir(dataset_dir)
        int_classes = [int(cls) for cls in classes if cls.isdigit()]
        int_classes.sort()
        classes = [str(cls) for cls in int_classes]

        for i, dir in enumerate(classes):
            
            # Sort the files numerically if they are digits
            files = os.listdir(os.path.join(dataset_dir, dir))
            int_files = [int(f.split('.')[0]) for f in files if f.split('.')[0].isdigit()]
            int_files.sort()
            files = [str(f) + '.jpg' for f in int_files]

            for file in files:
                self.image_paths.append(os.path.join(dir, file)) # Use relative paths
                self.labels.append(i)
        
        super().__init__(self.dataset_dir, self.image_paths, self.labels, image_transform, target_transform, subset, train_split, seed)


# Specific datasets can inherit from SimpleFaceRecognitionDataset
class VGGFace_dataset(SimpleFaceRecognitionDataset):
    def __init__(self, dataset_dir, image_transform = None, target_transform = None, subset = None, train_split = 0.7, seed = 100):
       super().__init__(dataset_dir, image_transform, target_transform, subset, train_split, seed)

class MS1MV2_dataset(SimpleFaceRecognitionDataset):
    def __init__(self, dataset_dir, image_transform = None, target_transform = None, subset = None, train_split = 0.7, seed = 100):
       super().__init__(dataset_dir, image_transform, target_transform, subset, train_split, seed)

class Glint360k_dataset(SimpleFaceRecognitionDataset):
    def __init__(self, dataset_dir, image_transform = None, target_transform = None, subset = None, train_split = 0.7, seed = 100):
       super().__init__(dataset_dir, image_transform, target_transform, subset, train_split, seed)

class CasiaWebFace_dataset(SimpleFaceRecognitionDataset):
    def __init__(self, dataset_dir, image_transform = None, target_transform = None, subset = None, train_split = 0.7, seed = 100):
       super().__init__(dataset_dir, image_transform, target_transform, subset, train_split, seed)





#############################################


# Verification Datasets


#############################################


