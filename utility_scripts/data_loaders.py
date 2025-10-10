import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import pandas as pd
from torchvision.io import decode_image


#############################################

# Classification Datasets

#############################################

class ClassificationDataset(torch.utils.data.Dataset):
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
        rng = np.random.RandomState(seed)
        
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
            
            chosen_classes = rng.choice(unique_classes, size=self.subset, replace=False)
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
            rng.shuffle(indices)
            
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


#############################################

#    Face Recognition Datasets

#############################################

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
        self.subset = subset
        self.train_split = train_split
        self.seed = seed

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
class VGGFace_Dataset(SimpleFaceRecognitionDataset):
    def __init__(self, dataset_dir, image_transform = None, target_transform = None, subset = None, train_split = 0.7, seed = 100):
       super().__init__(dataset_dir, image_transform, target_transform, subset, train_split, seed)

class MS1MV2_Dataset(SimpleFaceRecognitionDataset):
    def __init__(self, dataset_dir, image_transform = None, target_transform = None, subset = None, train_split = 0.7, seed = 100):
       super().__init__(dataset_dir, image_transform, target_transform, subset, train_split, seed)

class Glint360k_Dataset(SimpleFaceRecognitionDataset):
    def __init__(self, dataset_dir, image_transform = None, target_transform = None, subset = None, train_split = 0.7, seed = 100):
       super().__init__(dataset_dir, image_transform, target_transform, subset, train_split, seed)

class CasiaWebFace_Dataset(SimpleFaceRecognitionDataset):
    def __init__(self, dataset_dir, image_transform = None, target_transform = None, subset = None, train_split = 0.7, seed = 100):
       super().__init__(dataset_dir, image_transform, target_transform, subset, train_split, seed)





#############################################


# Face Verification Datasets


#############################################


class VerificationDataset(torch.utils.data.Dataset):
    """
        A general verification dataset that can be inherited. 
        These datasets will only be used for evaluation (no training).
        The dataset is assumed to be organized in pairs of images with corresponding labels/distances (0 for same identity, 1 for different identity).
    """

    def __init__(self, dataset_dir, image_pairs, labels, image_transform = None, target_transform = None, seed = 100):
        '''
        Args:
            dataset_dir: (str) The directory of the dataset. Used to obtain the full image paths.

            image_pairs: (list) A list of tuples containing pairs of image paths (has to be sorted according to the labels)

            labels: (list) A list of integers for the labels (0 for same identity, 1 for different identity) (has to be sorted in ascending order)

            image_transform: (transform) The transform to be applied to the images.

            target_transform: (transform) The transform to be applied to the target (labels).

            seed: A seed to initialize the random number generator.
        '''
        super().__init__()
        # The seed is passed to subclasses that may need it for shuffling.
        
        assert len(image_pairs) == len(labels), "Please make sure the image_pairs and labels are of equal length and correspond correctly!"

        self.dataset_dir = dataset_dir
        self.image_pairs = np.array(image_pairs, dtype = object)
        self.labels = np.array(labels, dtype = int)
        self.image_transform = image_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path1 = os.path.join(self.dataset_dir, self.image_pairs[idx][0]) # Full path for image 1
        image_path2 = os.path.join(self.dataset_dir, self.image_pairs[idx][1]) # Full path for image 2

        image1 = decode_image(image_path1)
        image2 = decode_image(image_path2)

        label = self.labels[idx]
        
        # Apply transforms if any
        if self.image_transform:
            image1 = self.image_transform(image1)
            image2 = self.image_transform(image2)
        if self.target_transform:
            label = self.target_transform(label)
        
        return (image1, image2), label


class _PairedTxtVerificationDataset(VerificationDataset):
    """
    Helper class for verification datasets that use a 'pairs_....txt' file
    where each pair is on two consecutive lines. (Used with CALFW and CPLFW)
    """
    def __init__(self, dataset_dir, pairs_filename, image_transform=None, target_transform=None, shuffle=False, seed=100):
        rng = np.random.RandomState(seed)
        self.image_pairs = []
        self.labels = []
        self.dataset_dir = dataset_dir

        # Read the pairs.txt file
        pairs_file = os.path.join(dataset_dir, pairs_filename)
        with open(pairs_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for i in range(0, len(lines), 2):
            image1, label1 = lines[i].split(' ')
            image2, label2 = lines[i+1].split(' ')

            assert label1.strip() == label2.strip(), "Labels do not match in pairs file!"

            label = 0 if eval(label1.strip()) > 0 else 1 # 0 for same, 1 for different
            self.labels.append(label)
            self.image_pairs.append((os.path.join('aligned images', image1.strip()), os.path.join('aligned images', image2.strip())))

        if shuffle:
            combined = list(zip(self.image_pairs, self.labels))
            rng.shuffle(combined)
            self.image_pairs[:], self.labels[:] = zip(*combined)

        super().__init__(self.dataset_dir, self.image_pairs, self.labels, image_transform, target_transform, seed)


class CALFW_Dataset(_PairedTxtVerificationDataset):
    def __init__(self, dataset_dir, image_transform=None, target_transform=None, shuffle=False, seed=100):
        super().__init__(dataset_dir, 'pairs_CALFW.txt', image_transform, target_transform, shuffle, seed)

class CPLFW_Dataset(_PairedTxtVerificationDataset):
    def __init__(self, dataset_dir, image_transform=None, target_transform=None, shuffle=False, seed=100):
        super().__init__(dataset_dir, 'pairs_CPLFW.txt', image_transform, target_transform, shuffle, seed)


class CFP_Dataset(VerificationDataset):
    
    def __init__(self, dataset_dir, image_transform = None, target_transform = None, shuffle = False, seed = 100):
        rng = np.random.RandomState(seed)
        self.image_pairs = []
        self.labels = []
        self.dataset_dir = dataset_dir

        # Read the labels.csv file
        pairs_file = os.path.join(dataset_dir, 'labels.csv') # Corrected path for standard CFP
        pairs_df = pd.read_csv(pairs_file)

        
        for i, row in pairs_df.iterrows():
            image1 = row.iloc[0]
            image2 = row.iloc[1]
            label = row.iloc[2]
            
            self.image_pairs.append((os.path.join('images', image1), os.path.join('images', image2)))
            self.labels.append(0 if label else 1) # append 0 if the label is true (0 represents zero distance for similar images) else append 1
        

        if shuffle:
            combined = list(zip(self.image_pairs, self.labels))
            rng.shuffle(combined)
            self.image_pairs[:], self.labels[:] = zip(*combined)

        super().__init__(self.dataset_dir, self.image_pairs, self.labels, image_transform, target_transform, seed)


class LFW_Dataset(VerificationDataset):
    """
    Loads the Labeled Faces in the Wild (LFW) dataset for face verification.
    It parses the standard `pairs.txt` file to create image pairs.
    
    Shuffling this dataset is not allowed, because it would result int folds that are 
    not comparable to SOTA work, so no shuffle argument is used.

    The LFW dataset structure is assumed to be:
    dataset_dir/
        Person_A/
            Person_A_0001.jpg
            ...
        Person_B/
            Person_B_0001.jpg
            ...
    """
    def __init__(self, dataset_dir, image_transform=None, target_transform=None, pairs_file="pairs.csv", seed=100):
        self.image_pairs = []
        self.labels = []
        self.dataset_dir = dataset_dir

        pairs_path = os.path.join(self.dataset_dir, pairs_file)
        pairs_df = pd.read_csv(pairs_path)

        for i, row in pairs_df.iterrows():

            if  pd.notna(row.iloc[3]): # a negative pair
                person_name1 = row.iloc[0]
                person_name2 = row.iloc[2]
                image1 = row.iloc[1]
                image2 = row.iloc[3]
                self.labels.append(1)
                self.image_pairs.append(
                    (
                        os.path.join("lfw-deepfunneled", person_name1, f"{person_name1}_{int(image1):04d}.jpg"),
                        os.path.join("lfw-deepfunneled", person_name2, f"{person_name2}_{int(image2):04d}.jpg")
                    )
                )


            
            else: # a positive pair
                person_name = row.iloc[0]
                image1 = row.iloc[1]
                image2 = row.iloc[2]

                self.labels.append(0)
                self.image_pairs.append(
                    (
                        os.path.join("lfw-deepfunneled", person_name, f"{person_name}_{int(image1):04d}.jpg"),
                        os.path.join("lfw-deepfunneled", person_name, f"{person_name}_{int(image2):04d}.jpg")
                    )
                )

        super().__init__(self.dataset_dir, self.image_pairs, self.labels, image_transform, target_transform, seed)


#############################################

#    Emotion Recognition Datasets

#############################################
"""

    All emotion recognition will follow the same label format, which is:
    0 ==> Angry
    1 ==> Disgust
    2 ==> Fear
    3 ==> Happy
    4 ==> Sad
    5 ==> Surprise
    6 ==> Neutral
    
"""


class RAF_Dataset(ClassificationDataset):
    """
    A RAF_Dataset class that inherits from ClassificationDataset.
    RAF_subet: (str) can be either 'train' or 'test', and is used to choose the RAF train or test split.
                     it is different from the subset variable in the ClassificationDataset class. (sets it to None by default).
    """
    def __init__(self, dataset_dir, image_transform = None, target_transform = None, RAF_subset = 'train', seed = 100, shuffle = False):
        self.dataset_dir = dataset_dir
        self.image_paths = []
        self.labels = []
        
        """
            RAF_DB original label structure is:
            1: Surprise  : 0
            2: Fear      : 1
            3: Disgust   : 2
            4: Happiness : 3
            5: Sadness   : 4
            6: Anger     : 5
            7: Neutral   : 6
        """
        self.label_translation = [5, 2, 1, 3, 4, 0, 6]

        rng = np.random.RandomState(seed)

        if RAF_subset == 'train': #if training
            label_file = os.path.join(dataset_dir, 'EmoLabel', 'list_train_label.txt')

            with open(label_file, 'r') as f:
                lines = f.readlines()

                for line in lines:
                    image_name, label = line.split(' ')
                    self.image_paths.append(os.path.join('Image', 'aligned', f"{image_name[:-4]}_aligned.jpg"))
                    L = int(label) - 1 #labels should be from 0 to 6, not 1 to 7
                    self.labels.append(self.label_translation[L]) #append translated label 

        else: #if testing
            label_file = os.path.join(dataset_dir, 'EmoLabel', 'list_test_label.txt')

            with open(label_file, 'r') as f:
                lines = f.readlines()   
                
                for line in lines:
                    image_name, label = line.split(' ')
                    self.image_paths.append(os.path.join('Image', 'aligned', f"{image_name[:-4]}_aligned.jpg"))
                    L = int(label) - 1 #labels should be from 0 to 6, not 1 to 7
                    self.labels.append(self.label_translation[L]) #append translated label
        
        if shuffle:
            combined = list(zip(self.image_paths, self.labels))
            rng.shuffle(combined)
            self.image_paths[:], self.labels[:] = zip(*combined)

        super().__init__(self.dataset_dir, self.image_paths, self.labels, image_transform, target_transform, subset = None, train_split = 0.7, seed = seed)


class ExpW_Dataset:
    """
    This class doesn't inherit from ClassificationDataset because it rewrites most of its functionality.

    The images must be cropped before used in this dataset.

    The __getitem__ method has to be overriden here.

    The labels in this dataset don't have to be translated.
    """
    def __init__(self, dataset_dir, image_transform = None, target_transform = None, seed = 100, shuffle = False):
        self.dataset_dir = dataset_dir
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.image_paths = []
        self.bboxes = []
        self.labels = []
        rng = np.random.RandomState(seed)

        labels_file = os.path.join(dataset_dir, 'label.lst')
        
        with open(labels_file, 'r') as f:
            lines = f.readlines()

            for line in lines:
                image_name, face_id_in_image, face_box_top, face_box_left, \
                face_box_right, face_box_bottom, face_box_cofidence, expression_label = line.split(' ')

                self.image_paths.append(os.path.join('origin', image_name))
                self.bboxes.append(
                    (
                        int(face_box_top),
                        int(face_box_left),
                        int(face_box_bottom),
                        int(face_box_right)
                    )
                )
                self.labels.append(int(expression_label))
        
        if shuffle:
            shuffle_indices = np.arange(len(self.labels))
            rng.shuffle(shuffle_indices)
            self.image_paths = [self.image_paths[i] for i in shuffle_indices]
            self.bboxes = [self.bboxes[i] for i in shuffle_indices]
            self.labels = [self.labels[i] for i in shuffle_indices]
        

    def __len__(self):
        return len(self.labels)
    

    def __getitem__(self, idx):
        image_path = os.path.join(self.dataset_dir, self.image_paths[idx]) # Full path
        image = decode_image(image_path)
        image = image[
            :,
            self.bboxes[idx][0]:self.bboxes[idx][2],
            self.bboxes[idx][1]:self.bboxes[idx][3],
        ]
        label = self.labels[idx]

        if self.image_transform:
            image = self.image_transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label

class AffectNet_Dataset(ClassificationDataset):
    
    def __init__(self, dataset_dir, image_transform = None, target_transform = None, shuffle = False, seed = 100):
        self.dataset_dir = dataset_dir
        self.image_paths = []
        self.labels = []

        self.label_translation = {
            'anger' : 0,
            'disgust' : 1,
            'fear' : 2,
            'happy' : 3,
            'sad' : 4,
            'surprise' : 5,
            'neutral' : 6
        }

        rng = np.random.RandomState(seed)

        labels_file = os.path.join(dataset_dir, 'labels.csv')
        labels_df = pd.read_csv(labels_file)

        for i, row in labels_df.iterrows():
            image_name = row.iloc[0]
            label = row.iloc[1]
            if label == 'contempt': #exclude contempt labels
                continue

            self.image_paths.append(image_name)
            self.labels.append(self.label_translation[label])
        
        if shuffle:
            combined = list(zip(self.image_paths, self.labels))
            rng.shuffle(combined)
            self.image_paths[:], self.labels[:] = zip(*combined)
        
        super().__init__(self.dataset_dir, self.image_paths, self.labels, image_transform, target_transform, subset = None, train_split = 0.7, seed = seed)
