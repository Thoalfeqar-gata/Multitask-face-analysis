import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import pandas as pd
from torchvision.io import decode_image
from utility_scripts.landmark_utilities import get_2d_landmarks_from_aflw2000, process_face_image_and_landmarks
from utility_scripts.pose_estimation_utilities import mat_to_rotation_matrix
import pathlib

#############################################

# Classification Datasets

#############################################

class ClassificationDataset(torch.utils.data.Dataset):
    """
        A general classification dataset that can be inherited. 
    """

    def __init__(self, dataset_dir, image_paths, labels, image_transform = None, subset = None, train_split = 0.7, seed = 100):
        '''
        Args:
            dataset_dir: (str) The directory of the dataset. Used to obtain the full image paths.

            image_paths: (list) A list of image paths (has to be sorted according to the labels)

            labels: (list) A list of integers for the labels (has to be sorted in ascending order)

            image_transform: (transform) The transform to be applied to the image.

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
        self.subset = subset
        self.train_split = train_split


        if isinstance(self.subset, int):  # Subset by number of classes
            unique_classes = np.unique(self.labels)
            assert self.subset <= len(unique_classes), "Subset must be less than or equal to the number of classes!"
            
            #choose a random subset
            chosen_classes = rng.choice(unique_classes, size=self.subset, replace=False)
            mask = np.isin(self.labels, chosen_classes)
            self.image_paths = self.image_paths[mask]
            self.labels = self.labels[mask]
        
            #Remap the labels to be from 0 to number_of_classes-1
            unique_labels = np.unique(self.labels)
            label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
            self.labels = np.array([label_mapping[label] for label in self.labels], dtype=int)
            self.num_classes = len(unique_labels)
        
        elif self.subset in ['train', 'test']: # Subset by train/test split
            assert 0 < self.train_split < 1, "Train split must be between 0 and 1!"
            assert self.subset == 'train' or self.subset == 'test', "Subset must be either 'train' or 'test'!"

            indices = np.arange(len(self.labels))
            rng.shuffle(indices) # Shuffle indices to ensure a random split

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
        image = decode_image(image_path, mode = torchvision.io.image.ImageReadMode.RGB)
        label = self.labels[idx]

        if self.image_transform:
            image = self.image_transform(image)
        
        return image, label
    

    def discard_classes(self, threshold):
        """
            Used to discard classes with images less than the threshold given.
            Args:
                threshold (int): The minimum number of images per class to keep
        """
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        classes_to_discard = unique_labels[counts < threshold]
        mask = np.isin(self.labels, classes_to_discard)
        self.image_paths = self.image_paths[~mask]
        self.labels = self.labels[~mask]

        #relabel
        unique_labels = np.unique(self.labels)
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        self.labels = np.array([label_mapping[label] for label in self.labels])
        self.num_classes = len(unique_labels)
        


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
    def __init__(self, dataset_dir, image_transform = None, subset = None, train_split = 0.7, seed=100):
        image_paths = []
        labels = []

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
                image_paths.append(os.path.join(dir, file)) # Use relative paths
                labels.append(i)
        
        super().__init__(dataset_dir, image_paths, labels, image_transform, subset, train_split, seed)


# Specific datasets can inherit from SimpleFaceRecognitionDataset
class VGGFace_Dataset(SimpleFaceRecognitionDataset):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'face recognition', 'VGG-Face', 'aligned') , image_transform = None, subset = None, train_split = 0.7, seed=100):
       super().__init__(dataset_dir, image_transform, subset, train_split, seed)

class MS1MV2_Dataset(SimpleFaceRecognitionDataset):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'face recognition', 'ms1mv2'), image_transform = None, subset = None, train_split = 0.7, seed=100):
       super().__init__(dataset_dir, image_transform, subset, train_split, seed)

class Glint360k_Dataset(SimpleFaceRecognitionDataset):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'face recognition', 'glint360k'), image_transform = None, subset = None, train_split = 0.7, seed=100):
       super().__init__(dataset_dir, image_transform, subset, train_split, seed)

class CasiaWebFace_Dataset(SimpleFaceRecognitionDataset):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'face recognition', 'Casia webface'), image_transform = None, subset = None, train_split = 0.7, seed=100):
       super().__init__(dataset_dir, image_transform, subset, train_split, seed)





#############################################


# Face Verification Datasets


#############################################


class VerificationDataset(torch.utils.data.Dataset):
    """
        A general verification dataset that can be inherited. 
        These datasets will only be used for evaluation (no training).
        The dataset is assumed to be organized in pairs of images with corresponding labels/distances (0 for same identity, 1 for different identity).
    """

    def __init__(self, dataset_dir, image_pairs, labels, image_transform = None):
        '''
        Args:
            dataset_dir: (str) The directory of the dataset. Used to obtain the full image paths.

            image_pairs: (list) A list of tuples containing pairs of image paths (has to be sorted according to the labels)

            labels: (list) A list of integers for the labels (0 for same identity, 1 for different identity) (has to be sorted in ascending order)

            image_transform: (transform) The transform to be applied to the images.

        '''
        super().__init__()
        # The seed is passed to subclasses that may need it for shuffling.
        
        assert len(image_pairs) == len(labels), "Please make sure the image_pairs and labels are of equal length and correspond correctly!"

        self.dataset_dir = dataset_dir
        self.image_pairs = np.array(image_pairs, dtype = object)
        self.labels = np.array(labels, dtype = int)
        self.image_transform = image_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path1 = os.path.join(self.dataset_dir, self.image_pairs[idx][0]) # Full path for image 1
        image_path2 = os.path.join(self.dataset_dir, self.image_pairs[idx][1]) # Full path for image 2

        image1 = decode_image(image_path1, mode = torchvision.io.image.ImageReadMode.RGB)
        image2 = decode_image(image_path2, mode = torchvision.io.image.ImageReadMode.RGB)

        label = self.labels[idx]
        
        # Apply transforms if any
        if self.image_transform:
            image1 = self.image_transform(image1)
            image2 = self.image_transform(image2)

        return (image1, image2), label



class _PairedTxtVerificationDataset(VerificationDataset):
    """
    Helper class for verification datasets that use a 'pairs_....txt' file
    where each pair is on two consecutive lines. (Used with CALFW and CPLFW)

    These datasets' labels are ordered such that all the positive pairs appear first, then the negative pairs, which
    requires shuffling them.
    """
    def __init__(self, dataset_dir, pairs_filename, image_transform=None, seed = 100):
        image_pairs = []
        labels = []

        # Read the pairs.txt file
        pairs_file = os.path.join(dataset_dir, pairs_filename)
        with open(pairs_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for i in range(0, len(lines), 2):
            image1, label1 = lines[i].split(' ')
            image2, label2 = lines[i+1].split(' ')

            assert label1.strip() == label2.strip(), "Labels do not match in pairs file!"

            label = 1 if eval(label1.strip()) > 0 else 0 # 1 for same, 0 for different
            labels.append(label)
            image_pairs.append((os.path.join('aligned images', image1.strip()), os.path.join('aligned images', image2.strip())))
        rng = np.random.RandomState(seed)
        
        #shuffle the data to mix positive and negative labels
        combined = list(zip(image_pairs, labels))
        rng.shuffle(combined)
        image_pairs, labels = zip(*combined)

        
        super().__init__(dataset_dir, image_pairs, labels, image_transform)


class CALFW_Dataset(_PairedTxtVerificationDataset):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'face recognition', 'calfw'), image_transform=None):
        super().__init__(dataset_dir, 'pairs_CALFW.txt', image_transform)

class CPLFW_Dataset(_PairedTxtVerificationDataset):
    def __init__(self, dataset_dir  = os.path.join('data', 'datasets', 'face recognition', 'cplfw'), image_transform=None):
        super().__init__(dataset_dir, 'pairs_CPLFW.txt', image_transform)



class CFP_Dataset(VerificationDataset):
    
    def __init__(self, dataset_dir, image_transform = None):
        image_pairs = []
        labels = []

        # Read the labels.csv file
        pairs_file = os.path.join(dataset_dir, 'labels.csv') # Corrected path for standard CFP
        pairs_df = pd.read_csv(pairs_file)
        
        for i, row in pairs_df.iterrows():
            image1 = row.iloc[0]
            image2 = row.iloc[1]
            label = row.iloc[2]
            image_pairs.append((os.path.join('images', image1), os.path.join('images', image2)))
            labels.append(1 if label else 0) # append 1 if the label is true (1 represents zero distance for similar images) else append 0

        super().__init__(dataset_dir, image_pairs, labels, image_transform)


class CFPFP_Dataset(CFP_Dataset):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'face recognition', 'cfp_fp'), image_transform = None):
        super().__init__(dataset_dir, image_transform)

class CFPFF_Dataset(CFP_Dataset):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'face recognition', 'cfp_ff'), image_transform = None):
        super().__init__(dataset_dir, image_transform)


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
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'face recognition', 'LFW'), image_transform=None, pairs_file="pairs.csv"):
        image_pairs = []
        labels = []

        pairs_path = os.path.join(dataset_dir, pairs_file)
        pairs_df = pd.read_csv(pairs_path)

        for i, row in pairs_df.iterrows():

            if  pd.notna(row.iloc[3]): # a negative pair
                person_name1 = row.iloc[0]
                person_name2 = row.iloc[2]
                image1 = row.iloc[1]
                image2 = row.iloc[3]
                labels.append(0)
                image_pairs.append(
                    (
                        os.path.join("lfw-deepfunneled", person_name1, f"{person_name1}_{int(image1):04d}.jpg"),
                        os.path.join("lfw-deepfunneled", person_name2, f"{person_name2}_{int(image2):04d}.jpg")
                    )
                )

            else: # a positive pair
                person_name = row.iloc[0]
                image1 = row.iloc[1]
                image2 = row.iloc[2]

                labels.append(1)
                image_pairs.append(
                    (
                        os.path.join("lfw-deepfunneled", person_name, f"{person_name}_{int(image1):04d}.jpg"),
                        os.path.join("lfw-deepfunneled", person_name, f"{person_name}_{int(image2):04d}.jpg")
                    )
                )

        super().__init__(dataset_dir, image_pairs, labels, image_transform)


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
    RAF_subet: (str) Either 'train' or 'test' for the train/test split.
                     it is different from the subset variable in the ClassificationDataset class. (sets it to None by default).
    """
    def __init__(
            self, 
            dataset_dir = os.path.join('data', 'datasets', 'emotion recognition', 'RAF_DB', 'RAF_DB', 'basic'), 
            train_image_transform = None, 
            test_image_transform = None, 
            RAF_subset = 'train', 
            seed=100
        ):
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

        def _load_data(label_file):
            image_paths = []
            labels = []

            with open(label_file, 'r') as f:
                lines = f.readlines()

                for line in lines:
                    image_name, label = line.split(' ')
                    image_paths.append(os.path.join('Image', 'aligned', f"{image_name[:-4]}_aligned.jpg"))
                    L = int(label) - 1 #labels should be from 0 to 6, not 1 to 7
                    labels.append(self.label_translation[L]) #append translated label 
            
            return image_paths, labels
        
        if RAF_subset == 'train': 
            label_file = os.path.join(dataset_dir, 'EmoLabel', 'list_train_label.txt')
            self.image_paths, self.labels = _load_data(label_file)

        else: #if testing
            label_file = os.path.join(dataset_dir, 'EmoLabel', 'list_test_label.txt')
            self.image_paths, self.labels = _load_data(label_file)
        
        super().__init__(dataset_dir, self.image_paths, self.labels, train_image_transform if RAF_subset == 'train' else test_image_transform, subset = None, train_split = 0.7, seed=seed)



class ExpW_Dataset(torch.utils.data.Dataset):
    """
    This class doesn't inherit from ClassificationDataset because it rewrites most of its functionality.

    The images must be cropped before used in this dataset.

    The __getitem__ method has to be overriden here.

    The labels in this dataset don't have to be translated.
    """
    def __init__(self, dataset_dir, image_transform = None, seed=100):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.image_transform = image_transform
        
        image_paths = []
        bboxes = []
        labels = []

        labels_file = os.path.join(dataset_dir, 'label.lst')
        
        with open(labels_file, 'r') as f:
            lines = f.readlines()

            for line in lines:
                image_name, face_id_in_image, face_box_top, face_box_left, \
                face_box_right, face_box_bottom, face_box_confidence, expression_label = line.split(' ')

                image_paths.append(os.path.join('origin', image_name))
                bboxes.append(
                    (
                        int(face_box_top),
                        int(face_box_left),
                        int(face_box_bottom),
                        int(face_box_right)
                    )
                )
                labels.append(int(expression_label))
        
        self.image_paths = np.array(image_paths)
        self.bboxes = np.array(bboxes)
        self.labels = np.array(labels)

    def __len__(self):
        return len(self.labels)
    

    def __getitem__(self, idx):
        image_path = os.path.join(self.dataset_dir, self.image_paths[idx]) # Full path
        image = decode_image(image_path, mode = torchvision.io.image.ImageReadMode.RGB)
        image = image[
            :,
            self.bboxes[idx][0]:self.bboxes[idx][2],
            self.bboxes[idx][1]:self.bboxes[idx][3],
        ]
        label = self.labels[idx]

        if self.image_transform:
            image = self.image_transform(image)
        
        return image, label



class AffectNet_Dataset(ClassificationDataset):
    
    def __init__(self, dataset_dir, image_transform = None, seed=100):
        image_paths = []
        labels = []

        self.label_translation = {
            'anger' : 0,
            'disgust' : 1,
            'fear' : 2,
            'happy' : 3,
            'sad' : 4,
            'surprise' : 5,
            'neutral' : 6
        }

        labels_file = os.path.join(dataset_dir, 'labels.csv')
        labels_df = pd.read_csv(labels_file)

        for i, row in labels_df.iterrows():
            image_name = row.iloc[0]
            label = row.iloc[1]
            if label == 'contempt': #exclude contempt labels
                continue

            image_paths.append(image_name)
            labels.append(self.label_translation[label])
        
        super().__init__(dataset_dir, image_paths, labels, image_transform, subset = None, train_split = 0.7, seed=seed)


#############################################

#   Landmark detection datasets

#############################################

"""
    To-Do:
        Find a way to implement augmentations for the image and the landmarks.

"""

class W300_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, image_transform = None, subset = None, train_split = 0.7, seed=100, padding = 1.2, size = 112):
        """
            Padding and size are used for processing the image (cropping and resizing) along with its landmarks.
        """
        
        super().__init__()
        self.dataset_dir = dataset_dir
        self.image_transform = image_transform
        landmarks = []
        self.padding = padding
        self.size = size
        rng = np.random.RandomState(seed)
        
        #construct the image files as named in the directories
        indoor_image_files = [f"indoor_{i:03d}.png" for i in range(1, 301)]
        outdoor_image_files = [f"outdoor_{i:03d}.png" for i in range(1, 301)]

        # construct the landmarks
        for image_file in indoor_image_files:
            name = image_file.split('.')[0]
            pts_file = os.path.join(self.dataset_dir, '01_Indoor', f"{name}.pts")
            landmarks.append(self._process_points(pts_file))

        for image_file in outdoor_image_files:
            name = image_file.split('.')[0]
            pts_file = os.path.join(self.dataset_dir, '02_Outdoor', f"{name}.pts")
            landmarks.append(self._process_points(pts_file))   
        

        #Complete the indoor and outdoor image directories
        indoor_image_files = [os.path.join('01_Indoor', name) for name in indoor_image_files]
        outdoor_image_files = [os.path.join('02_Outdoor', name) for name in outdoor_image_files]


        #Combine outdoor and indoor paths
        image_paths = indoor_image_files + outdoor_image_files

        
        if subset is not None:
            assert subset == 'train' or subset == 'test', "Subset must be either 'train' or 'test'!"

            indices = np.arange(len(image_paths))
            rng.shuffle(indices) #Shuffle to ensure that the train/test split isn't always fixed.

            split_idx = int(len(indices) * train_split)

            if subset == 'train':
                selected_indices = indices[:split_idx]
            else:
                selected_indices = indices[split_idx:]
            
            self.image_paths = np.array(image_paths)[selected_indices]
            self.landmarks = np.array(landmarks, dtype=object)[selected_indices]
        else:
            self.image_paths = np.array(image_paths)
            self.landmarks = np.array(landmarks, dtype=object)
        
    def __len__(self):
        return len(self.landmarks)


    # a function that turns a pts file into a list of points
    def _process_points(self, points):
        landmarks = []

        with open(points, 'r') as f:
            lines = f.readlines()[3:-1] #only read the part of the file where the actual landmark coordinates exist

            for line in lines:
                x, y = line.split(' ')
                landmarks.append((eval(x), eval(y)))

        return landmarks

    
    def __getitem__(self, idx):
        image = decode_image(os.path.join(self.dataset_dir, self.image_paths[idx]), mode = torchvision.io.image.ImageReadMode.RGB) # Merge the complete path and decode
        landmarks = self.landmarks[idx]

        if self.image_transform:
            image = self.image_transform(image)
        
        output_image, output_landmarks = process_face_image_and_landmarks(image, landmarks, self.size, self.padding)
        return output_image, output_landmarks



class AFLW2000_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, image_transform = None, subset = None, train_split = 0.7, seed=100, padding = 1.2, size = 112):
        self.dataset_dir = dataset_dir
        self.image_transform = image_transform
        self.padding = padding
        self.size = size
        self.image_paths = []
        self.landmarks = []


        for file in os.listdir(self.dataset_dir):
            if file.endswith('.jpg'):
                self.image_paths.append(file)
                self.landmarks.append(get_2d_landmarks_from_aflw2000(os.path.join(self.dataset_dir, file.split('.')[0] + '.mat')))
        
        if subset is not None:
            assert subset == 'train' or subset == 'test', "Subset must be either 'train' or 'test'!"

            indices = np.arange(len(self.image_paths))
            rng = np.random.RandomState(seed)
            rng.shuffle(indices)
            split_idx = int(len(indices) * train_split)

            if subset == 'train':
                selected_indices = indices[:split_idx]
            else:
                selected_indices = indices[split_idx:]
            
            self.image_paths = [self.image_paths[i] for i in selected_indices]
            self.landmarks = [self.landmarks[i] for i in selected_indices]

        super().__init__()
    

    def __len__(self):
        return len(self.landmarks)
    

    def __getitem__(self, index):
        image = decode_image(os.path.join(self.dataset_dir, self.image_paths[index]), mode = torchvision.io.image.ImageReadMode.RGB)
        landmarks = self.landmarks[index]

        if self.image_transform:
            image = self.image_transform(image) 
        
        output_image, output_landmarks = process_face_image_and_landmarks(image, landmarks, self.size, self.padding)


        return output_image, output_landmarks



class COFW_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, image_transform = None, subset = 'train', seed=100, padding = 1.2, size = 112):
        self.dataset_dir = dataset_dir
        self.image_transform = image_transform
        self.padding = padding
        self.size = size
        self.image_paths = []
        self.landmarks = []


        assert subset == 'train' or subset == 'test', "Subset must be either 'train' or 'test'!"

        if subset == 'train':
            self.dataset_dir = os.path.join(self.dataset_dir, 'train')
        else:
            self.dataset_dir = os.path.join(self.dataset_dir, 'test')


        for file in os.listdir(self.dataset_dir):
            if file.endswith('.jpg'):
                self.image_paths.append(file)
                start = file.find('_')
                end = file.find('.')
                file_number = int(file[start+1:end])

                excel_file = pd.read_excel(os.path.join(self.dataset_dir, f'pts{file_number}' + '.xlsx'))

                self.landmarks.append(excel_file.to_numpy())
        
        super().__init__()
    
    def __len__(self):
        return len(self.landmarks)

    def __getitem__(self, index):
        image = decode_image(os.path.join(self.dataset_dir, self.image_paths[index]), mode = torchvision.io.image.ImageReadMode.RGB)
        
        landmarks = self.landmarks[index]

        if self.image_transform:
            image = self.image_transform(image)

        output_image, output_landmarks = process_face_image_and_landmarks(image, landmarks, size = self.size, padding = self.padding)

        return output_image, output_landmarks


#############################################

#   Face caption dataset

#############################################

class FaceCaption1M_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, image_transform = None, subset = None, train_split = 0.7, seed=100):
        self.dataset_dir = dataset_dir
        self.image_transform = image_transform
        self.image_paths = []
        self.captions = []

        data_splits = [f'split_{i:05d}' for i in range(32)]
        
        for split in data_splits:
            json_file = pd.read_json(os.path.join(self.dataset_dir, 'json', f'{split}.json'))
            for i, row in json_file.iterrows():
                image_path = row['image']

                if not pathlib.Path(os.path.join(self.dataset_dir, 'images', image_path)).is_file(): #some of the images exist in the json files but don't exist in the actual image data, so ignore them.
                    continue

                caption = row['image_short_caption']
                self.image_paths.append(image_path)
                self.captions.append(caption)

        if subset is not None:
            assert subset == 'train' or subset == 'test', "Subset must be either 'train' or 'test'!"

            indices = np.arange(len(self.image_paths))
            rng = np.random.RandomState(seed)
            rng.shuffle(indices)
            split_idx = int(len(indices) * train_split)

            if subset == 'train':
                selected_indices = indices[:split_idx]
            else:
                selected_indices = indices[split_idx:]

            self.image_paths = [self.image_paths[i] for i in selected_indices]
            self.captions = [self.captions[i] for i in selected_indices]
        

        super().__init__()


    def __len__(self):
        return len(self.captions)
    

    def __getitem__(self, index):
        image = decode_image(os.path.join(self.dataset_dir, 'images', self.image_paths[index]), mode = torchvision.io.image.ImageReadMode.RGB)
        caption = self.captions[index]

        if self.image_transform:
            image = self.image_transform(image)

        return image, caption


#############################################

#   Attribute recognition

#############################################

class CelebA_Dataset(torch.utils.data.Dataset):
    """
        This dataset is already split into training, validation, and testing split, so no training split will be specified in the constructor.
    """
    def __init__(self, dataset_dir, image_transform = None, subset = 'train',  seed=100):
        self.dataset_dir = dataset_dir
        self.image_transform = image_transform
        self.image_paths = []
        self.attributes = []
        assert subset == 'train' or subset == 'test' or subset == 'validation', "Subset must be either 'train', 'validation', or test'!"


        eval_file = pd.read_csv(os.path.join(self.dataset_dir, 'list_eval_partition.csv'))
        attr_file = pd.read_csv(os.path.join(self.dataset_dir, 'list_attr_celeba.csv'))

        if subset == 'train':
            eval_file = eval_file[eval_file['partition'] == 0]
        elif subset == 'validation':
            eval_file = eval_file[eval_file['partition'] == 1]
        else:
            eval_file = eval_file[eval_file['partition'] == 2]


        for i, row in eval_file.iterrows():
            image_path = row['image_id']
            self.image_paths.append(image_path)
            attr_row = attr_file.iloc[i]
            attributes = attr_row.to_numpy()[1:]
            attributes[attributes == -1] = 0 #convert -1 to 0
            attributes = attributes.astype(np.uint8)
            self.attributes.append(attributes)
            
                
        super().__init__()
    
    def __len__(self):
        return len(self.attributes)
    

    def __getitem__(self, index):
        image = decode_image(os.path.join(self.dataset_dir, 'img_align_celeba', self.image_paths[index]), mode = torchvision.io.image.ImageReadMode.RGB)
        attributes = self.attributes[index]

        if self.image_transform:
            image = self.image_transform(image)
        
        return image, attributes


#############################################

#   Pose estimation

#############################################

class W300LP_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, image_transform = None, subset = None, train_split = 0.7, seed=100):
        self.dataset_dir = dataset_dir
        self.image_transform = image_transform
        self.image_paths = []
        self.pose_paths = []
        
        # Load all the subdirectories of 300W-LP dataset except the "Code" directory since it doesn't have any data
        self.dirs = [directory for directory in os.listdir(self.dataset_dir) if directory != 'Code']

        for directory in self.dirs:
            image_dir = os.path.join(self.dataset_dir, directory)

            for file in os.listdir(image_dir):
                if file.endswith('.jpg'):

                    # Don't join the full paths here to save up on RAM
                    self.image_paths.append(os.path.join(directory, file)) 
                    self.pose_paths.append(os.path.join(directory, file.split('.')[0] + '.mat'))
    
        super().__init__()


    def __len__(self):
        return len(self.image_paths)
    

    def __getitem__(self, index):
        image = decode_image(os.path.join(self.dataset_dir, self.image_paths[index]), mode = torchvision.io.image.ImageReadMode.RGB)
        pose = mat_to_rotation_matrix(os.path.join(self.dataset_dir, self.pose_paths[index]))

        if self.image_transform:
            image = self.image_transform(image)
        
        return image, pose

#############################################

#   Face parsing

#############################################

"""
    Implement here
"""

#############################################

#   Age estimation
#   Note: Some of the following datasets return multiple labels.
#   For example, AgeDB has three labels, one for the identity, one for the age, and one for the gender.
#   If a dataset has multiple labels, all the labels will be returned by the dataset class.

#############################################

class AgeDB_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'age gender and race estimation', 'AgeDB'), 
                 image_transform = None,  
                 subset = None, 
                 train_split = 0.7, 
                 seed = 100
        ):
        
        # In this dataset, we will assume the test and validation datasets to be half of what remains after the training split
        # For example, if train_split = 0.7, test_split = 0.15, validations_split = 0.15
        self.dataset_dir = dataset_dir
        self.image_transform = image_transform


        # Load the entire dataset
        self.image_paths = []
        self.identity_labels = []
        self.age_labels = []
        self.gender_labels = []


        indices = []
        for image_name in os.listdir(self.dataset_dir):
            if image_name.endswith('.jpg'):
                self.image_paths.append(image_name)
            
                index, identity, age, gender = image_name.split('_') # split the labels from the image name
                indices.append(int(index))
                self.identity_labels.append(identity)
                self.age_labels.append(float(age))
                self.gender_labels.append(gender.split('.')[0]) # remove the .jpg from the gender

        # sort the indices
        sorting_indices = np.argsort(indices)
        self.image_paths = [self.image_paths[i] for i in sorting_indices]
        self.identity_labels = [self.identity_labels[i] for i in sorting_indices]
        self.age_labels = [self.age_labels[i] for i in sorting_indices]
        self.gender_labels = [self.gender_labels[i] for i in sorting_indices]
        
        # transform the identity and gender into numbers
        unique_identities = []
        for identity in self.identity_labels:
            if identity not in unique_identities:
                unique_identities.append(identity)
        
        identity_mapping = {identity: i for i, identity in enumerate(unique_identities)}
        self.identity_labels = [identity_mapping[identity] for identity in self.identity_labels]

        self.gender_labels = [1 if gender == 'm' else 0 for gender in self.gender_labels]

        self.classnum = len(unique_identities)

        if subset is not None:
            assert subset in ['train', 'test', 'validation'], "Subset must be either 'train', 'test', or 'validation'!"
            indices = np.arange(len(self.image_paths))
            rng = np.random.RandomState(seed)
            rng.shuffle(indices)
            split_idx = int(len(indices) * train_split)

            if subset == 'train':
                selected_indices = indices[:split_idx]
            elif subset == 'test':
                selected_indices = indices[split_idx:split_idx + int(len(indices) * (1 - train_split) * 0.5)]
            else:
                selected_indices = indices[split_idx + int(len(indices) * (1 - train_split) * 0.5):]
            
            self.image_paths = [self.image_paths[i] for i in selected_indices]
            self.identity_labels = [self.identity_labels[i] for i in selected_indices]
            self.age_labels = [self.age_labels[i] for i in selected_indices]
            self.gender_labels = [self.gender_labels[i] for i in selected_indices]  

        super().__init__()


    def __len__(self):
        return len(self.image_paths)
    

    def __getitem__(self, index):
        image = decode_image(os.path.join(self.dataset_dir, self.image_paths[index]), mode = torchvision.io.image.ImageReadMode.RGB)
        identity = self.identity_labels[index]
        age = self.age_labels[index]
        gender = self.gender_labels[index]

        if self.image_transform:
            image = self.image_transform(image)

        return image, (identity, age, gender)
   

#############################################

#   Gender recognition

#############################################

"""
    Implement here
"""

#############################################

#   Race/Ethnicity recognition

#############################################

"""
    Implement here
"""



#############################################
"""
    The following class is the main one used in the multitask training setup.
    It will be updated with more tasks as dataset classes above are finalized.
"""
#############################################

class MultiTaskDataLoader:
    def __init__(self, batch_size = 192, image_transform = None, face_recognition_image_transform = None, min_num_images_per_class = 20):
        
        self.face_recognition_dataset = CasiaWebFace_Dataset(image_transform=face_recognition_image_transform)
        if min_num_images_per_class is not None:
            self.face_recognition_dataset.discard_classes(min_num_images_per_class)

        self.datasets = {
            'face recognition' : self.face_recognition_dataset,
            'emotion recognition' : RAF_Dataset(train_image_transform=image_transform),
            'age and gender' : AgeDB_Dataset(image_transform=image_transform),
        }

        self.batch_composition = {
            'face recognition' : 1,
            'emotion recognition' : 1,
            'age and gender' : 1,
        }
        
        total_weight = sum(self.batch_composition.values())

        self.data_loaders = {}
        for task in self.datasets.keys():
            self.data_loaders[task] = torch.utils.data.DataLoader(
                self.datasets[task],
                batch_size = self.batch_composition[task] * batch_size // total_weight,
                shuffle = True,
                num_workers = 4,
                pin_memory = True
            )

        
    def __iter__(self):
        self.loader_iters = {task: iter(loader) for task, loader in self.data_loaders.items()}
        return self

    def __len__(self):
        total_batches = max(len(loader) for loader in self.data_loaders.values())
        return total_batches

    def __next__(self):
        
        """
            The face recognition dataloader is the largest among the dataloader.
            It will be processed at the end of this method to let other loaders
            give one final batch before the epoch ends.
        """
        try:
            # Use the persistent iterator from self.loader_iters
            emotion_images, emotion_labels = next(self.loader_iters['emotion recognition'])
        except StopIteration:
            # If the iterator is exhausted, reset it and get the first batch
            self.loader_iters['emotion recognition'] = iter(self.data_loaders['emotion recognition'])
            emotion_images, emotion_labels = next(self.loader_iters['emotion recognition'])
        
        try:
            age_gender_images, (identity, age, gender) = next(self.loader_iters['age and gender'])
        except StopIteration:
            self.loader_iters['age and gender'] = iter(self.data_loaders['age and gender'])
            age_gender_images, (identity, age, gender) = next(self.loader_iters['age and gender'])
        
        # Get the next batch from the largest loader.
        # When this raises StopIteration, it will correctly terminate the epoch.
        face_images, face_labels = next(self.loader_iters['face recognition'])


        return {
            'face recognition' : (face_images, face_labels),
            'emotion recognition' : (emotion_images, emotion_labels),
            'age and gender' : (age_gender_images, (age, gender))
        }
        


