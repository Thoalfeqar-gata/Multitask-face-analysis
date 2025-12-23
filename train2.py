import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import backbones.backbones as backbones
import datasets as db
import multitask.face_recognition_heads as face_recognition_heads
import eval, os, numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset
from torchvision.transforms import v2
from torch.optim import lr_scheduler
from augmenter import Augmenter
from multitask.subnets import FaceRecognitionEmbeddingSubnet, GenderRecognitionSubnet, AgeEstimationSubnet, \
                              EmotionRecognitionSubnet, RaceRecognitionSubnet
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from multitask.models import MultiTaskFaceAnalysisModel
from configs.train_multitask import config
from multitask.loss_weighing import DynamicWeightAverage
from matplotlib import pyplot as plt

torch.set_float32_matmul_precision('medium')

"""

To-Do: Implement the weigted random sampler you discussed with gemini


"""



def main(**kwargs): 

    use_validation = True # toggle validation datasets on or off. 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = kwargs.get('resume_from_checkpoint')

    # train and test transforms
    train_face_rec_transform = v2.Compose([ # for face recognition during training.
        v2.ToPILImage(),
        Augmenter(crop_augmentation_prob=0.2, low_res_augmentation_prob=0.2, photometric_augmentation_prob=0.2),
        v2.Resize((112, 112)),
        v2.RandomHorizontalFlip(),
        v2.ToTensor(),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_transform = v2.Compose([ # for other datasets during training.
        v2.ToPILImage(),
        v2.Resize((112, 112)),
        v2.RandomHorizontalFlip(),
        v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        v2.RandomRotation(degrees=10),
        v2.RandomGrayscale(p=0.1),
        v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2))], p=0.1),
        v2.ToTensor(),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    test_transform = v2.Compose([ # for testing on datasets other than face recognition.
        v2.ToPILImage(),
        v2.Resize((112, 112)),
        v2.ToTensor(),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Data Preparation
    
    # training
    train_db_list = [
        # Face recognition
        db.MS1MV2(transform=train_face_rec_transform, return_name = True),

        # emotion recognition
        db.AffectNet(transform = train_transform, subset = 'train', return_name = True), 
        db.RAFDB(transform = train_transform, subset = 'train', return_name = True),

        # Age, Gender, and Race
        db.FairFace(transform = train_transform, subset = 'train', return_name = True), # gender and race
        db.UTKFace(transform = train_transform, subset = 'train', return_name = True), # age, gender, race
        db.MORPH(transform = train_transform, subset = 'train', return_name = True) # age, gender
    ]    
    
    batch_size = kwargs.get('batch_size')
    num_workers = kwargs.get('num_workers')

    train_loader = db.get_balanced_loader(
        train_db_list,
        batch_size = batch_size, 
        num_workers = num_workers,
    )


    if use_validation:
        # validation
        fairface_test_db = torch.utils.data.DataLoader(
            dataset = db.FairFace(transform = test_transform, subset = 'test'),
            batch_size = 64,
            shuffle = True,
            num_workers = 2,
            pin_memory = True,
        )

        affectnet_validation_db = torch.utils.data.DataLoader(
            dataset = db.AffectNet(transform = test_transform, subset = 'test'),
            batch_size = 64,
            shuffle = True,
            num_workers = 2,
            pin_memory = True,
        )

        morph_test_db = torch.utils.data.DataLoader(
            dataset = db.MORPH(transform = test_transform, subset = 'validation'),
            batch_size = 64,
            shuffle = True,
            num_workers = 2,
            pin_memory = True,
        )

        print('fairface lenght: ', len(fairface_test_db) * 64)
        print('affectnet lenght: ', len(affectnet_validation_db) * 64)
        print('morph lenght: ', len(morph_test_db) * 64)



    print(len(train_loader))
    images, labels = next(iter(train_loader))


    emotion_translation = {
    0 : 'anger',
    1 : 'disgust',
    2 : 'fear',
    3 : 'happy',
    4 : 'sad',
    5 : 'surprise',
    6 : 'neutral'
    }

    gender_translation = {
    1 : 'Male',
    0 : 'Female'
    }

    race_translation = {
    0 : 'White',
    1 : 'Black',
    2 : 'Asian',
    3 : 'Indian',
    4 : 'Other'
    }

    plt.figure(figsize=(24, 8), dpi = 150)
    for i in range(16):
        idx = np.random.randint(0, len(images))
        image = images[idx]
        image = image + 1
        image = image / 2
        label_text = ''

        face_label = labels['face_recognition'][idx]
        if face_label != -1:
            label_text += f'ID: {face_label} '

        emotion_label = labels['emotion'][idx]
        if emotion_label != -1:
            label_text += f'emotion: {emotion_translation[int(emotion_label)]} '

        age_label = labels['age'][idx]
        if age_label != -1:
            label_text += f'age: {age_label} '

        gender_label = labels['gender'][idx]
        if gender_label != -1:
            label_text += f'gender: {gender_translation[int(gender_label)]} '

        race_label = labels['race'][idx]
        if race_label != -1:
            label_text += f'race: {race_translation[int(race_label)]} '
        
        if 'dataset_name' in labels:
            label_text += f'dataset: {labels["dataset_name"][idx]}'


        plt.subplot(4, 4, i+1)
        plt.imshow(image.permute(1, 2, 0))
        plt.title(label_text)
        plt.axis('off')

    plt.show()


if __name__ == '__main__':
    for key, value in config.items():
        print(f'{key}: {value}')
    main(**config)