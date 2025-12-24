import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import backbones.backbones as backbones
import datasets as db
import multitask.face_recognition_heads as face_recognition_heads
import eval, os, numpy as np
from torchsummary import summary
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
    return_name = False # Whether to return the name of the dataset or not.
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
    ms1mv2 = db.MS1MV2(transform=train_face_rec_transform, return_name = return_name)
    num_classes = ms1mv2.number_of_classes()

    train_db_list = [
        # Face recognition
        ms1mv2,
        # emotion recognition
        db.AffectNet(transform = train_transform, subset = 'train', return_name = return_name), 
        db.RAFDB(transform = train_transform, subset = 'train', return_name = return_name),

        # Age, Gender, and Race
        db.FairFace(transform = train_transform, subset = 'train', return_name = return_name), # gender and race
        db.UTKFace(transform = train_transform, subset = 'train', return_name = return_name), # age, gender, race
        db.MORPH(transform = train_transform, subset = 'train', return_name = return_name) # age, gender
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

    # Training setup
    model = MultiTaskFaceAnalysisModel(
        num_classes = num_classes,
        **kwargs
    )
    epochs = kwargs.get('max_epochs')
    optimizer_name = kwargs.get('optimizer')
    weight_decay = kwargs.get('weight_decay')
    lr_scheduler_name = kwargs.get('scheduler')
    milestones = kwargs.get('scheduler_milestones')
    start_factor = kwargs.get('start_factor')
    min_lr = kwargs.get('min_lr')
    warmup_epochs = kwargs.get('warmup_epochs')
    learning_rate = kwargs.get('learning_rate')


    # Separate parameters into backbone and other parts
    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    # Create parameter groups with different learning rates
    param_groups = [
        {'params': backbone_params, 'lr': learning_rate / 20},
        {'params': other_params, 'lr': learning_rate}
    ]


    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(param_groups, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(param_groups, lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    if lr_scheduler_name == 'cosine':
        linear = lr_scheduler.LinearLR(optimizer, start_factor=start_factor, end_factor=1.0, total_iters=warmup_epochs)
        cosine = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min = min_lr)
        scheduler = lr_scheduler.SequentialLR(optimizer, [linear, cosine], milestones=[warmup_epochs])
    elif lr_scheduler_name == 'multistep':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones)
    else:
        raise ValueError(f"Unsupported scheduler: {lr_scheduler_name}")

    scaler = torch.amp.GradScaler(device = device)
    dwa = DynamicWeightAverage(num_tasks = 4)
    weights = dwa.calculate_weights(avg_losses_current_epoch=None)



    model.to(device)
    # load the previous checkpoint if it exists:
    if checkpoint_path is not None and os.path.exists(checkpoint_path) and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")
    else:
        start_epoch = 0

    # Training loop
    model.train()
    for epoch in range(start_epoch, epochs):
        running_face_rec_loss = 0.0
        running_emotion_rec_loss = 0.0
        running_age_estimation_loss = 0.0
        running_gender_rec_loss = 0.0

        num_face_rec_samples = 0
        num_emotion_samples = 0
        num_age_gender_samples = 0

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")

        for i, batch in progress_bar:
            
            images = batch[0]
            labels = batch[1]

            # Extract data
            # Face Recognition
            face_rec_labels = labels['face_recognition']
            face_rec_mask = (face_rec_labels != -1)
            face_rec_images = images[face_rec_mask].to(device, non_blocking=True)
            face_rec_labels = face_rec_labels[face_rec_mask].to(device, non_blocking=True)
            
            # Emotion Recognition
            emotion_labels = labels['emotion_recognition']
            emotion_recognition_mask = (emotion_labels != -1)
            emotion_images = images[emotion_recognition_mask].to(device, non_blocking=True)
            emotion_labels = emotion_labels[emotion_recognition_mask].to(device, non_blocking=True)

            # Age estimation
            age_gender_labels = labels['age_estimation']
            age_gender_mask = (age_gender_labels['age'] != -1)
            age_gender_images = images[age_gender_mask].to(device, non_blocking=True)
            age_gender_labels = age_gender_labels[age_gender_mask].to(device, non_blocking=True)

            # Gender Recognition
            gender_recognition_labels = labels['gender_recognition']
            gender_recognition_mask = (gender_recognition_labels != -1)
            gender_recognition_images = images[gender_recognition_mask].to(device, non_blocking=True)
            gender_recognition_labels = gender_recognition_labels[gender_recognition_mask].to(device, non_blocking=True)

            # Race Recognition
            race_recognition_labels = labels['race_recognition']
            race_recognition_mask = (race_recognition_labels != -1)
            race_recognition_images = images[race_recognition_mask].to(device, non_blocking=True)
            race_recognition_labels = race_recognition_labels[race_recognition_mask].to(device, non_blocking=True)


            
            
if __name__ == '__main__':
    for key, value in config.items():
        print(f'{key}: {value}')
    main(**config)