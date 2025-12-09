import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import backbones.backbones as backbones
import datasets
import multitask.face_recognition_heads as face_recognition_heads
import eval, os, numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import lr_scheduler
from augmenter import Augmenter
from multitask.subnets import FaceRecognitionEmbeddingSubnet, GenderRecognitionSubnet, AgeEstimationSubnet, \
                              EmotionRecognitionSubnet
from datasets import MultiTaskDataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from multitask.models import MultiTaskFaceAnalysisModel
from configs.train_multitask import config
from multitask.loss_weighing import DynamicWeightAverage



torch.set_float32_matmul_precision('medium')



def main(**kwargs): 

    use_validation = True # toggle validation datasets on or off. 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = kwargs.get('resume_from_checkpoint')

    # train and test transforms
    train_face_rec_transform = transforms.Compose([ # for face recognition during training.
        transforms.ToPILImage(),
        Augmenter(crop_augmentation_prob=0.2, low_res_augmentation_prob=0.2, photometric_augmentation_prob=0.2),
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_transform = transforms.Compose([ # for other datasets during training.
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    test_transform = transforms.Compose([ # for testing on datasets other than face recognition.
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # data preparation
    generator = torch.Generator().manual_seed(42)

    # Face Recognition
    face_recognition_dataset = datasets.MS1MV2_Dataset()
    face_recognition_dataset.discard_classes(kwargs.get('min_num_images_per_class'))
    num_classes = face_recognition_dataset.num_classes # used later with the face recognition subnet
    face_recognition_dataset = datasets.TransformedDataset(face_recognition_dataset, train_face_rec_transform)


    # Emotion recognition
    raf_train_dataset = datasets.RAF_Dataset(RAF_subset = 'train')
    if use_validation:
        raf_train_dataset, raf_val_dataset = torch.utils.data.random_split(raf_train_dataset, lengths = [0.8, 0.2], generator = generator) # split the train dataset to create a validation dataset
        raf_val_dataset = datasets.TransformedDataset(raf_val_dataset, test_transform)
    raf_train_dataset = datasets.TransformedDataset(raf_train_dataset, train_transform)

    # Age and gender
    agedb_train_dataset = datasets.AgeDB_Dataset(subset = 'train')
    if use_validation:
        agedb_train_dataset, agedb_val_dataset = torch.utils.data.random_split(agedb_train_dataset, lengths = [0.8, 0.2], generator = generator)
        agedb_val_dataset = datasets.TransformedDataset(agedb_val_dataset, test_transform)
    agedb_train_dataset = datasets.TransformedDataset(agedb_train_dataset, train_transform)

    batch_size = kwargs.get('batch_size')
    batch_sizes = [0.5, 0.25, 0.25] # default ratios for each task/dataset
    batch_sizes = [int(batch_size * batch_sizes[i]) for i in range(len(batch_sizes))] 
    num_workers = kwargs.get('num_workers')

    
    train_dataloader = datasets.MultiTaskDataLoader(
        datasets = [face_recognition_dataset, raf_train_dataset, agedb_train_dataset],
        batch_sizes = batch_sizes,
        num_workers = num_workers
    )

    if use_validation:
        raf_val_dataloader = torch.utils.data.DataLoader(
            dataset = raf_val_dataset,
            batch_size = 32,
            shuffle = False,
            num_workers = num_workers,
            pin_memory = True,
        )

        agedb_val_dataloader = torch.utils.data.DataLoader(
            dataset = agedb_val_dataset,
            batch_size = 32,
            shuffle = False,
            num_workers=num_workers,
            pin_memory=True
        )


    # Training setup 
    model = MultiTaskFaceAnalysisModel(num_classes, **kwargs)
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
        {'params': backbone_params, 'lr': learning_rate / 10},
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

    # training loop
    for epoch in range(start_epoch, epochs):
        model.train()
        
        running_face_rec_loss = 0.0
        running_emotion_rec_loss = 0.0
        running_age_estimation_loss = 0.0
        running_gender_rec_loss = 0.0

        num_face_rec_samples = 0
        num_emotion_samples = 0
        num_age_gender_samples = 0

        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}/{epochs}")

        for i, batch in progress_bar:
            db1, db2, db3 = batch
            face_rec_images, face_rec_labels = db1
            emotion_images, emotion_labels = db2
            age_images, (_, age_labels, gender_labels) = db3

            num_face_rec_samples += face_rec_images.size(0)
            num_emotion_samples += emotion_images.size(0)
            num_age_gender_samples += age_images.size(0)

            split_indices = np.cumsum([0, face_rec_images.size(0), emotion_images.size(0), age_images.size(0)])

            # combine images
            x = torch.cat((face_rec_images, emotion_images, age_images), dim=0)

            # to cuda
            x = x.to(device, non_blocking=True)
            face_rec_labels = face_rec_labels.to(device, non_blocking=True)
            emotion_labels = emotion_labels.to(device, non_blocking=True)
            age_labels = age_labels.to(device, non_blocking=True).view(-1, 1)
            gender_labels = gender_labels.to(device, non_blocking=True).view(-1, 1).float()

            optimizer.zero_grad()

            # forward pass
            with torch.amp.autocast(device_type = device):
                (normalized_embedding, embedding_norm), emotion_output, age_output, gender_output = model(x)

                # extract outputs for each class
                ## face recognition
                normalized_embedding, embedding_norm = normalized_embedding[split_indices[0] : split_indices[1]], embedding_norm[split_indices[0]:split_indices[1]] 
                
                ## emotion recognition
                emotion_output = emotion_output[split_indices[1] : split_indices[2]]

                ## age and gender come from the same images since agedb has multiple labels. So the same split_indices will be used.
                ## age estimation
                age_output = age_output[split_indices[2] : split_indices[3]]

                ## gender recognition
                gender_output = gender_output[split_indices[2] : split_indices[3]]


                # Loss calculation
                ## face recognition
                face_rec_logits = model.margin_head(normalized_embedding, embedding_norm, face_rec_labels)
                face_rec_loss = F.cross_entropy(face_rec_logits, face_rec_labels)

                ## emotion recognition
                emotion_rec_loss = F.cross_entropy(emotion_output, emotion_labels)

                ## age estimation
                age_estimation_loss = F.l1_loss(age_output, age_labels)

                ## gender recognition
                gender_rec_loss = F.binary_cross_entropy_with_logits(gender_output,gender_labels)

                ## total_loss
                losses = [face_rec_loss, emotion_rec_loss, age_estimation_loss, gender_rec_loss]
                total_loss = 0
                for i in range(len(weights)):
                    total_loss += weights[i] * losses[i]


            
            # backward pass
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()


            running_face_rec_loss += face_rec_loss.item() * face_rec_images.size(0)
            running_emotion_rec_loss += emotion_rec_loss.item() * emotion_images.size(0)
            running_age_estimation_loss += age_estimation_loss.item() * age_images.size(0)
            running_gender_rec_loss += gender_rec_loss.item() * age_images.size(0)

            progress_bar.set_postfix(
                loss=f"{total_loss.item():.4f}",
                fr_loss=f"{face_rec_loss.item():.4f}",
                em_loss=f"{emotion_rec_loss.item():.4f}",
                age_loss=f"{age_estimation_loss.item():.4f}",
                gen_loss=f"{gender_rec_loss.item():.4f}",
                lr=f"{optimizer.param_groups[1]['lr']:.6f}",
                lr_backbone = f'{optimizer.param_groups[0]['lr']:.6f}',
                weights = f'{weights}'
            )

        scheduler.step()

        epoch_fr_loss = running_face_rec_loss / num_face_rec_samples
        epoch_em_loss = running_emotion_rec_loss / num_emotion_samples
        epoch_age_loss = running_age_estimation_loss / num_age_gender_samples
        epoch_gen_loss = running_gender_rec_loss / num_age_gender_samples
        weights = dwa.calculate_weights(np.array([epoch_fr_loss, epoch_em_loss, epoch_age_loss, epoch_gen_loss]))
        epoch_loss = epoch_fr_loss + epoch_em_loss + epoch_age_loss + epoch_gen_loss

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Total Loss: {epoch_loss:.4f}")
        print(f"  Face Recognition Loss: {epoch_fr_loss:.4f}")
        print(f"  Emotion Loss: {epoch_em_loss:.4f}")
        print(f"  Age Loss: {epoch_age_loss:.4f}")
        print(f"  Gender Loss: {epoch_gen_loss:.4f}")
        print(f"  Weights: {weights}")

        checkpoint = {
            'epoch' : epoch,
            'model_state_dict' : model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'scheduler_state_dict' : scheduler.state_dict(),
            'loss' : epoch_loss,
        }

        # Create the checkpoint file if it doesn't exist.
        if not os.path.exists(checkpoint_path):
            os.makedirs(os.path.split(checkpoint_path)[0], exist_ok = True)
            f = open(checkpoint_path, 'x')
        
        torch.save(
            checkpoint, 
            checkpoint_path
        )

        
        if use_validation:
            print('Validating...')
            model.eval()
            with torch.no_grad():
                # face recognition
                face_rec_model = nn.Sequential(
                    model.backbone,
                    model.face_recognition_embedding_subnet
                )
                metrics = eval.evaluate_backbone(face_rec_model, datasets_to_test=['CPLFW', 'CALFW'])    
                del face_rec_model #cleanup
                
                for key, db_metrics in metrics.items():
                    accuracy, _, _, f1_score, _, _, _, _ = db_metrics
                    print(f'Accuracy for {key} = {accuracy}.')
                    print(f'F1 score for {key} = {f1_score}')

                # Emotion recognition
                predictions = []
                labels = []
                for image, label in raf_val_dataloader:
                    image = image.to('cuda', non_blocking = True)
                    label = label.to('cuda', non_blocking = True)

                    emotion_logits = model.emotion_recognition_subnet(model.backbone(image))
                    emotion_prediction = torch.argmax(emotion_logits, dim = 1)
                    predictions.append(emotion_prediction)
                    labels.append(label)

                predictions = torch.cat(predictions, dim = 0)
                labels = torch.cat(labels, dim = 0)
                emotion_accuracy = (predictions == labels).float().mean().item()

                # cleanup
                del predictions, labels


                print(f'Accuracy for RAF = {emotion_accuracy}')
                print()

                # Age and Gender Estimation
                age_predictions, gender_predictions = [], []
                gender_labels, age_labels = [], []
                for image, (_, age_label, gender_label) in agedb_val_dataloader:
                    image = image.to('cuda', non_blocking = True)
                    age_label = age_label.to('cuda', non_blocking = True).view(-1, 1)
                    gender_label = gender_label.to('cuda', non_blocking = True).view(-1, 1).float()

                    age_prediction = model.age_estimation_subnet(model.backbone(image))
                    gender_prediction = model.gender_recognition_subnet(model.backbone(image))
                    age_predictions.append(age_prediction)
                    gender_predictions.append(gender_prediction)
                    age_labels.append(age_label)
                    gender_labels.append(gender_label)
                
                age_predictions = torch.cat(age_predictions, dim = 0)
                gender_predictions = torch.cat(gender_predictions, dim = 0)
                age_labels = torch.cat(age_labels, dim = 0)
                gender_labels = torch.cat(gender_labels, dim = 0)

                age_l1_score = F.l1_loss(age_predictions, age_labels).item()
                gender_accuracy = (torch.sigmoid(gender_predictions).round() == gender_labels).float().mean().item()

                # cleanup
                del age_predictions, gender_predictions, age_labels, gender_labels

                print(f'Age L1 score = {age_l1_score}')
                print(f'Gender accuracy = {gender_accuracy}')
                print()
    
    torch.save(
        model.state_dict(),
        os.path.join('data', 'models', 'multitask experiment number 1', 'model.pth')
    )


if __name__ == '__main__':
    for key, value in config.items():
        print(f'{key}: {value}')
    main(**config)