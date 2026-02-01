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
from augmentation import Augmenter, get_task_augmentation_transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from multitask.framework1.model import MultiTaskFaceAnalysisModel
from configs.train_swinv2_t_ms1mv2_face_age_gender_race_emotion_attribute_light_final import config
from multitask.loss_weighing import DynamicWeightAverage
from matplotlib import pyplot as plt
from utility_scripts.losses import dldl_loss, geodesic_loss, AsymmetricLossOptimized
from utility_scripts.pose_estimation_utilities import compute_rotation_matrix_from_ortho6d, euler_angles_to_matrix

torch.set_float32_matmul_precision('medium')

def main(**kwargs): 

    use_validation = True # toggle validation datasets on or off. 
    return_name = False # Whether to return the name of the dataset or not.
    output_folder_name = kwargs.get('output_folder_name')
    checkpoint_path = os.path.join('checkpoints', output_folder_name, 'model.pth')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # train and test transforms
    transforms = get_task_augmentation_transforms()
    
    train_face_rec_transform = transforms['face_recognition']
    emotion_recognition_transform = transforms['emotion_recognition']
    age_gender_race_transform = transforms['age_gender_race_recognition']
    attribute_recognition_transform = transforms['attribute_recognition']
    # pose_estimation_transform = transforms['head_pose_estimation']
    

    test_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale = True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Data Preparation
    
    # training
    ms1mv2 = db.MS1MV2(transform=train_face_rec_transform, return_name = return_name)
    num_classes = ms1mv2.number_of_classes()
    celeba = db.CelebA(transform = attribute_recognition_transform, subset = 'combined')
    attribute_pos_weight = celeba.get_attribute_weights().to(device)

    train_db_dict = {
        'face_recognition' : [ms1mv2],
        'emotion_recognition' : [
            db.FERPlus(transform = emotion_recognition_transform, subset = 'combined'),
            db.AffectNet(transform = emotion_recognition_transform, subset = 'combined'), 
            db.RAFDB(transform = emotion_recognition_transform, subset = 'train')
        ],
        'age_gender_race_recognition' : [
            db.MORPH(transform = age_gender_race_transform, subset = 'combined'),
            db.FairFace(transform = age_gender_race_transform, subset = 'train'),
            db.UTKFace(transform = age_gender_race_transform, subset = 'train'),
            db.IMDB_WIKI(transform = age_gender_race_transform)
        ],
        'attribute_recognition' : [celeba],
        # 'head_pose_estimation' : [db.W300LP(transform = pose_estimation_transform)],
    }
    
    batch_size = kwargs.get('batch_size')
    num_workers = kwargs.get('num_workers')

    train_loader = db.get_balanced_loader(
        train_db_dict,
        batch_size = batch_size, 
        num_workers = num_workers,
        epoch_size = int(kwargs.get('epoch_size')),
    )


    if use_validation:
        # validation
        fairface_test_db = torch.utils.data.DataLoader( # for race recognition and gender recognition
            dataset = db.FairFace(transform = test_transform, subset = 'test'),
            batch_size = 64,
            shuffle = True,
            num_workers = 2,
            pin_memory = True,
        )

        rafdb_validation_db = torch.utils.data.DataLoader( # for emotion recognition
            dataset = db.RAFDB(transform = test_transform, subset = 'test'),
            batch_size = 64,
            shuffle = True,
            num_workers = 2,
            pin_memory = True,
        )

        morph_test_db = torch.utils.data.DataLoader( # for age estimation and gender recognition 
            dataset = db.MORPH(transform = test_transform, subset = 'test'),
            batch_size = 64,
            shuffle = True,
            num_workers = 2,
            pin_memory = True,
        )

        celeba_validation_db = torch.utils.data.DataLoader( # for attribute recognition
            dataset = db.CelebA(transform = test_transform, subset = 'test'),
            batch_size = 64,
            shuffle = True,
            num_workers = 2,
            pin_memory = True,
        )

        # biwi = torch.utils.data.DataLoader( # for head pose estimation
        #     dataset=db.BIWI(transform=test_transform),
        #     batch_size=64,
        #     shuffle=True,
        #     num_workers=2,
        #     pin_memory=True,
        # )


        print('fairface length: ', len(fairface_test_db) * 64)
        print('affectnet length: ', len(rafdb_validation_db) * 64)
        print('morph length: ', len(morph_test_db) * 64)
        print('celeba length: ', len(celeba_validation_db) * 64)
        # print('biwi length: ', len(biwi) * 64)

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
    freeze_backbone_epochs = kwargs.get('freeze_backbone_epochs')
    learning_rate = kwargs.get('learning_rate')


    # Separate parameters into backbone and other parts and count the parameters
    margin_head_params = []
    face_rec_params = []
    backbone_params = []
    other_params = []
    backbone_params_count = 0
    face_rec_subnet_parameters_count = 0
    margin_head_params_count = 0
    emotion_rec_subnet_parameters_count = 0
    age_estimation_subnet_parameters_count = 0
    gender_rec_subnet_parameters_count = 0
    race_rec_subnet_parameters_count = 0
    attribute_rec_subnet_parameters_count = 0
    # pose_estimation_subnet_parameters_count = 0

    
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
            backbone_params_count += param.numel()

        elif 'face_recognition_embedding_subnet' in name:
            face_rec_params.append(param)
            face_rec_subnet_parameters_count += param.numel()

        elif 'margin_head' in name:
            margin_head_params.append(param)
            margin_head_params_count += param.numel()

        elif 'emotion_recognition_subnet' in name:
            other_params.append(param)
            emotion_rec_subnet_parameters_count += param.numel()

        elif 'age_estimation_subnet' in name:
            other_params.append(param)
            age_estimation_subnet_parameters_count += param.numel()

        elif 'gender_recognition_subnet' in name:
            other_params.append(param)
            gender_rec_subnet_parameters_count += param.numel()

        elif 'race_recognition_subnet' in name:
            other_params.append(param)
            race_rec_subnet_parameters_count += param.numel()

        elif 'attribute_recognition_subnet' in name:
            other_params.append(param)
            attribute_rec_subnet_parameters_count += param.numel()

        # elif 'pose_estimation_subnet' in name:
        #     other_params.append(param)
        #     pose_estimation_subnet_parameters_count += param.numel()

        else:
            other_params.append(param)
    
    assert len(backbone_params) != 0, 'No backbone parameters found!'
    assert len(face_rec_params) != 0, 'No face recognition subnet parameters found!'
    assert len(margin_head_params) != 0, 'No margin head parameters found!'
    assert len(other_params) != 0, 'No subnet parameters found!'


    print(f'Backbone parameters: {backbone_params_count} <'.ljust(75, '='))
    print(f'Face recognition subnet parameters: {face_rec_subnet_parameters_count} <'.ljust(75, '='))
    print(f'Margin head parameters:  {margin_head_params_count} <'.ljust(75, '='))
    print(f'Emotion recognition subnet parameters: {emotion_rec_subnet_parameters_count} <'.ljust(75, '='))
    print(f'Age estimation subnet parameters: {age_estimation_subnet_parameters_count} <'.ljust(75, '='))
    print(f'Gender recognition subnet parameters: {gender_rec_subnet_parameters_count} <'.ljust(75, '='))
    print(f'Race recognition subnet parameters: {race_rec_subnet_parameters_count} <'.ljust(75, '='))
    print(f'Attribute recognition subnet parameters: {attribute_rec_subnet_parameters_count} <'.ljust(75, '='))
    # print(f'Head pose estimation subnet parameters: {pose_estimation_subnet_parameters_count} <'.ljust(75, '='))
    print(f'Total (excluding the margin head parameters): {backbone_params_count + 
                    face_rec_subnet_parameters_count + 
                    emotion_rec_subnet_parameters_count + 
                    age_estimation_subnet_parameters_count + 
                    gender_rec_subnet_parameters_count + 
                    race_rec_subnet_parameters_count + 
                    attribute_rec_subnet_parameters_count}')

    # Create parameter groups with different learning rates
    param_groups = [
        {'params': backbone_params, 'lr': learning_rate * float(kwargs.get('backbone_lr_multiplier'))},
        {'params': face_rec_params, 'lr': learning_rate},
        {'params': margin_head_params, 'lr': learning_rate},
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


    precision_mode = kwargs.get('precision')

    if precision_mode == 'bf16-mixed':
        amp_dtype = torch.bfloat16
        use_scaler = False # no need to use scaler since bf16-mixed has enough range
    elif precision_mode == '32':
        amp_dtype = torch.float32
        use_scaler = False
    else:
        amp_dtype == torch.float16
        use_scaler = True
    print(f'Using amp dtype: {amp_dtype}')

    scaler = torch.amp.GradScaler(device = device, enabled = use_scaler)
    dwa = DynamicWeightAverage(num_tasks = 6)
    losses_weights = dwa.calculate_weights(avg_losses_current_epoch=None)
    losses_weights_history = [losses_weights]

    # asymmetric loss for attribute recognition
    asymmetric_loss = AsymmetricLossOptimized()



    model.to(device)
    # load the previous checkpoint if it exists:
    if checkpoint_path is not None and os.path.exists(checkpoint_path) and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only = False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        losses_weights_history = checkpoint['losses_weights_history']
        dwa = checkpoint['dwa']
        losses_weights = checkpoint['losses_weights']

        print(f"Resuming training from epoch {start_epoch}")
    else:
        start_epoch = 0

    # Training loop
    for epoch in range(start_epoch, epochs):
        model.train() # Put here to ensure the epoch starts with the model in training mode

        if epoch < freeze_backbone_epochs:
            print(f"🔒 Epoch {epoch}: Backbone + face recognition subnet + margin head are FROZEN (Warmup Phase)")            

            for param in backbone_params:
                param.requires_grad = False
            
            for param in face_rec_params:
                param.requires_grad = False
            
            for param in margin_head_params:
                param.requires_grad = False

            for param in other_params: # just to make sure the head parameters are trainable
                param.requires_grad = True

        elif epoch == freeze_backbone_epochs:
            print(f"🔓 Epoch {epoch}: Unfreezing! (Finetuning Phase)")
            for param in backbone_params:
                param.requires_grad = True
            
            for param in face_rec_params:
                param.requires_grad = True
            
            for param in margin_head_params:
                param.requires_grad = True
            
            

        running_face_rec_loss = 0.0
        running_emotion_rec_loss = 0.0
        running_age_estimation_loss = 0.0
        running_gender_rec_loss = 0.0
        running_race_rec_loss = 0.0
        running_attribute_rec_loss = 0.0
        # running_pose_estimation_loss = 0.0



        num_face_rec_samples = 0
        num_emotion_samples = 0
        num_age_samples = 0
        num_gender_samples = 0
        num_race_samples = 0
        num_attribute_samples = 0
        # num_pose_samples = 0



        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")

        for counter, batch in progress_bar:
            
            images = batch[0]
            labels = batch[1]
            images = images.to(device, non_blocking=True) # The labels are in a dict format, which cannot be moved to cuda. Task labels will be extracted below and converted to cuda one by one.


            # Initialize the loss tensors at the start
            # We use torch.tensor(0.0) so we can use .item() later safely
            face_rec_loss = torch.tensor(0.0, device=device)
            emotion_loss = torch.tensor(0.0, device=device)
            age_loss = torch.tensor(0.0, device=device)
            gender_loss = torch.tensor(0.0, device=device)
            race_loss = torch.tensor(0.0, device=device)
            attribute_loss = torch.tensor(0.0, device=device)
            # pose_loss = torch.tensor(0.0, device=device)

            optimizer.zero_grad(set_to_none = True)

            # forward pass
            with torch.amp.autocast(device_type = device, dtype = amp_dtype, enabled = (precision_mode != '32')):

                (normalized_embedding, embedding_norm), \
                emotion_output, age_output, \
                gender_output, race_output, \
                attribute_output = model(images)

                # Face recognition loss
                face_recognition_labels = labels['face_recognition'].to(device, non_blocking=True)
                face_recognition_mask = face_recognition_labels != -1

                if face_recognition_mask.sum() > 0: # check if we have face recognition samples in this batch
                    valid_embeddings = normalized_embedding[face_recognition_mask]
                    valid_embedding_norms = embedding_norm[face_recognition_mask]
                    valid_labels = face_recognition_labels[face_recognition_mask]

                    face_rec_logits = model.get_face_recognition_logits(valid_embeddings, valid_embedding_norms, valid_labels)
                    face_rec_loss = F.cross_entropy(face_rec_logits, valid_labels)
                    num_face_rec_samples += face_recognition_mask.sum().item()

                
                
                # Emotion recognition
                emotion_labels = labels['emotion'].to(device, non_blocking=True)
                emotion_mask = emotion_labels != -1
                if emotion_mask.sum() > 0: # check if we have emotion recognition samples in this batch
                    emotion_loss = F.cross_entropy(emotion_output[emotion_mask], emotion_labels[emotion_mask]) * 5 # scale to match other losses
                    num_emotion_samples += emotion_mask.sum().item()
                



                # Age estimation
                age_labels = labels['age'].to(device, non_blocking=True)
                age_mask = age_labels != -1
                if age_mask.sum() > 0: # check if we have age estimation samples in this batch
                    age_loss = dldl_loss(age_output[age_mask], age_labels[age_mask])
                    # age_loss = F.l1_loss(age_output[age_mask], age_labels[age_mask].view(-1, 1))
                    num_age_samples += age_mask.sum().item()



                # Gender recognition
                gender_labels = labels['gender'].to(device, non_blocking=True)
                gender_mask = gender_labels != -1
                if gender_mask.sum() > 0: # check if we have gender recognition samples in this batch
                    gender_loss = F.binary_cross_entropy_with_logits(gender_output[gender_mask], gender_labels[gender_mask].view(-1, 1).float())
                    num_gender_samples += gender_mask.sum().item()
                

                
                # Race recognition
                race_labels = labels['race'].to(device, non_blocking=True)
                race_mask = race_labels != -1
                if race_mask.sum() > 0: # check if we have race recognition samples in this batch
                    race_loss = F.cross_entropy(race_output[race_mask], race_labels[race_mask]) * 5 # scale to match other losses                                                                                                                                                   
                    num_race_samples += race_mask.sum().item()

                
                # Attribute recognition
                attribute_labels = labels['attributes'].to(device, non_blocking=True)
                # CelebA returns either 0 or 1 for valid sample, where 0 = the attribute doesn't exist, 1 = attribute exists. The default label for dummy attributes is -1,
                # so check if the first attribute from all the samples is not -1 to see which one is a valid label. 
                attibute_mask = attribute_labels[:, 0] != -1
                if attibute_mask.sum() > 0:
                    # # Alternative 1: Asymmetric Loss
                    # attribute_loss = asymmetric_loss(
                    #     attribute_output[attibute_mask], 
                    #     attribute_labels[attibute_mask].view(-1, 40).float()
                    # )
                    
                    # Alternative 2: Binary Cross Entropy (BCE)
                    raw_loss = F.binary_cross_entropy_with_logits(
                        attribute_output[attibute_mask], 
                        attribute_labels[attibute_mask].view(-1, 40).float(),
                        pos_weight=attribute_pos_weight,
                        reduction='none' # <--- get raw loss (without reduction)
                    )
                    # Sum over attributes (dim=1)
                    loss_per_image = raw_loss.sum(dim=1) 
                    
                    # mean over the batch
                    attribute_loss = loss_per_image.mean()
                    num_attribute_samples += attibute_mask.sum().item()  
 


                

                # # head pose estimation
                # pose_labels = labels['pose'].to(device, non_blocking=True)
                # pose_mask = pose_labels[:, 0] != -999 # same logic as attribute recognition
                # if pose_mask.sum() > 0: # check if we have head pose estimation samples in this batch
                #     # turn the raw angles into a rotation matrix
                #     target_rotation_matrices = euler_angles_to_matrix(pose_labels[pose_mask])
                #     # the model predicts two x and y vectors that are orthogonal to each other. Turn them into a rotation matrix.
                #     predicted_rotation_matrices = compute_rotation_matrix_from_ortho6d(pose_output[pose_mask])
                #     pose_loss = geodesic_loss(predicted_rotation_matrices, target_rotation_matrices)
                #     num_pose_samples += pose_mask.sum().item()
                                
                losses = [face_rec_loss, emotion_loss, age_loss, gender_loss, race_loss, attribute_loss]

                total_loss = 0
                for i in range(len(losses_weights)):
                    total_loss += losses_weights[i] * losses[i]
            
            
            # backward pass
            if total_loss != 0:
                scaler.scale(total_loss).backward()
                
                # gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=kwargs.get('gradient_clip_val'))

                # update weights
                scaler.step(optimizer)
                scaler.update()
            
            running_face_rec_loss += face_rec_loss.item() * face_recognition_mask.sum().item()
            running_emotion_rec_loss += emotion_loss.item() * emotion_mask.sum().item()
            running_age_estimation_loss += age_loss.item() * age_mask.sum().item()
            running_gender_rec_loss += gender_loss.item() * gender_mask.sum().item()
            running_race_rec_loss += race_loss.item() * race_mask.sum().item()
            running_attribute_rec_loss += attribute_loss.item() * attibute_mask.sum().item()
            # running_pose_estimation_loss += pose_loss.item() * pose_mask.sum().item()
            

            progress_bar.set_postfix(ordered_dict={
                'loss' : f"{total_loss.item():.4f}",
                'fr' : f"{face_rec_loss.item():.4f}",
                'em' : f"{emotion_loss.item():.4f}",
                'age' : f"{age_loss.item():.4f}",
                'gen' : f"{gender_loss.item():.4f}",
                'race' : f"{race_loss.item():.4f}",
                'attr' : f"{attribute_loss.item():.4f}",
                # 'pose' : f"{pose_loss.item():.4f}",
                'lr' : f"{optimizer.param_groups[3]['lr']:.7f}",
                'lr_back' : f'{optimizer.param_groups[0]['lr']:.7f}',
            })

        scheduler.step()

        epoch_fr_loss = running_face_rec_loss / num_face_rec_samples
        epoch_em_loss = running_emotion_rec_loss / num_emotion_samples
        epoch_age_loss = running_age_estimation_loss / num_age_samples
        epoch_gen_loss = running_gender_rec_loss / num_gender_samples
        epoch_race_loss = running_race_rec_loss / num_race_samples
        epoch_attr_loss = running_attribute_rec_loss / num_attribute_samples
        # epoch_pose_loss = running_pose_estimation_loss / num_pose_samples
        losses_weights = dwa.calculate_weights(np.array([epoch_fr_loss, epoch_em_loss, epoch_age_loss, epoch_gen_loss, epoch_race_loss, epoch_attr_loss]))
        losses_weights_history.append(losses_weights)
        epoch_loss = epoch_fr_loss + epoch_em_loss + epoch_age_loss + epoch_gen_loss + epoch_race_loss + epoch_attr_loss


        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Total Loss: {epoch_loss:.4f}")
        print(f"  Face Recognition Loss: {epoch_fr_loss:.4f}")
        print(f"  Emotion Loss: {epoch_em_loss:.4f}")
        print(f"  Age Loss: {epoch_age_loss:.4f}")
        print(f"  Gender Loss: {epoch_gen_loss:.4f}")
        print(f"  Race Loss: {epoch_race_loss:.4f}")
        print(f"  Attribute Loss: {epoch_attr_loss:.4f}")
        # print(f"  Pose Loss: {epoch_pose_loss:.4f}")
        print(f"  Losses Weights: {losses_weights}")

        checkpoint = { 
            'epoch' : epoch,
            'model_state_dict' : model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'scheduler_state_dict' : scheduler.state_dict(),
            'loss' : epoch_loss,
            'losses_weights_history' : losses_weights_history,
            'dwa' : dwa,
            'losses_weights' : losses_weights
        }

        # Create the checkpoint file if it doesn't exist.
        if not os.path.exists(checkpoint_path):
            os.makedirs(os.path.split(checkpoint_path)[0], exist_ok = True) # create directory
            f = open(checkpoint_path, 'x') # create file
        
        torch.save(
            checkpoint, 
            checkpoint_path
        )

        # Validation
        if use_validation:
            model.eval()
            print(' Validating '.center(100, '='))
            
            # Face Recognition
            face_rec_model = nn.Sequential(
                model.backbone,
                model.face_recognition_embedding_subnet
            )
            metrics = eval.evaluate_face_recognition(face_rec_model, datasets_to_test = ['CPLFW', 'CALFW'])

            for key, db_metrics in metrics.items():
                accuracy, _, _, f1_score, _, _, _, _ = db_metrics
                print(f'Accuracy for {key} = {accuracy}.')

            # Emotion Recognition
            emotion_accuracy, _, _, _, _= eval.evaluate_emotion(model = model, dataloader = rafdb_validation_db)
            print(f'Accuracy for RAFDB (emotion recognition) = {emotion_accuracy}.')

            # Age estimation
            age_mae, _, _ = eval.evaluate_age(model = model, dataloader = morph_test_db)
            print(f'MAE for MORPH (age estimation) = {age_mae}.')

            # Gender recognition
            gender_accuracy, _, _, _, _, _, _, _, _ = eval.evaluate_gender(model = model, dataloader = morph_test_db)
            print(f'Accuracy for MORPH (gender recognition) = {gender_accuracy}.')

            gender_accuracy, _, _, _, _, _, _, _, _ = eval.evaluate_gender(model = model, dataloader = fairface_test_db)
            print(f'Accuracy for FairFace (gender recognition) = {gender_accuracy}.')

            # Race recognition
            race_accuracy, _, _, _, _ = eval.evaluate_race(model = model, dataloader = fairface_test_db, device = device)
            print(f'Accuracy for FairFace (race recognition) = {race_accuracy}.')

            # Attribute recognition
            attribute_accuracy, attribute_f1, _, _ = eval.evaluate_attributes(model = model, dataloader = celeba_validation_db)
            print(f'Accuracy for CelebA (attribute recognition) = {attribute_accuracy}.')
            print(f'F1 score for CelebA (attribute recognition) = {attribute_f1}')

            # # Head pose estimation 
            # mae_mean, _, _, _ = eval.evaluate_head_pose(model = model, dataloader = biwi)
            # print(f'MAE for BIWI (head pose estimation) = {mae_mean}.')
    

    # Save the final model
    output_model_path = os.path.join('data', 'models', output_folder_name)
    os.makedirs(output_model_path, exist_ok = True)
    torch.save(
        model.state_dict(), 
        os.path.join(output_model_path, 'model.pth')
    )

    # Plot dynamic weight average history
    task_names = ['Face Recognition', 'Emotion Recognition', 'Age Estimation', 'Gender Recognition', 'Race Recognition', 'Attribute Recognition', 'Head pose estimation']
    plt.figure(figsize=(10, 5), dpi=600)
    plt.title('The history of the loss weights per task using Dynamic Weight Average.')
    
    weight_history = np.array(losses_weights_history)
    
    # Create an X-axis array (e.g., Epoch 1, 2, 3...)
    epochs_x = range(1, weight_history.shape[0] + 1)

    for i in range(weight_history.shape[1]):
        # Plot X (Epochs) vs Y (Weights for task i)
        plt.plot(epochs_x, weight_history[:, i], label=task_names[i])
        
    plt.legend(loc='best')
    plt.xlabel('Epoch')
    plt.ylabel('Task weight')
    
    # Ensure directory exists before saving
    os.makedirs(os.path.join('data', 'figures', 'MultiTask Training', output_folder_name), exist_ok=True)
    
    plt.savefig(
        os.path.join('data', 'figures', 'MultiTask Training', output_folder_name, 'Task_loss_weights_history.png')
    )
    plt.close()

            
if __name__ == '__main__':
    for key, value in config.items():
        print(f'{key}: {value}')
    main(**config)