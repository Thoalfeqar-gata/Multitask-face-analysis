import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import backbones.backbones as backbones
import datasets
import multitask.face_recognition_heads as face_recognition_heads
import eval, os
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import lr_scheduler
from augmenter import Augmenter
from multitask.subnets import FaceRecognitionEmbeddingSubnet, GenderRecognitionSubnet, AgeEstimationSubnet, \
                              EmotionRecognitionSubnet
from datasets import MultiTaskDataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


torch.set_float32_matmul_precision('medium')


class MultiTaskFaceAnalysisModel(nn.Module):
    def __init__(self, **kwargs):
        """
            Args:
            kwargs: contains all the hyperparameters.
        """
        super().__init__()
        # Backbone args
        self.backbone_name = kwargs.get('backbone_name')
        self.pretrained_backbone_path = kwargs.get('pretrained_backbone_path')
        self.imagenet_pretrained = kwargs.get('pretrained')

        # Face recognition args
        self.head_type = kwargs.get('head_type')
        self.embedding_dim = kwargs.get('embedding_dim')
        self.num_classes = kwargs.get('num_classes')
        self.margin = kwargs.get('margin')
        self.scale = kwargs.get('scale')
        self.h = kwargs.get('h')
        self.t_alpha = kwargs.get('t_alpha')

        ##########
        # Backbone
        ##########

        # Obtain the backbone and load the weights if specified
        self.backbone = backbones.get_backbone(
            backbone_name = self.backbone_name, 
            imagenet_pretrained=self.imagenet_pretrained, 
            embedding_dim=self.embedding_dim
        )

        if self.pretrained_backbone_path is not None:
            self.backbone.load_state_dict(torch.load(self.pretrained_backbone_path))
        

        ##########
        # Subnets
        ##########

        # Face Recognition

        if self.backbone_name in ['swin_b', 'swin_v2_b', 'davit_b']:
            self.feature_embedding_dim = 1024
            self.transformer_embedding_dim = 128
        else:
            self.feature_embedding_dim = 768
            self.transformer_embedding_dim = 96

        self.face_recognition_embedding_subnet = FaceRecognitionEmbeddingSubnet(
            feature_embedding_dim=self.feature_embedding_dim,
        )

        self.margin_head = face_recognition_heads.build_head(
            head_type = self.head_type,
            embedding_size = self.embedding_dim,
            classnum = self.num_classes,
            m = self.margin,
            s = self.scale,
            h = self.h,
            t_alpha = self.t_alpha
        )

        # Emotion Recognition
        self.emotion_recognition_subnet = EmotionRecognitionSubnet(
            transformer_embedding_dim = self.transformer_embedding_dim
        )

        # Age Estimation
        self.age_estimation_subnet = AgeEstimationSubnet(
            transformer_embedding_dim=self.transformer_embedding_dim
        )

        # Gender Recognition
        self.gender_recognition_subnet = GenderRecognitionSubnet(
            transformer_embedding_dim=self.transformer_embedding_dim
        )
    

    def forward(self, x):
        
        multiscale_features = self.backbone(x)
        
        # Face recognition
        normalized_embedding, embedding_norm = self.face_recognition_embedding_subnet(multiscale_features)
        
        # Emotion Recognition
        emotion_output = self.emotion_recognition_subnet(multiscale_features)
        
        # Age Estimation
        age_output = self.age_estimation_subnet(multiscale_features)
        
        # Gender Recognition
        gender_output = self.gender_recognition_subnet(multiscale_features)
        
        return (normalized_embedding, embedding_norm), emotion_output, age_output, gender_output


class FaceRecognitionDataModule:
    def __init__(self, **kwargs):
        super().__init__()
        
        self.batch_size =kwargs.get('batch_size')
        self.min_num_images_per_class = kwargs.get('min_num_images_per_class')
        self.val_datasets = kwargs.get('val_datasets')
        self.num_workers = kwargs.get('num_workers')

        
        self.face_recognition_train_transform = transforms.Compose([
        transforms.ToPILImage(),
        Augmenter(),
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
        self.image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        self.val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
        self.train_loader = MultiTaskDataLoader(
            image_transform=self.image_transform,
            face_recognition_image_transform=self.face_recognition_train_transform,
            batch_size = self.batch_size,
            min_num_images_per_class=self.min_num_images_per_class
        )
        self.num_classes = self.train_loader.face_recognition_dataset.num_classes

        if self.val_datasets:
            self.instantiated_val_datasets = []
            for val_dataset_name in self.val_datasets:
                val_dataset_class = getattr(datasets, val_dataset_name)
                self.instantiated_val_datasets.append(val_dataset_class(image_transform=self.val_transform))


    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        if not hasattr(self, 'instantiated_val_datasets') or not self.instantiated_val_datasets:
            return None
        
        val_loaders = []
        for val_dataset in self.instantiated_val_datasets:
            val_loaders.append(
                DataLoader(val_dataset, batch_size=self.batch_size, 
                                  num_workers=self.num_workers, pin_memory=True)
            )
        return val_loaders


def main(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(args.seed)

    data_module = FaceRecognitionDataModule(**vars(args))
    
    train_loader = data_module.train_dataloader()
    val_loaders = data_module.val_dataloader()

    model = MultiTaskFaceAnalysisModel(num_classes=data_module.num_classes, **vars(args))
    model.to(device)

    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")

    if args.scheduler == 'cosine':
        scheduler1 = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=args.start_factor,
            total_iters=args.warmup_epochs
        )
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.max_epochs - args.warmup_epochs,
            eta_min=args.min_lr
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[scheduler1, scheduler2],
            milestones=[args.warmup_epochs]
        )
    elif args.scheduler == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=args.scheduler_milestones
        )
    else:
        raise ValueError(f"Scheduler {args.scheduler} not supported")

    
    writer = SummaryWriter(log_dir=os.path.join('logs', args.backbone_name, args.head_type, 'multitask_training_test'))
    
    scaler = torch.amp.GradScaler(device = 'cuda', enabled=args.precision == '16-mixed')
    
    start_epoch = 0
    if args.resume_from_checkpoint:
        checkpoint = torch.load(args.resume_from_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    for epoch in range(start_epoch, args.max_epochs):
        model.train()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.max_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            
            face_images, face_labels = batch['face recognition']
            emotion_images, emotion_labels = batch['emotion recognition']
            age_images, (age, gender) = batch['age and gender']
            
            face_images = face_images.to(device)
            face_labels = torch.tensor(face_labels, device=device)
            emotion_images = emotion_images.to(device)
            emotion_labels = emotion_labels.to(device)
            age_images = age_images.to(device)
            age = age.to(device)
            gender = gender.to(device)

            images = torch.cat([face_images, emotion_images, age_images], dim=0)

            with torch.amp.autocast(device_type = 'cuda', enabled=args.precision == '16-mixed'):
                (normalized_embedding, embedding_norm), emotion_output, age_output, gender_output = model(images)

                extraction_indices = [0]
                for size in [face_images.size(0), emotion_images.size(0), age_images.size(0)]:
                    extraction_indices.append(extraction_indices[-1] + size)

                face_recognition_output = normalized_embedding[extraction_indices[0]:extraction_indices[1]]
                face_recognition_embedding_norm = embedding_norm[extraction_indices[0]:extraction_indices[1]]

                emotion_output = emotion_output[extraction_indices[1]:extraction_indices[2]]

                age_output = age_output[extraction_indices[2]:extraction_indices[3]]
                gender_output = gender_output[extraction_indices[2]:extraction_indices[3]]

                output = model.margin_head(face_recognition_output, face_recognition_embedding_norm, face_labels)
                face_loss = F.cross_entropy(output, face_labels)

                emotion_loss = F.cross_entropy(emotion_output, emotion_labels)
                age_loss = F.l1_loss(age_output, age.unsqueeze(1).float())
                gender_loss = F.binary_cross_entropy_with_logits(gender_output, gender.unsqueeze(1).float())

                total_loss = face_loss + emotion_loss + age_loss + gender_loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            progress_bar.set_postfix({
                'face_loss': face_loss.item(),
                'emotion_loss': emotion_loss.item(),
                'age_loss': age_loss.item(),
                'gender_loss': gender_loss.item(),
                'total_loss': total_loss.item(),
                'lr': optimizer.param_groups[0]['lr']
            })
            
            writer.add_scalar('Loss/face', face_loss.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('Loss/emotion', emotion_loss.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('Loss/age', age_loss.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('Loss/gender', gender_loss.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('Loss/total', total_loss.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch * len(train_loader) + batch_idx)

        scheduler.step()

        if val_loaders:
            model.eval()
            with torch.no_grad():
                for dataloader_idx, val_loader in enumerate(val_loaders):
                    validation_outputs = []
                    for batch in tqdm(val_loader, desc=f"Validating on {args.val_datasets[dataloader_idx]}"):
                        (image1_batch, image2_batch), label_batch = batch
                        image1_batch = image1_batch.to(device)
                        image2_batch = image2_batch.to(device)
                        label_batch = label_batch.to(device)

                        embeddings1, norm1 = model.face_recognition_embedding_subnet(model.backbone(image1_batch))
                        embeddings2, norm2 = model.face_recognition_embedding_subnet(model.backbone(image2_batch))
                        flipped_embeddings1, flipped_norm1 = model.face_recognition_embedding_subnet(model.backbone(torch.flip(image1_batch, dims=[3])))
                        flipped_embeddings2, flipped_norm2 = model.face_recognition_embedding_subnet(model.backbone(torch.flip(image2_batch, dims=[3])))

                        embeddings1 = embeddings1 * norm1
                        embeddings2 = embeddings2 * norm2
                        flipped_embeddings1 = flipped_embeddings1 * flipped_norm1
                        flipped_embeddings2 = flipped_embeddings2 * flipped_norm2

                        embeddings1 = (embeddings1 + flipped_embeddings1)
                        embeddings2 = (embeddings2 + flipped_embeddings2)

                        embeddings1 = F.normalize(embeddings1, p=2, dim=1)
                        embeddings2 = F.normalize(embeddings2, p=2, dim=1)

                        diff = embeddings1 - embeddings2
                        dist = torch.sum(torch.square(diff), axis=1)
                        validation_outputs.append({'distances': dist, 'labels': label_batch})

                    distances = torch.cat([x['distances'] for x in validation_outputs]).cpu().float().numpy()
                    labels = torch.cat([x['labels'] for x in validation_outputs]).cpu().float().numpy()
                    
                    dataset_name = args.val_datasets[dataloader_idx]
                    metrics_results = eval.get_metrics_from_distances({dataset_name: (distances, labels)})
                    mean_accuracy, _, _, _, _, _, _, _ = metrics_results[dataset_name]
                    
                    writer.add_scalar(f'Val_Acc/{dataset_name}', mean_accuracy, epoch)
                    print(f"Epoch {epoch+1} - {dataset_name} Val Acc: {mean_accuracy:.4f}")

        
        checkpoint_path = os.path.join('checkpoints', args.backbone_name, args.head_type, 'multitask_training_test', f'epoch_{epoch}.pth')
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, checkpoint_path)

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Multitask training test')

    # Model Hyperparameters
    parser.add_argument('--backbone_name', type=str, default='swin_t', help='Backbone model name')
    parser.add_argument('--pretrained', type = int, default = 0, help='Use pretrained weights 0 = False, 1 = True')
    parser.add_argument('--head_type', type=str, default='adaface', help='Head name (e.g., arcface, cosface, adaface)')
    parser.add_argument('--embedding_dim', type=int, default=512, help='Embedding dimension')
    parser.add_argument('--optimizer', type=str, default='adamw', help='Optimizer name (can be either adamw or sgd)')
    parser.add_argument('--weight_decay', type = float, default = 5e-4, help = 'Weight decay for the optimizer')
    parser.add_argument('--scheduler', type=str, default='cosine', help='Learning rate scheduler name (can be either cosine or multistep)')
    parser.add_argument('--scheduler_milestones', type=int, nargs='+', default=[20, 30], help='Milestones for the multistep scheduler (not needed for cosine)')
    parser.add_argument('--start_factor', type=float, default=0.1, help='Start factor for the linear warmup learning rate scheduler')
    parser.add_argument('--min_lr', type=float, default=5e-6, help='Minimum learning rate for the cosine scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of warmup epochs')
    parser.add_argument('--margin', type=float, default=0.5, help='Margin for the loss function')
    parser.add_argument('--scale', type=float, default=64.0, help='Scale for the loss function')
    parser.add_argument('--h', type=float, default=0.333, help='h parameter for AdaFace loss')
    parser.add_argument('--t_alpha', type=float, default=1.0, help='t_alpha parameter for AdaFace loss')

    # Training Hyperparameters
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=40, help='Number of training epochs')
    parser.add_argument('--precision', type=str, default = '16-mixed', help='Use mixed precision training')

    # Data
    parser.add_argument('--val_datasets', nargs='+', default=None, help='List of validation dataset names')
    parser.add_argument('--min_num_images_per_class', type = int, default = 20, help = "The minimum number of images per class, classes less than this number are discarded.")
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')

    # System
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Path to a checkpoint to resume training from')

    args = parser.parse_args()
    main(args)
