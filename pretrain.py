import argparse
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.optim import lr_scheduler
from lightning.pytorch import loggers 
from augmenter import Augmenter

import backbones.backbones as backbones
import datasets
import multitask.face_recognition_heads as face_recognition_heads
from multitask.subnets import FaceRecognitionEmbeddingSubnet
import eval, os

torch.set_float32_matmul_precision('medium')



class FaceRecognitionModel(pl.LightningModule):
    """
    PyTorch Lightning module for training a face recognition model.
    """
    def __init__(self, **kwargs):
        """
        Args:
            kwargs: contains all the hyperparameters.
        """
        super().__init__()
        self.backbone_name = kwargs.get('backbone_name')
        self.pretrained = kwargs.get('pretrained')
        self.head_name = kwargs.get('head_name')
        self.embedding_dim = kwargs.get('embedding_dim')
        self.num_classes = kwargs.get('num_classes')
        self.optimizer_name = kwargs.get('optimizer')
        self.weight_decay = kwargs.get('weight_decay')
        self.scheduler_name = kwargs.get('scheduler')
        self.milestones = kwargs.get('scheduler_milestones')
        self.start_factor = kwargs.get('start_factor')
        self.min_lr = kwargs.get('min_lr')
        self.warmup_epochs = kwargs.get('warmup_epochs')
        self.learning_rate = kwargs.get('learning_rate')
        self.margin = kwargs.get('margin')
        self.scale = kwargs.get('scale')
        self.h = kwargs.get('h')
        self.t_alpha = kwargs.get('t_alpha')
        self.max_epochs = kwargs.get('max_epochs')
        self.val_datasets = kwargs.get('val_datasets')
        self.save_hyperparameters()


        # Instantiate the backbone
        self.backbone = backbones.get_backbone(
            backbone_name=self.backbone_name,
            embedding_dim=self.embedding_dim,
            imagenet_pretrained = bool(self.pretrained)
        )

        # Instantiate the face recognition subnet
        if self.backbone_name in ['swin_b', 'swin_v2_b', 'davit_b']:
            feature_embedding_dim = 1024 # The final feature vector dim
                                         # output by the backbone
        else:
            feature_embedding_dim = 768
        
        self.recognition_subnet = FaceRecognitionEmbeddingSubnet(
            feature_embedding_dim=feature_embedding_dim,
            embedding_dim=self.embedding_dim
        )   

        # Instantiate the margin head (AdaFace, ArcFace, or CosFace)
        self.margin_head = face_recognition_heads.build_head(
            head_type=self.head_name,
            embedding_size=self.embedding_dim,
            classnum=self.num_classes,
            m=self.margin,
            s=self.scale,
            h=self.h,
            t_alpha=self.t_alpha
        )

        self.global_counter = 0
        self.validation_outputs = []

    def forward(self, x):
        """
        Forward pass through the backbone.
        """
        multiscale_features = self.backbone(x)
        normalized_embedding, embedding_norm = self.recognition_subnet(multiscale_features)
        return normalized_embedding, embedding_norm

    def training_step(self, batch, batch_idx):
        """
        A single training step.
        """

        
        self.global_counter += 1
        
        images, labels = batch
        normalized_embedding, embedding_norm = self(images)
        
        output = self.margin_head(normalized_embedding, embedding_norm, labels)
        loss = F.cross_entropy(output, labels)
        accuracy = (output.argmax(dim=1) == labels).float().mean()

        self.log_dict({'train_loss': loss, 'train_acc': accuracy}, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss


    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        A single validation step.
        """
        (image1_batch, image2_batch), label_batch = batch
        
        # Get embeddings with tta
        embeddings1, norm1 = self(image1_batch)
        embeddings2, norm2 = self(image2_batch)
        flipped_embeddings1, flipped_norm1 = self(torch.flip(image1_batch, dims=[3]))
        flipped_embeddings2, flipped_norm2 = self(torch.flip(image2_batch, dims=[3]))

        #unnormalize
        embeddings1 = embeddings1 * norm1
        embeddings2 = embeddings2 * norm2
        flipped_embeddings1 = flipped_embeddings1 * flipped_norm1
        flipped_embeddings2 = flipped_embeddings2 * flipped_norm2

        # sum   
        embeddings1 = (embeddings1 + flipped_embeddings1)
        embeddings2 = (embeddings2 + flipped_embeddings2)

        # normalize
        embeddings1 = F.normalize(embeddings1, p=2, dim=1)
        embeddings2 = F.normalize(embeddings2, p=2, dim=1)



        # Calculate distances
        diff = embeddings1 - embeddings2
        dist = torch.sum(torch.square(diff), axis=1)

        self.validation_outputs.append({'distances': dist, 'labels': label_batch, 'dataloader_idx': dataloader_idx})


    def on_validation_epoch_end(self):
        """
        Called at the end of the validation epoch.
        """
        
        outputs = self.validation_outputs
        # Group outputs by dataloader_idx
        grouped_outputs = [[] for _ in range(len(self.trainer.val_dataloaders))]
        for output in outputs:
            grouped_outputs[output['dataloader_idx']].append(output)

        metrics_to_log = {}
        for i, val_output in enumerate(grouped_outputs):
            if not val_output:
                continue
            distances = torch.cat([x['distances'] for x in val_output]).cpu().float().numpy()
            labels = torch.cat([x['labels'] for x in val_output]).cpu().float().numpy()
            
            dataset_name = self.val_datasets[i]
            
            metrics_results = eval.get_metrics_from_distances({dataset_name: (distances, labels)})
            
            mean_accuracy, mean_precision, mean_recall, mean_f1_score, global_auc_score, _, _, _ = metrics_results[dataset_name]

            metrics_to_log[f'{dataset_name}_val_acc'] = mean_accuracy

        self.log_dict(metrics_to_log, on_epoch=True, prog_bar=True, logger=True)
        
        self.validation_outputs.clear()


    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.
        """
        if self.optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay = self.weight_decay)
        elif self.optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay = self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")


        if self.scheduler_name == 'cosine':
            linear = lr_scheduler.LinearLR(optimizer, start_factor=self.start_factor, end_factor=1.0, total_iters=self.warmup_epochs)
            cosine = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs - self.warmup_epochs, eta_min = self.min_lr)
            scheduler = lr_scheduler.SequentialLR(optimizer, [linear, cosine], milestones=[self.warmup_epochs])
        elif self.scheduler_name == 'multistep':
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones)
        else:
            raise ValueError(f"Unsupported scheduler: {self.scheduler_name}")


        return [optimizer], [scheduler]



class FaceRecognitionDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module for face recognition datasets.
    """
    def __init__(self, **kwargs):
        """
        Args:
            dataset_name (str): The dataset name (e.g. VGGFace2_Dataset). 
            val_datasets (list): A list of dataset names for validation.
            batch_size (int): Batch size for the dataloaders.
            num_workers (int): Number of worker processes for data loading.
        """
        super().__init__()
        self.dataset_name = kwargs.get('dataset_name')
        self.val_datasets = kwargs.get('val_datasets')
        self.min_num_image_per_class = kwargs.get('min_num_images_per_class')
        self.batch_size = kwargs.get('batch_size')
        self.num_workers = kwargs.get('num_workers')
        self.train_transform = kwargs.get('train_transform')
        self.val_transform = kwargs.get('val_transform')


    def setup(self, stage=None):
        """
        Set up the datasets.
        """
        # Training dataset
        dataset_class = getattr(datasets, self.dataset_name)
        self.train_dataset = dataset_class(image_transform=self.train_transform)
        self.train_dataset.discard_classes(self.min_num_image_per_class)
        self.num_classes = len(torch.unique(torch.tensor(self.train_dataset.labels)))
        
        # Validation datasets
        if self.val_datasets:
            self.instantiated_val_datasets = []
            for val_dataset_name in self.val_datasets:
                val_dataset_class = getattr(datasets, val_dataset_name)
                self.instantiated_val_datasets.append(val_dataset_class(image_transform=self.val_transform))


    def train_dataloader(self):
        """
        Returns the training dataloader.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                          num_workers=self.num_workers, shuffle=True, pin_memory=True, prefetch_factor=3)

    def val_dataloader(self):
        """
        Returns the validation dataloader.
        """
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
    """
    Main training function.
    """
    pl.seed_everything(args.seed)


    augmenter = Augmenter(crop_augmentation_prob=args.crop_prob, low_res_augmentation_prob=args.low_res_prop, photometric_augmentation_prob=args.photometric_prop)

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        augmenter,
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    data_module = FaceRecognitionDataModule(train_transform=train_transform, val_transform=val_transform, **vars(args))
    data_module.setup()
    
    model = FaceRecognitionModel(num_classes=data_module.num_classes, **vars(args))

    checkpoint_callback = ModelCheckpoint(
        dirpath=f'checkpoints/{args.backbone_name}_{args.head_name}_{args.dataset_name}',
        filename='{epoch}-{train_loss:.5f}',
        save_top_k=1,
        verbose=True,
        monitor='train_loss',
        mode='min'
    )

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        logger= loggers.TensorBoardLogger(save_dir = f'logs/{args.backbone_name}_{args.head_name}'),
        max_epochs = args.max_epochs,
        accelerator='auto',
        precision=args.precision,
        check_val_every_n_epoch=1,
    )

    trainer.fit(model, data_module, ckpt_path=args.resume_from_checkpoint)

    # Save after training
    output_path = os.path.join('data', 'models', f'{args.backbone_name}_{args.head_name}_{args.dataset_name}', 'model.pth')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    torch.save(
        model.backbone.state_dict(),
        output_path
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Lightning Face Recognition Training')
    
    # Model Hyperparameters
    parser.add_argument('--backbone_name', type=str, default='ResNet', help='Backbone model name')
    parser.add_argument('--pretrained', type = int, default = 0, help='Use pretrained weights 0 = False, 1 = True')
    parser.add_argument('--head_name', type=str, default='arcface', help='Head name (e.g., arcface, cosface, adaface)')
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
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=40, help='Number of training epochs')
    parser.add_argument('--precision', type=str, default = '16-mixed', help='Use mixed precision training')

    # Data
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset class (e.g., VGGFace_Dataset)')
    parser.add_argument('--val_datasets', nargs='+', default=None, help='List of validation dataset names')
    parser.add_argument('--min_num_images_per_class', type = int, default = 20, help = "The minimum number of images per class, classes less than this number are discarded.")
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--crop_prob', type = float, default = 0.2, help = 'The probability of applying random cropping')
    parser.add_argument('--low_res_prop', type = float, default = 0.2, help = 'The probability of applying low resolution augmentation')
    parser.add_argument('--photometric_prop', type = float, default = 0.2, help = 'The probability of applying photometric augmentation')

    # System
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Path to a checkpoint to resume training from')

    
    args = parser.parse_args()
    main(args)
