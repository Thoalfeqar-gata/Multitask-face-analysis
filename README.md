# Project Overview

This is a PyTorch-based project for face analysis. It utilizes PyTorch Lightning for organizing the training process and supports various backbones, loss functions (heads), and datasets. The project is still under development, and here are the key features:

## Key Features

*   **Backbones:** Supports multiple backbone architectures for feature extraction, including:
    *   ResNet (`resnet18`, `resnet34`, `resnet50`, etc.)
    *   MobileNetV2 (`mobilenetv2`)
    *   Swin Transformer (`swin_t`, `swin_s`, `swin_b`)
    *   IR-Net (Improved ResNet) and IR-SE-Net (`ir_18`, `ir_se_50`, etc.)
*   **Heads (Loss Functions for Face Recognition):** Implements several popular loss functions for deep face recognition:
    *   AdaFace
    *   ArcFace
    *   CosFace
    *   SphereFace
*   **Datasets:** Includes data loaders for a wide range of public datasets for various face-related tasks:
    *   **Face Recognition:** VGGFace, MS1MV2, Glint360k, Casia-WebFace
    *   **Face Verification:** LFW, CPLFW, CALFW, CFP-FP, CFP-FF
    *   **Emotion Recognition:** RAF-DB, ExpW, AffectNet
    *   **Landmark Detection:** 300W, AFLW2000, COFW
    *   **Face Captioning:** FaceCaption1M
    *   **Attribute Recognition:** CelebA
    *   **Pose Estimation:** 300W-LP
*   **Training:** PyTorch Lightning for structured and reproducible training.
*   **Evaluation:** Currently, only face recognition evaluation is supported. More tasks will be added soon!

# Building and Running

## Dependencies

The project requires Python and the libraries listed in `requirements.txt`.

To install the dependencies, run:
```bash
pip install -r requirements.txt
```

## Training

The main training script for now is `pretrain.py`. You can train a model by running this script with the desired arguments.

**Example:**

To train an `ir_se_18` backbone with `adaface` head on the `CasiaWebFace_Dataset`, you can use the following command:

```bash
python train.py \
    --backbone_name ir_se_18 \
    --head_name adaface \
    --embedding_dim 512 \
    --margin 0.4 \
    --scale 64 \
    --optimizer adamw \
    --learning_rate 5e-4 \
    --weight_decay 0.05 \
    --scheduler cosine \
    --max_epochs 120 \
    --warmup_epochs 5 \
    --batch_size 512 \
    --dataset_name CasiaWebFace_Dataset \
    --val_datasets LFW_Dataset CALFW_Dataset CPLFW_Dataset \
    --num_workers 8 \
```

The `scripts/` directory also contains example shell scripts for training, like `train_irse18_casiawebface_adaface.sh`.

## Evaluation

The `eval.py` script can be used to evaluate a trained model on face verification datasets (LFW, CPLFW, CALFW, CFP-FP, CFP-FF).