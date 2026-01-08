import cv2
import random
import numpy as np
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms.v2 as v2
from torchvision.transforms import functional as F
from PIL import Image
from torchvision import transforms


def get_task_augmentation_transforms():

    face_recognition_transform = v2.Compose([ 
        v2.ToPILImage(),
        Augmenter(crop_augmentation_prob=0.2, low_res_augmentation_prob=0.2, photometric_augmentation_prob=0.2),
        v2.RandomHorizontalFlip(p = 0.5),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale = True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    age_gender_race_recognition = v2.Compose([
        v2.ToPILImage(),
        v2.RandomHorizontalFlip(p = 0.5),
        v2.RandomAffine(degrees = 15, translate = (0.1, 0.1), scale = (0.9, 1.1)),
        v2.RandomApply([v2.GaussianBlur(kernel_size = 3, sigma = (0.1, 2))], p = 0.1),
        RandomGamma(min_gamma=0.8, max_gamma=1.2, p=0.1),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale = True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    head_pose_estimation = v2.Compose([
        v2.ToPILImage(),
        v2.RandomResizedCrop(size = (112, 112), scale = (0.8, 1.0), ratio = (1, 1)),
        v2.RandomGrayscale(p = 0.1),
        v2.RandomApply([v2.GaussianBlur(kernel_size = 3, sigma = (0.1, 2))], p = 0.1),
        v2.RandomApply([v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)], p = 0.1),
        RandomGamma(min_gamma=0.5, max_gamma=1.5, p=0.1),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale = True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    emotion_recognition_transform = v2.Compose([ 
        v2.ToPILImage(),
        v2.RandomHorizontalFlip(p = 0.5),
        v2.RandomAffine(degrees = 15, translate = (0.1, 0.1), scale = (0.9, 1.1)),
        v2.RandomGrayscale(p = 0.1),
        v2.RandomApply([v2.GaussianBlur(kernel_size = 3, sigma = (0.1, 2))], p = 0.1),
        v2.RandomApply([v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)], p = 0.1),
        RandomGamma(min_gamma=0.5, max_gamma=1.5, p=0.1),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale = True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    attribute_recognition_transform = v2.Compose([
        v2.ToPILImage(),
        v2.RandomHorizontalFlip(p = 0.5),
        v2.RandomAffine(degrees = 10, padding_mode = 'reflection'),
        v2.RandomGrayscale(p = 0.1),
        v2.RandomApply([v2.GaussianBlur(kernel_size = 3, sigma = (0.1, 2))], p = 0.1),
        v2.RandomApply([v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)], p = 0.1),
        RandomGamma(min_gamma=0.5, max_gamma=1.5, p=0.1),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale = True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    return {
        'face_recognition': face_recognition_transform,
        'emotion_recognition': emotion_recognition_transform,
        'age_gender_race_recognition': age_gender_race_recognition,
        'attribute_recognition': attribute_recognition_transform,
        'head_pose_estimation': head_pose_estimation
    }

class RandomGamma(object):
    def __init__(self, min_gamma=0.5, max_gamma=1.5, p=0.5):
        """
        Args:
            min_gamma (float): Lower bound for gamma (below 1.0 brightens).
            max_gamma (float): Upper bound for gamma (above 1.0 darkens).
            p (float): Probability of applying the transform.
        """
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            # Sample a random gamma value from the range
            gamma = random.uniform(self.min_gamma, self.max_gamma)
            
            # Apply gamma correction
            return TF.adjust_gamma(img, gamma, gain=1)
        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(p={self.p}, gamma_range=({self.min_gamma}, {self.max_gamma}))'


class Augmenter():

    def __init__(self, crop_augmentation_prob = 0.2, photometric_augmentation_prob = 0.2, low_res_augmentation_prob = 0.2):
        self.crop_augmentation_prob = crop_augmentation_prob
        self.photometric_augmentation_prob = photometric_augmentation_prob
        self.low_res_augmentation_prob = low_res_augmentation_prob

        self.random_resized_crop = transforms.RandomResizedCrop(size=(112, 112),
                                                                scale=(0.5, 1.0),
                                                                ratio=(0.75, 1.3333333333333333))
        self.photometric = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0)

    def __call__(self, sample):

        # crop with zero padding augmentation
        if np.random.random() < self.crop_augmentation_prob:
            # RandomResizedCrop augmentation
            sample, crop_ratio = self.crop_augment(sample)

        # low resolution augmentation
        if np.random.random() < self.low_res_augmentation_prob:
            # low res augmentation
            img_np, resize_ratio = self.low_res_augmentation(np.array(sample))
            sample = Image.fromarray(img_np.astype(np.uint8))

        # photometric augmentation
        if np.random.random() < self.photometric_augmentation_prob:
            sample = self.photometric_augmentation(sample)

        return sample

    def crop_augment(self, sample):
        new = np.zeros_like(np.array(sample))
        if hasattr(F, '_get_image_size'):
            orig_W, orig_H = F._get_image_size(sample)
        else:
            # torchvision 0.11.0 and above
            orig_W, orig_H = F.get_image_size(sample)
        i, j, h, w = self.random_resized_crop.get_params(sample,
                                                         self.random_resized_crop.scale,
                                                         self.random_resized_crop.ratio)
        cropped = F.crop(sample, i, j, h, w)
        new[i:i+h,j:j+w, :] = np.array(cropped)
        sample = Image.fromarray(new.astype(np.uint8))
        crop_ratio = min(h, w) / max(orig_H, orig_W)
        return sample, crop_ratio

    def low_res_augmentation(self, img):
        # resize the image to a small size and enlarge it back
        img_shape = img.shape
        side_ratio = np.random.uniform(0.2, 1.0)
        small_side = int(side_ratio * img_shape[0])
        interpolation = np.random.choice(
            [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
        small_img = cv2.resize(img, (small_side, small_side), interpolation=interpolation)
        interpolation = np.random.choice(
            [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
        aug_img = cv2.resize(small_img, (img_shape[1], img_shape[0]), interpolation=interpolation)

        return aug_img, side_ratio

    def photometric_augmentation(self, sample):
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
            self.photometric.get_params(self.photometric.brightness, self.photometric.contrast,
                                        self.photometric.saturation, self.photometric.hue)
        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                sample = F.adjust_brightness(sample, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                sample = F.adjust_contrast(sample, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                sample = F.adjust_saturation(sample, saturation_factor)

        return sample