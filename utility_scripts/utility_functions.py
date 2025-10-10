import numpy as np
import cv2
import os 
import torch
import torchvision


def get_bbox_from_landmarks(image, landmarks, padding = 1.25):
    """
        Calculate the bounding box from the landmarks and add padding.
        This function will always return a square bounding box. (width = height)
    """

    min_x, min_y = np.min(landmarks[:, 0]), np.min(landmarks[:, 1])
    max_x, max_y = np.max(landmarks[:, 0]), np.max(landmarks[:, 1])

    face_width = max_x - min_x
    face_height = max_y - min_y
    face_size = max(face_width, face_height) * padding
    center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2

    return int(center_x - face_size / 2), int(center_y - face_size / 2), int(face_size)


def crop_face(image, landmarks, padding = 1.25):
    x, y, size = get_bbox_from_landmarks(image, landmarks, padding)

    # make sure the crop doesn't go outside the limits of the image
    x = max(0, x)
    y = max(0, y)
    size = min(size, image.shape[0] - y, image.shape[1] - x)
    #crop the image
    cropped_image = image[y : y + size, x : x + size]

    #adjust the landmarks to match the cropping
    for i in range(len(landmarks)):
        landmarks[i][0] -= x
        landmarks[i][1] -= y

    return cropped_image, landmarks


def resize_image_and_landmarks(image, landmarks, size = 112):
    """
    This function resizes the image and adjusts the landmarks accordingly.
    """
    h, w = image.shape[:2]
    scale = size / h
    image = cv2.resize(image, (size, size), interpolation = cv2.INTER_CUBIC)

    for i in range(len(landmarks)):
        landmarks[i][0] *= scale
        landmarks[i][1] *= scale

    return image, landmarks


def process_face_image_and_landmarks(image, landmarks, size = 112, padding = 1.25):
    landmarks = np.array(landmarks)

    #Crop the image with padding
    image, landmarks = crop_face(image, landmarks, padding)
    
    #Resize the image
    image, landmarks = resize_image_and_landmarks(image, landmarks, size)

    return image, landmarks