import numpy as np
import cv2
import os 
import torch
import torchvision
import scipy
import datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
import torch.nn.functional as F


##################################

#   Landmark detection utilities

##################################


def get_bbox_from_landmarks(image, landmarks, padding = 1.25):
    """
    Calculate a square bounding box from the given landmarks with padding.

    This function determines the minimum and maximum coordinates of the landmarks
    to find the extent of the face, then creates a square bounding box centered
    around the face with a specified padding factor.

    Args:
        image (np.ndarray): The image containing the face (used for context, not direct processing).
        landmarks (np.ndarray): A NumPy array of shape (N, 2) that stores the landmark coordinates.
        padding (float, optional): The factor to expand the bounding box. Defaults to 1.25.

    Returns:
        tuple: A tuple (x, y, size) representing the top-left corner and the size of the square bounding box.
    """
    # Find the min and max coordinates of the landmarks to define the initial bounding box
    min_x, min_y = np.min(landmarks[:, 0]), np.min(landmarks[:, 1])
    max_x, max_y = np.max(landmarks[:, 0]), np.max(landmarks[:, 1])

    # Calculate the width and height of the face
    face_width = max_x - min_x
    face_height = max_y - min_y

    # Determine the size of the square bounding box by taking the larger dimension and applying padding
    face_size = max(face_width, face_height) * padding
    # Find the center of the face
    center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2

    # Calculate the top-left corner of the square bounding box
    return int(center_x - face_size / 2), int(center_y - face_size / 2), int(face_size)


def crop_face(image, landmarks, padding = 1.25):
    """
    Crops the face from an image based on landmarks and adjusts the landmarks accordingly.

    Args:
        image (np.ndarray): The input image.
        landmarks (np.ndarray): The facial landmarks.
        padding (float, optional): Padding for the bounding box. Defaults to 1.25.

    Returns:
        tuple: A tuple containing the cropped image (np.ndarray) and the adjusted landmarks (np.ndarray).
    """
    # Get the square bounding box for the face
    x, y, size = get_bbox_from_landmarks(image, landmarks, padding)

    # Ensure the crop coordinates do not go outside the image boundaries
    x = max(0, x)
    y = max(0, y)
    size = min(size, image.shape[0] - y, image.shape[1] - x)
    
    # Crop the image using the calculated bounding box
    cropped_image = image[y : y + size, x : x + size]

    # Adjust the landmark coordinates to be relative to the new cropped image
    for i in range(len(landmarks)):
        landmarks[i][0] -= x
        landmarks[i][1] -= y

    return cropped_image, landmarks


def resize_image_and_landmarks(image, landmarks, size = 112):
    """
    Resizes an image to a target size and scales the landmarks accordingly.

    Args:
        image (np.ndarray): The input image.
        landmarks (np.ndarray): The facial landmarks.
        size (int, optional): The target size for the height and width of the image. Defaults to 112.

    Returns:
        tuple: A tuple containing the resized image (np.ndarray) and the scaled landmarks (np.ndarray).
    """
    # Get the original height and width
    h, w = image.shape[:2]
    # Calculate the scaling factor
    scale = size / h
    # Resize the image using cubic interpolation for better quality
    image = cv2.resize(image, (size, size), interpolation = cv2.INTER_CUBIC)

    # Scale the landmarks to match the new image size
    for i in range(len(landmarks)):
        landmarks[i][0] *= scale
        landmarks[i][1] *= scale

    return image, landmarks


def process_face_image_and_landmarks(image: torch.Tensor, landmarks: list, size = 112, padding = 1.25):
    """
    A processing pipeline to crop and resize a face image and its landmarks.

    Args:
        image (torch.Tensor): The input image as a PyTorch tensor.
        landmarks (list): A list of landmark coordinates.
        size (int, optional): The final desired size of the image. Defaults to 112.
        padding (float, optional): The padding to use when cropping the face. Defaults to 1.25.

    Returns:
        tuple: A tuple that contains the processed image tensor and processed landmarks tensor.
    """
    # Convert inputs to NumPy arrays for processing with OpenCV and NumPy
    landmarks = np.array(landmarks, dtype = np.float32)
    image = image.permute(1, 2, 0).numpy()
    
    # Crop the image with padding around the landmarks
    image, landmarks = crop_face(image, landmarks, padding)
    
    # Resize the image and landmarks to the target size
    image, landmarks = resize_image_and_landmarks(image, landmarks, size)

    # Convert the processed NumPy arrays back to PyTorch tensors
    return torch.from_numpy(image).permute(2, 0, 1), torch.from_numpy(landmarks)


def get_2d_landmarks_from_aflw2000(mat_path):
    """
    Extracts the 2D landmarks from an AFLW2000 .mat file.

    According to the dataset's provided scripts, the X and Y coordinates
    in the 'pt3d_68' variable are the final 2D landmarks. This function
    simply loads the .mat file and extracts them.

    Args:
        mat_path (str): The file path to the .mat annotation file.

    Returns:
        np.ndarray: A NumPy array of shape (68, 2) containing the
            final 2D landmark coordinates.
    """
    mat_data = scipy.io.loadmat(mat_path)
    
    pt3d_68 = mat_data['pt3d_68']
    
    landmarks_2d = pt3d_68[:2, :]
    
    return landmarks_2d.T #(2, 68) => (68, 2)