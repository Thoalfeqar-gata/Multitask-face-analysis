import numpy as np
import cv2
import os 
import torch
import torchvision
import scipy
from utility_scripts import datasets
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


##################################

#   Pose Estimation Utilities

##################################

def mat_to_rotation_matrix(mat_file: str):
    """
    (Incomplete) Intended to extract a rotation matrix from a .mat file.

    Args:
        mat_file (str): Path to the .mat file.

    Returns:
        str: A placeholder string.
    """
    # TODO: Implement the logic to extract and return the actual rotation matrix.
    content = scipy.io.loadmat(mat_file)
    print(content)

    return "Cookie!"



##################################

#   Face recognition testing utilities

##################################


def evaluate_backbone(backbone):
    """
    Evaluates a face recognition backbone on standard benchmarks using 10-fold cross-validation.

    This function computes squared L2 distance scores on various test datasets
    and then calculates the average accuracy and other metrics following the
    standard face recognition evaluation protocol (e.g., LFW).

    Args:
        backbone (torch.nn.Module): The trained face recognition model (embedding extractor).

    Returns:
        dict: A dictionary where keys are dataset names (e.g., 'LFW') and values
              are tuples containing:
              (mean_accuracy, mean_precision, mean_recall, mean_f1_score,
               global_auc_score, global_fpr, global_tpr, global_roc_thresholds).
              The primary metric reported in papers is 'mean_accuracy'.
    """
    distances_results = get_face_recognition_distances_from_backbone(backbone)
    metrics_results = get_metrics_from_distances(distances_results)
    return metrics_results

# --- Feature Extraction and Distance Calculation ---

def get_face_recognition_distances_from_backbone(backbone: torch.nn.Module,
                                                 datasets_to_test = ['LFW', 'CPLFW', 'CALFW', 'CFP-FP', 'CFP-FF'],
                                                 use_tta = True,
                                                 batch_size = 512):
    """
    Computes squared L2 distance scores for image pairs from specified face recognition datasets.

    Extracts embeddings using the provided backbone, applies optional Test-Time Augmentation (TTA),
    and calculates the squared Euclidean distance between embeddings for each pair.

    Args:
        backbone (torch.nn.Module): The trained face recognition model.
        datasets_to_test (list, optional): List of dataset names.
            Defaults to ['LFW', 'CPLFW', 'CALFW', 'CFP-FP', 'CFP-FF'].
        use_tta (bool, optional): If True, averages embeddings from original and horizontally flipped images.
            Defaults to True.
        batch_size (int, optional): Batch size for processing images. Defaults to 512.


    Returns:
        dict: A dictionary where keys are dataset names and values are tuples
              (distances, labels), where 'distances' is a NumPy array of squared L2
              distances and 'labels' is a NumPy array of ground-truth labels (1=same, 0=different).
    """
    backbone.to('cuda')
    backbone.eval()

    image_transform = torchvision.transforms.v2.Compose([
        torchvision.transforms.v2.Resize((112, 112)),
        torchvision.transforms.v2.ToImage(),
        torchvision.transforms.v2.ToDtype(torch.float32, scale = True),
        torchvision.transforms.v2.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
    ])

    test_datasets_map = {
        'LFW': datasets.LFW_Dataset,
        'CPLFW': datasets.CPLFW_Dataset,
        'CALFW': datasets.CALFW_Dataset,
        'CFP-FP': datasets.CFPFP_Dataset,
        'CFP-FF': datasets.CFPFF_Dataset
    }
    test_datasets = []
    loaded_dataset_names = []
    for name in datasets_to_test:
        if name in test_datasets_map:
            test_datasets.append(test_datasets_map[name](image_transform=image_transform))
            loaded_dataset_names.append(name)
        else:
            print(f"Warning: Dataset '{name}' not recognized and will be skipped.")
            continue

    results = {}

    for i, dataset in enumerate(test_datasets):
        dataset_name = loaded_dataset_names[i]
        print(f"Processing dataset: {dataset_name}...")
        predicted_dist_list = []
        actual_list = []

        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=4,
                                             pin_memory=True)

        with torch.no_grad():
            for (image1_batch, image2_batch), label_batch in loader:
                image1_batch = image1_batch.to('cuda', non_blocking=True)
                image2_batch = image2_batch.to('cuda', non_blocking=True)

                if use_tta:
                    image1_flipped = torch.flip(image1_batch, dims=[3])
                    image2_flipped = torch.flip(image2_batch, dims=[3])

                    embeddings1_original = backbone(image1_batch)
                    embeddings2_original = backbone(image2_batch)
                    embeddings1_flipped = backbone(image1_flipped)
                    embeddings2_flipped = backbone(image2_flipped)

                    embeddings1_original = F.normalize(embeddings1_original, p=2, dim=1)
                    embeddings2_original = F.normalize(embeddings2_original, p=2, dim=1)
                    embeddings1_flipped = F.normalize(embeddings1_flipped, p=2, dim=1)
                    embeddings2_flipped = F.normalize(embeddings2_flipped, p=2, dim=1)

                    embeddings1 = (embeddings1_original + embeddings1_flipped) / 2.0
                    embeddings2 = (embeddings2_original + embeddings2_flipped) / 2.0

                    embeddings1 = embeddings1.cpu().numpy()
                    embeddings2 = embeddings2.cpu().numpy()

                else: # No TTA
                    embeddings1 = backbone(image1_batch)
                    embeddings2 = backbone(image2_batch)

                    embeddings1 = F.normalize(embeddings1, p=2, dim=1).cpu().numpy()
                    embeddings2 = F.normalize(embeddings2, p=2, dim=1).cpu().numpy()

                diff = embeddings1 - embeddings2
                dist = np.sum(np.square(diff), axis=1)

                predicted_dist_list.extend(dist)
                actual_list.extend(label_batch.cpu().numpy())

        results[dataset_name] = (np.array(predicted_dist_list), np.array(actual_list))
        print(f"Finished processing {dataset_name}.")

    return results

# --- Metric Calculation Utilities ---

def get_metrics_from_distances(results: dict):
    """
    Calculates face recognition metrics using 10-fold cross-validation on distance scores.

    Follows the standard LFW protocol: finds the best accuracy threshold on 9 folds,
    tests on the remaining fold, and averages the test accuracy over all 10 folds.
    Also calculates average precision, recall, F1, and global ROC/AUC.

    Args:
        results (dict): Dictionary from `get_face_recognition_distances_from_backbone`.
                        Keys are dataset names, values are (distances, labels) tuples.

    Returns:
        dict: A dictionary where keys are dataset names and values are tuples:
              (mean_accuracy, mean_precision, mean_recall, mean_f1_score,
               global_auc_score, global_fpr, global_tpr, global_roc_thresholds).
    """
    dataset_metrics = {}
    for key, (predicted_dist, actual_labels) in results.items():

        kf = KFold(n_splits=10, shuffle=False)
        fold_metrics = []

        for fold_idx, (train_indices, test_indices) in enumerate(kf.split(predicted_dist)):
            fold_best_threshold, _ = find_best_threshold_by_accuracy(
                predicted_dist[train_indices], actual_labels[train_indices]
            )

            accuracy, precision, recall, f1_score = compute_metrics_at_threshold(
                predicted_dist[test_indices], actual_labels[test_indices], fold_best_threshold
            )

            fold_metrics.append([accuracy, precision, recall, f1_score])

        mean_metrics = np.mean(fold_metrics, axis=0)
        mean_accuracy, mean_precision, mean_recall, mean_f1_score = mean_metrics

        fpr, tpr, roc_thresholds = roc_curve(actual_labels, -predicted_dist)
        auc_score = auc(fpr, tpr)

        dataset_metrics[key] = (mean_accuracy, mean_precision, mean_recall, mean_f1_score,
                                auc_score, fpr, tpr, roc_thresholds)

    return dataset_metrics


def compute_metrics_at_threshold(distances, labels, threshold, eta=1e-11):
    """
    Computes accuracy, precision, recall, and F1-score for a given distance threshold.

    Args:
        distances (np.ndarray): Array of squared L2 distance scores.
        labels (np.ndarray): Array of ground-truth labels (1 for same ID, 0 for different).
        threshold (float): Decision threshold. Distances < threshold are predicted as positive (same ID).
        eta (float, optional): Small epsilon to prevent division by zero. Defaults to 1e-11.

    Returns:
        tuple: (accuracy, precision, recall, f1_score).
    """
    distances = np.array(distances)
    labels = np.array(labels)

    predictions = (distances < threshold).astype(int)

    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    tn = np.sum((predictions == 0) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))

    total_samples = len(labels)
    accuracy = (tp + tn) / (total_samples + eta)
    precision = tp / (tp + fp + eta)
    recall = tp / (tp + fn + eta)
    f1_score = 2 * precision * recall / (precision + recall + eta)

    return accuracy, precision, recall, f1_score


def find_best_threshold_by_accuracy(distances, labels, num_thresholds=1000):
    """
    Finds the decision threshold that maximizes accuracy on the given distances and labels.

    Iterates through potential thresholds in the range [0, 4] (valid range for
    squared L2 distance between normalized vectors) and returns the one yielding
    the highest accuracy.

    Args:
        distances (np.ndarray): Array of squared L2 distance scores.
        labels (np.ndarray): Array of ground-truth labels (1=same, 0=different).
        num_thresholds (int, optional): Number of threshold values to test.
            Defaults to 1000.

    Returns:
        tuple: (best_threshold, best_accuracy). Returns (0.0, 0.0) if inputs are empty or invalid.
    """
    if len(distances) == 0 or len(labels) == 0:
        return 0.0, 0.0 # Handle empty input case

    thresholds = np.linspace(0.0, 4.0, num_thresholds)

    best_threshold = 0.0
    best_accuracy = 0.0

    for threshold in thresholds:
        accuracy, _, _, _ = compute_metrics_at_threshold(distances, labels, threshold)

        if accuracy >= best_accuracy: # Use >= to prefer higher thresholds in case of tie
            best_accuracy = accuracy
            best_threshold = threshold

    return best_threshold, best_accuracy