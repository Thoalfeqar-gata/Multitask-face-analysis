import numpy as np
import cv2
import os 
import torch
import torchvision
import scipy
from utility_scripts import datasets
from sklearn.metrics import roc_curve, auc




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
        landmarks (np.ndarray): A NumPy array of shape (N, 2) पुलिस the landmark coordinates.
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
        tuple: A tuple पुलिस the processed image tensor and processed landmarks tensor.
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
    A convenience wrapper to evaluate a face recognition backbone on standard benchmarks.

    This function first computes the similarity scores on various test datasets and then
    calculates the performance metrics based on those similarities.

    Args:
        backbone (torch.nn.Module): The face recognition model to evaluate.

    Returns:
        dict: A dictionary containing the performance metrics for each dataset.
    """
    similarities = get_face_recognition_similarities_from_backbone(backbone)
    return get_metrics_from_similarities(similarities)


def get_face_recognition_similarities_from_backbone(backbone: torch.nn.Module, datasets_to_test = ['LFW', 'CPLFW', 'CALFW', 'CFP-FP', 'CFP-FF']):
    """
    Computes cosine similarity scores for image pairs from specified face recognition datasets.

    Args:
        backbone (torch.nn.Module): The trained face recognition model.
        datasets_to_test (list, optional): A list of dataset names to test.
            Defaults to ['LFW', 'CPLFW', 'CALFW', 'CFP-FP', 'CFP-FF'].

    Returns:
        dict: A dictionary where keys are dataset names and values are tuples
              containing an array of similarity scores and an array of ground-truth labels.
    """
    backbone.to('cuda')
    backbone.eval()
    
    image_transform = torchvision.transforms.v2.Compose([
        torchvision.transforms.v2.Resize((112, 112)),
        torchvision.transforms.v2.ToImage(),
        torchvision.transforms.v2.ToDtype(torch.float32, scale = True),
        torchvision.transforms.v2.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) #using imagenet defaults
    ])

    # Load the specified test datasets
    test_datasets = []
    for dataset in datasets_to_test:
        if dataset == 'LFW':
            test_datasets.append(
                datasets.LFW_Dataset(image_transform = image_transform),
            )
        elif dataset == 'CPLFW':
            test_datasets.append(
            datasets.CPLFW_Dataset(image_transform = image_transform),
        )
        elif dataset == 'CALFW':
            test_datasets.append(
                datasets.CALFW_Dataset(image_transform = image_transform),
            )
        elif dataset == 'CFP-FP':
            test_datasets.append(
                datasets.CFPFP_Dataset(image_transform = image_transform),
            )
        elif dataset == 'CFP-FF':
            test_datasets.append(
                datasets.CFPFF_Dataset(image_transform = image_transform)
            )
        else:
            continue
    
    results = dict() 

    # Iterate through each loaded dataset
    for i, dataset in enumerate(test_datasets):
        predicted_sim = []
        actual = []

        # Create a DataLoader for the current dataset
        loader = torch.utils.data.DataLoader(dataset, batch_size = 512, shuffle = False)
        
        # Process each batch of image pairs
        for (image1, image2), label in loader:

            image1 = image1.to('cuda')
            image2 = image2.to('cuda')
            
            # Generate embeddings for both images in the pair
            embeddings1 = backbone(image1).cpu().detach().numpy()
            embeddings2 = backbone(image2).cpu().detach().numpy()

            # Calculate cosine similarity for each pair in the batch
            for j in range(len(embeddings1)):
                embedding1 = embeddings1[j]
                embedding2 = embeddings2[j]
                
                # L2-normalize the embeddings before calculating cosine similarity
                # (dot product of two unit vectors is their cosine similarity)
                norm1 = np.linalg.norm(embedding1)
                norm2 = np.linalg.norm(embedding2)
                embedding1 /= norm1
                embedding2 /= norm2
                cosine_sim = np.dot(embedding1, embedding2) #both embeddings1 and 2 are l2 normalized
                predicted_sim.append(cosine_sim)
                actual.append(label[j])
        
        # Store the similarity scores and labels for the current dataset
        results[datasets_to_test[i]] = (np.array(predicted_sim), np.array(actual))
    
    return results


def get_metrics_from_similarities(results : dict):
    """
    Calculates face recognition metrics using 10-fold cross-validation on similarity scores.

    For each dataset in the results, this function splits the data into 10 folds,
    finds the best decision threshold on each fold, and computes metrics. The final
    metrics for the dataset are the average over all 10 folds.

    Args:
        results (dict): A dictionary from `get_face_recognition_similarities_from_backbone`.
                        Keys are dataset names, values are (similarities, labels) tuples.

    Returns:
        dict: A dictionary where keys are dataset names and values are tuples
              containing (accuracy, precision, recall, f1_score).
    """
    dataset_results = dict()
    for key in results.keys():
        predicted_sim, actual = results[key]
        
        # Prepare for 10-fold cross-validation
        dataset_length = len(predicted_sim)
        fold_length = dataset_length // 10

        accumulated_fold_metrics = np.zeros(shape = (4,), dtype = np.float64) #accumulate the fold metrics here: accuracy, precision, recall, f1_score
        # Perform 10-fold cross-validation
        for fold in range(10):
            start_index = fold * fold_length
            end_index = (fold + 1) * fold_length
            if fold == 9:
                end_index = dataset_length
            fold_best_threshold, _ = find_best_threshold(predicted_sim[start_index:end_index], actual[start_index:end_index])
            # Get the accuracy, precision, recall, and f1_score using the best threshold for this fold
            accuracy, precision, recall, f1_score = compute_metrics_at_threshold(predicted_sim[start_index:end_index], actual[start_index:end_index], fold_best_threshold)
            
            accumulated_fold_metrics += np.array([accuracy, precision, recall, f1_score])
        
        # Average the metrics over the 10 folds
        accumulated_fold_metrics /= 10
        accuracy, precision, recall, f1_score= accumulated_fold_metrics


        #Obtain ROC and AUC metrics
        fpr, tpr, thresholds = roc_curve(actual[start_index:end_index], predicted_sim[start_index:end_index])
        auc_score = auc(fpr, tpr)

        dataset_results[key] = (accuracy, precision, recall, f1_score, auc_score, fpr, tpr, thresholds)
    
    return dataset_results


def compute_metrics_at_threshold(sims, labels, threshold, eta=1e-11):
    """
    Computes accuracy, precision, recall, and F1-score for a given similarity threshold.

    Args:
        sims (np.ndarray): An array of cosine similarity scores.
        labels (np.ndarray): An array of ground-truth labels (1 for same ID, 0 for different).
        threshold (float): The decision threshold. Similarities >= threshold are classified as positive.
        eta (float, optional): A small epsilon to avoid division by zero. Defaults to 1e-11.

    Returns:
        tuple: A tuple containing (accuracy, precision, recall, f1_score).
    """
    sims = np.array(sims)
    labels = np.array(labels)

    # Predict positive (same ID) if similarity >= threshold
    predictions = (sims >= threshold).astype(int)

    # Calculate True Positives, False Positives, True Negatives, and False Negatives
    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    tn = np.sum((predictions == 0) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))

    # Compute the metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn + eta)
    precision = tp / (tp + fp + eta)
    recall = tp / (tp + fn + eta)
    f1_score = 2 * precision * recall / (precision + recall + eta)

    return accuracy, precision, recall, f1_score


def find_best_threshold(similarities, labels, num_thresholds = 20001):
    """
    Finds the best decision threshold that maximizes the F1-score.

    This function iterates through a range of possible thresholds, calculates the
    F1-score for each, and returns the threshold that yields the highest F1-score.

    Args:
        similarities (np.ndarray): An array of cosine similarity scores.
        labels (np.ndarray): An array of ground-truth labels.
        num_thresholds (int, optional): The number of threshold values to test. Defaults to 20001.

    Returns:
        tuple: A tuple containing (best_threshold, best_f1_score).
    """
    # Generate a range of potential thresholds to test
    thresholds = np.linspace(similarities.min(), similarities.max(), num_thresholds)

    best_threshold = None
    best_f1_score = 0
    # Iterate over all thresholds to find the one that maximizes the F1 score
    for threshold in thresholds:
        accuracy, precision, recall, f1_score = compute_metrics_at_threshold(similarities, labels, threshold)
        if f1_score > best_f1_score:
            best_f1_score = f1_score
            best_threshold = threshold
    
    return best_threshold, best_f1_score