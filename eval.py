import numpy as np
import torch
from torchvision.transforms import v2
import torch.nn.functional as F
import datasets
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.model_selection import KFold
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from utility_scripts.pose_estimation_utilities import compute_rotation_matrix_from_ortho6d, matrix_to_euler_angles


##################################

#   MultiTask testing and validation utilities

#################################

def evaluate_emotion(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device = 'cuda'):
    """
    Evaluates Emotion Recognition (Multi-class Classification).
    """
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating Emotion", leave=False):
            images, labels = batch
            images = images.to(device)
            targets = labels['emotion'].to(device)
            
            # Forward pass
            _, emotion_out, _, _, _, _, _ = model(images)
            
            preds = torch.argmax(emotion_out, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)
    
    return accuracy, f1, precision, recall, cm






def evaluate_age(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device='cuda', min_age=0, max_age=101):
    """
    Evaluates Age Estimation (DLDL approach).
    Returns: Mean Absolute Error (MAE) in years.
    """
    model.to(device)
    model.eval()
    true_ages = []
    predicted_ages_list = []
    
    # Pre-create the age vector (0, 1, ..., 101) once to save time
    age_values = torch.arange(min_age, max_age + 1, dtype=torch.float32, device=device).view(1, -1)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating Age", leave=False):
            images, labels = batch
            images = images.to(device)

            targets = labels['age'].to(device).float()
            
            _, _, age_logits, _, _, _, _ = model(images)
            
            probs = F.softmax(age_logits, dim=1)
            predicted_ages = torch.sum(probs * age_values, dim=1) 
            
            true_ages.extend(targets.cpu().numpy())
            predicted_ages_list.extend(predicted_ages.cpu().numpy())
    
    
    true_ages = np.array(true_ages)
    predicted_ages_list = np.array(predicted_ages_list)

    mae = np.mean(np.abs(true_ages - predicted_ages_list))

    return mae, true_ages, predicted_ages_list



def evaluate_gender(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device = 'cuda'):
    """
    Evaluates Gender Recognition (Binary Classification).
    """
    model.to(device)
    model.eval()
    true_labels = []
    predicted_labels = []
    probs_list = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating Gender", leave=False):
            images, labels = batch
            images = images.to(device)
            targets = labels['gender'].to(device)
            
            # Forward pass - Extract only gender output
            _, _, _, gender_out, _, _, _ = model(images)
            
            # Accuracy (Sigmoid > 0.5)
            probs = torch.sigmoid(gender_out)
            preds = (probs > 0.5).long()

            true_labels.extend(targets.cpu().numpy())
            predicted_labels.extend(preds.cpu().numpy())
            probs_list.extend(probs.cpu().numpy())

    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    probs_list = np.array(probs_list)

    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    cm = confusion_matrix(true_labels, predicted_labels)
    fpr, tpr, thresholds = roc_curve(true_labels, probs_list)
    auc_score = roc_auc_score(true_labels, probs_list)
    return accuracy, f1, precision, recall, cm, fpr, tpr, auc_score, thresholds


def evaluate_race(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device = 'cuda'):
    """
    Evaluates Race Recognition (Multi-class Classification).
    """
    model.to(device)
    model.eval()
    predicted_labels = []
    actual_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating Race", leave=False):
            images, labels = batch
            images = images.to(device)
            targets = labels['race'].to(device)
            
            _, _, _, _, race_out, _, _ = model(images)
            
            preds = torch.argmax(race_out, dim=1)

            predicted_labels.extend(preds.cpu().numpy())
            actual_labels.extend(targets.cpu().numpy())

    predicted_labels = np.array(predicted_labels)
    actual_labels = np.array(actual_labels)

    accuracy = accuracy_score(actual_labels, predicted_labels)
    f1 = f1_score(actual_labels, predicted_labels, average='weighted')
    precision = precision_score(actual_labels, predicted_labels, average='weighted')
    recall = recall_score(actual_labels, predicted_labels, average='weighted')
    cm = confusion_matrix(actual_labels, predicted_labels)

    return accuracy, f1, precision, recall, cm


def evaluate_attributes(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device='cuda'):
    """
    Evaluates Attribute Recognition (Multi-label Classification).
    """
    model.to(device)
    model.eval()
    
    predicted_labels = []
    actual_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating Attributes", leave=False):
            images, labels = batch
            images = images.to(device)
            
            # targets is the Tensor (B, 40)
            targets = labels['attributes'].to(device)

            _, _, _, _, _, attribute_out, _ = model(images)
            
            predictions = torch.sigmoid(attribute_out)
            preds = (predictions > 0.5).long()

            actual_labels.extend(targets.cpu().numpy())
            predicted_labels.extend(preds.cpu().numpy())

    actual_labels = np.array(actual_labels)
    predicted_labels = np.array(predicted_labels)

    
    # Element-wise Accuracy (Lenient: % of correct individual bits)
    element_wise_accuracy = (actual_labels == predicted_labels).mean()

    f1 = f1_score(actual_labels, predicted_labels, average='micro')
    precision = precision_score(actual_labels, predicted_labels, average='micro')
    recall = recall_score(actual_labels, predicted_labels, average='micro')
    
    return element_wise_accuracy, f1, precision, recall


def evaluate_head_pose(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device='cuda'):
    """
    Evaluates Head Pose Estimation (Regression).
    Returns Mean Absolute Error (MAE) for Roll, Pitch, and Yaw in Degrees.
    """
    model.to(device)
    model.eval()
    
    # Accumulators for errors
    roll_errors = []
    pitch_errors = []
    yaw_errors = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating Head Pose", leave=False):
            images, labels = batch
            images = images.to(device)
            
            # Ground Truth (Degrees)
            pose_labels = labels['pose'].to(device)
            
            _, _, _, _, _, _, pose_output = model(images)
            
            # 1. Convert 6D Output -> Rotation Matrix
            pred_matrices = compute_rotation_matrix_from_ortho6d(pose_output)
            
            # 2. Convert Matrix -> Euler Angles (Degrees)
            pred_degrees = matrix_to_euler_angles(pred_matrices, to_degrees=True)
            
            # 3. Calculate Absolute Error (L1)
            # Error = |Predicted - GT|
            errors = torch.abs(pred_degrees - pose_labels)
            
            # Store errors (CPU)
            roll_errors.extend(errors[:, 0].cpu().numpy())
            pitch_errors.extend(errors[:, 1].cpu().numpy())
            yaw_errors.extend(errors[:, 2].cpu().numpy())

    # Calculate Mean Errors
    mae_roll = np.mean(roll_errors)
    mae_pitch = np.mean(pitch_errors)
    mae_yaw = np.mean(yaw_errors)
    
    # Average MAE across all axes
    mae_mean = np.mean([mae_roll, mae_pitch, mae_yaw])
    
    return mae_mean, mae_roll, mae_pitch, mae_yaw




##################################

#   Face recognition testing utilities

##################################


def evaluate_face_recognition(backbone, datasets_to_test = ['LFW', 'CPLFW', 'CALFW', 'CFP-FP', 'CFP-FF', 'AgeDB30', 'VGG2FP'], batch_size = 128):
    """
    Evaluates a face recognition backbone on standard benchmarks using 10-fold cross-validation.

    This function computes squared L2 distance scores on various test datasets
    and then calculates the average accuracy and other metrics following the
    standard face recognition evaluation protocol (e.g., LFW).

    Args:
        backbone : The trained face recognition model (embedding extractor).

    Returns:
        dict: A dictionary where keys are dataset names (e.g., 'LFW') and values
              are tuples containing:
              (mean_accuracy, mean_precision, mean_recall, mean_f1_score,
               global_auc_score, global_fpr, global_tpr, global_roc_thresholds).
              The primary metric reported in papers is 'mean_accuracy'.
    """
    distances_results = get_face_recognition_distances_from_backbone(backbone, datasets_to_test, batch_size = batch_size)
    metrics_results = get_metrics_from_distances(distances_results)
    return metrics_results

# --- Feature Extraction and Distance Calculation ---

def get_face_recognition_distances_from_backbone(backbone: torch.nn.Module,
                                                 datasets_to_test = ['LFW', 'CPLFW', 'CALFW', 'CFP-FP', 'CFP-FF', 'AgeDB30', 'VGG2FP'],
                                                 use_tta = True,
                                                 batch_size = 512,
                                                 use_gpu = True,
                                                 test_in_bgr = False):
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
        preprocess (bool): Whether to apply preprocessing on the images. Defaults to True.
        use_gpu (bool): Whether to use GPU for processing. Defaults to True.


    Returns:
        dict: A dictionary where keys are dataset names and values are tuples
              (distances, labels), where 'distances' is a NumPy array of squared L2
              distances and 'labels' is a NumPy array of ground-truth labels (1=same, 0=different).
    """
    if use_gpu:
        backbone.to('cuda')
    backbone.eval()

    image_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale = True),
        v2.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]) # To [-1, 1]
    ])

    test_datasets_map = {
        'LFW': datasets.LFW,
        'CPLFW': datasets.CPLFW,
        'CALFW': datasets.CALFW,
        'CFP-FP': datasets.CFPFP,
        'CFP-FF': datasets.CFPFF,
        'AgeDB30' : datasets.AgeDB30,
        'VGG2FP' : datasets.VGG2FP
    }
    test_datasets = []
    loaded_dataset_names = []
    for name in datasets_to_test:
        if name in test_datasets_map:
            test_datasets.append(test_datasets_map[name](transform=image_transform))
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
                
                if test_in_bgr:
                    #rgb to bgr
                    image1_batch = torch.flip(image1_batch, dims = [1])
                    image2_batch = torch.flip(image2_batch, dims = [1])

                if use_gpu:
                    image1_batch = image1_batch.to('cuda', non_blocking=True)
                    image2_batch = image2_batch.to('cuda', non_blocking=True)

                if use_tta:
                    image1_flipped = torch.flip(image1_batch, dims=[3])
                    image2_flipped = torch.flip(image2_batch, dims=[3])

                    embeddings1_original, norm1_original = backbone(image1_batch)
                    embeddings2_original, norm2_original = backbone(image2_batch)
                    embeddings1_flipped, norm1_flipped = backbone(image1_flipped)
                    embeddings2_flipped, norm2_flipped = backbone(image2_flipped)

                    #unnormalize
                    embeddings1_original = embeddings1_original * norm1_original
                    embeddings2_original = embeddings2_original * norm2_original
                    embeddings1_flipped = embeddings1_flipped * norm1_flipped
                    embeddings2_flipped = embeddings2_flipped * norm2_flipped

                    # sum
                    embeddings1 = (embeddings1_original + embeddings1_flipped)
                    embeddings2 = (embeddings2_original + embeddings2_flipped)

                    #normalize
                    embeddings1 = F.normalize(embeddings1, p=2, dim=1)
                    embeddings2 = F.normalize(embeddings2, p=2, dim=1)

                    embeddings1 = embeddings1.cpu().numpy()
                    embeddings2 = embeddings2.cpu().numpy()

                else: # No TTA
                    embeddings1, norm1 = backbone(image1_batch)
                    embeddings2, norm2 = backbone(image2_batch)

                    embeddings1 = embeddings1.cpu().numpy()
                    embeddings2 = embeddings2.cpu().numpy()


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

    thresholds = np.linspace(0.0, 4.0, num_thresholds) # the maximum squared distance between two normalized vectors is 4, so the range is from 0 to 4

    best_threshold = 0.0
    best_accuracy = 0.0

    for threshold in thresholds:
        accuracy, _, _, _ = compute_metrics_at_threshold(distances, labels, threshold)

        if accuracy >= best_accuracy: # Use >= to prefer higher thresholds in case of tie
            best_accuracy = accuracy
            best_threshold = threshold

    return best_threshold, best_accuracy