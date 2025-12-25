import numpy as np
import torch
from torchvision.transforms import v2
import torch.nn.functional as F
import datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold

##################################

#   Face recognition testing utilities

##################################


def evaluate_backbone(backbone, datasets_to_test = ['LFW', 'CPLFW', 'CALFW', 'CFP-FP', 'CFP-FF'], batch_size = 128):
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
                                                 datasets_to_test = ['LFW', 'CPLFW', 'CALFW', 'CFP-FP', 'CFP-FF'],
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
        v2.ToPILImage(),
        v2.ToTensor(),
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

    thresholds = np.linspace(0.0, 4.0, num_thresholds)

    best_threshold = 0.0
    best_accuracy = 0.0

    for threshold in thresholds:
        accuracy, _, _, _ = compute_metrics_at_threshold(distances, labels, threshold)

        if accuracy >= best_accuracy: # Use >= to prefer higher thresholds in case of tie
            best_accuracy = accuracy
            best_threshold = threshold

    return best_threshold, best_accuracy