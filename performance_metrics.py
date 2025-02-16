import json
import torch
from model import BasicPitchModel  # Import your model class
import torch.nn.functional as F
from MaestroDataset import MaestroDataset  # Import your dataset class
from torch.utils.data import DataLoader, random_split


def calculate_metrics(predictions, ground_truths, threshold=0.2):
    """
    Calculate precision, recall, and F1 score for binary predictions.

    Args:
        predictions (torch.Tensor): Model predictions (probabilities between 0 and 1).
        ground_truths (torch.Tensor): Ground truth binary labels (0 or 1).
        threshold (float): Threshold to binarize predictions. Defaults to 0.5.

    Returns:
        dict: A dictionary containing precision, recall, and F1 score.
    """
    pred_binary = (predictions > threshold).int()
    true_binary = ground_truths.int()

    true_positive = (pred_binary & true_binary).sum().item()
    false_positive = (pred_binary & ~true_binary).sum().item()
    false_negative = (~pred_binary & true_binary).sum().item()

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1_score}


def load_and_evaluate(model_path, data_dir, batch_size=4, device="cuda"):
    """
    Load a model and evaluate it on the validation dataset.

    Args:
        model_path (str): Path to the saved model.
        data_dir (str): Path to the dataset directory.
        batch_size (int): Batch size for the DataLoader.
        device (str): Device to use ('cuda', 'mps', 'cpu').

    Returns:
        None
    """
    # Load model
    model = BasicPitchModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Prepare dataset
    dataset = MaestroDataset(data_dir)

    # Split dataset into training and validation subsets
    val_size = int(0.2 * len(dataset))  # 20% for validation
    train_size = len(dataset) - val_size
    _, val_dataset = random_split(dataset, [train_size, val_size])

    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize metrics storage
    all_metrics = {"onset": [], "note": [], "pitch": []}

    # Evaluation loop
    with torch.no_grad():
        for inputs, labels in val_dataloader:
            # Move data to device
            inputs = inputs.to(device)
            yo_true = labels['onset'].to(device)
            yn_true = labels['note'].to(device)
            yp_true = labels['pitch'].to(device)

            # Model predictions
            yo_pred, yp_pred, yn_pred = model(inputs)

            # Resize ground truths to match predictions
            yo_true_resized = F.interpolate(yo_true.unsqueeze(1), size=yo_pred.shape[2:], mode='bilinear', align_corners=False).squeeze(1)
            yn_true_resized = F.interpolate(yn_true.unsqueeze(1), size=yn_pred.shape[2:], mode='bilinear', align_corners=False).squeeze(1)
            yp_true_resized = F.interpolate(yp_true.unsqueeze(1), size=yp_pred.shape[2:], mode='bilinear', align_corners=False).squeeze(1)

            # Compute metrics for each posteriorgram
            metrics_onset = calculate_metrics(yo_pred, yo_true_resized)
            metrics_note = calculate_metrics(yn_pred, yn_true_resized)
            metrics_pitch = calculate_metrics(yp_pred, yp_true_resized)

            # Store metrics
            all_metrics["onset"].append(metrics_onset)
            all_metrics["note"].append(metrics_note)
            all_metrics["pitch"].append(metrics_pitch)

    # Aggregate metrics
    for key in all_metrics:
        precision = sum(m["precision"] for m in all_metrics[key]) / len(all_metrics[key])
        recall = sum(m["recall"] for m in all_metrics[key]) / len(all_metrics[key])
        f1_score = sum(m["f1"] for m in all_metrics[key]) / len(all_metrics[key])

        print(f"Metrics for {key}: Precision = {precision:.4f}, Recall = {recall:.4f}, F1 Score = {f1_score:.4f}")


if __name__ == "__main__":
    # Define paths and parameters
    model_path = "basic_pitch_epoch_1.pth"
    # Load config.json
    with open("config.json", "r") as f:
        config = json.load(f)
    data_dir = config.get("data_dir", "./data/maestro")  # Default to ./data/maestro if missing

    print(f"Data directory: {data_dir}")  # Debugging
    batch_size = 4

    # Choose device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Run evaluation
    load_and_evaluate(model_path, data_dir, batch_size=batch_size, device=device)
