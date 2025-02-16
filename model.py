import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class BasicPitchModel(nn.Module):
    def __init__(self):
        super(BasicPitchModel, self).__init__()

        # First Conv Block (Processes HCQT)
        self.conv1 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(5, 5))
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()

        # Second Conv Block (Refines features for Yp)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 39))  # Large receptive field in frequency
        self.batch_norm2 = nn.BatchNorm2d(32)

        # Yp (Pitch Posteriorgram) path
        self.conv_pitch = nn.Conv2d(32, 1, kernel_size=(5, 5))  # Small kernel for final Yp
        self.sigmoid = nn.Sigmoid()

        # Yn (Note Event Posteriorgram) path
        self.conv_note1 = nn.Conv2d(1, 32, kernel_size=(7, 7), stride=(1, 2), padding=(3, 3))  # Larger receptive field
        self.conv_note2 = nn.Conv2d(32, 1, kernel_size=(7, 3), padding=(3, 1))  # Final note event

        # Yo (Onset Posteriorgram) path
        self.conv_onset_audio = nn.Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 2), padding=(2, 2))
        self.conv_onset_concat = nn.Conv2d(33, 1, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x):
        # HCQT Processing
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)

        # Feature Refinement
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)

        # Yp Path
        yp = self.conv_pitch(x)
        yp = self.sigmoid(yp)

        # Yn Path
        yn = self.conv_note1(yp)
        yn = self.relu(yn)
        yn = self.conv_note2(yn)
        yn = self.sigmoid(yn)

        # Yo Path
        audio_onset_features = self.conv_onset_audio(x)
        audio_onset_features = self.relu(audio_onset_features)

        # Adjust `Yn` dimensions to match `Yo`
        diff_h = audio_onset_features.shape[2] - yn.shape[2]
        diff_w = audio_onset_features.shape[3] - yn.shape[3]
        pad = [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2]
        yn_padded = F.pad(yn, pad)

        # Adjust `Yp` dimensions to match `Yn`
        yp_resized = F.interpolate(yp, size=(yn_padded.shape[2], yn_padded.shape[3]), mode="bilinear", align_corners=False)

        # Concatenate Yn and audio features
        concat_features = torch.cat([audio_onset_features, yn_padded], dim=1)
        yo = self.conv_onset_concat(concat_features)
        yo = self.sigmoid(yo)

        return yo, yp_resized, yn_padded

# Binary Cross-Entropy Loss function
bce_loss = nn.BCELoss(reduction='none')  # We'll calculate the loss for each element without reduction

# Class-balanced BCE for Yo (onset posteriorgram)
def class_balanced_bce_loss(y_true, y_pred, pos_weight=0.95, neg_weight=0.05):
    """
    Class-balanced binary cross-entropy loss.
    
    Args:
        y_true: Ground truth tensor (0 or 1).
        y_pred: Predicted probabilities tensor.
        pos_weight: Weight for the positive class.
        neg_weight: Weight for the negative class.
    
    Returns:
        Loss value.
    """
    # Compute BCE for each element
    loss = bce_loss(y_pred, y_true)
    
    # Apply class weights
    weights = y_true * pos_weight + (1 - y_true) * neg_weight
    loss = loss * weights
    
    return torch.mean(loss)

# Standard BCE Loss for Yn and Yp
def standard_bce_loss(y_true, y_pred):
    """
    Standard binary cross-entropy loss.
    
    Args:
        y_true: Ground truth tensor (0 or 1).
        y_pred: Predicted probabilities tensor.
    
    Returns:
        Loss value.
    """
    return torch.mean(bce_loss(y_pred, y_true))

# Total Loss function
def total_loss(yo_true, yo_pred, yn_true, yn_pred, yp_true, yp_pred, onset_pos_weight=0.95, onset_neg_weight=0.05):
    """
    Compute the total loss for the model, summing up the losses for Yo (onset), Yn (note events), and Yp (pitch).
    
    Args:
        yo_true: Ground truth onset posteriorgram.
        yo_pred: Predicted onset posteriorgram.
        yn_true: Ground truth note event posteriorgram.
        yn_pred: Predicted note event posteriorgram.
        yp_true: Ground truth pitch posteriorgram.
        yp_pred: Predicted pitch posteriorgram.
        onset_pos_weight: Weight for the positive onset class.
        onset_neg_weight: Weight for the negative onset class.
    
    Returns:
        Total loss value.
    """
    # Class-balanced BCE loss for Yo (onsets)
    loss_yo = class_balanced_bce_loss(yo_true, yo_pred, onset_pos_weight, onset_neg_weight)
    
    # Standard BCE loss for Yn (note events) and Yp (pitch)
    loss_yn = standard_bce_loss(yn_true, yn_pred)
    loss_yp = standard_bce_loss(yp_true, yp_pred)
    
    # Total loss is the sum of the three losses
    return loss_yo + loss_yn + loss_yp

def visualize_outputs(yo, yp, yn):
    """
    Visualize the Yo, Yp, and Yn outputs as heatmaps.
    
    Args:
        yo: Onset posteriorgram output.
        yp: Pitch posteriorgram output.
        yn: Note event posteriorgram output.
    """
    outputs = {'Onset (Yo)': yo, 'Pitch (Yp)': yp, 'Note (Yn)': yn}
    batch_idx = 0  # Visualize the first sample in the batch

    for title, output in outputs.items():
        output_sample = output[batch_idx, 0].detach().cpu().numpy()  # Select batch and channel
        plt.figure(figsize=(10, 5))
        plt.title(title)
        plt.xlabel('Time Steps')
        plt.ylabel('Frequency Bins')
        plt.imshow(output_sample, aspect='auto', origin='lower', cmap='hot')
        plt.colorbar(label='Probability')
        plt.show()

def check_value_ranges(yo, yp, yn):
    """
    Check the value ranges of the model outputs.

    Args:
        yo: Onset posteriorgram output.
        yp: Pitch posteriorgram output.
        yn: Note event posteriorgram output.
    """
    outputs = {'Onset (Yo)': yo, 'Pitch (Yp)': yp, 'Note (Yn)': yn}
    
    for name, output in outputs.items():
        min_val = output.min().item()
        max_val = output.max().item()
        print(f"{name}: Min Value = {min_val}, Max Value = {max_val}")
        if not (0.0 <= min_val and max_val <= 1.0):
            print(f"Warning: {name} values are out of expected range [0, 1]!")
        else:
            print(f"{name} values are within the expected range [0, 1].")

#model = BasicPitchModel()
#sample_input = torch.randn(4, 8, 224, 908)  # Batch size 4
#yo, yp, yn = model(sample_input)
#print(f"Yo Shape: {yo.shape}, Yp Shape: {yp.shape}, Yn Shape: {yn.shape}")

#visualize_outputs(yo, yp, yn)

