"""Released Basic Pitch shallow architecture with explicit (B,H,T,F) axes.

Spotify's released implementation skips the first convolution in Figure 1.
This version follows that implementation (16,780 trainable parameters; preprocessing normalization is fixed).
"""

import torch
from torch import nn
import torch.nn.functional as F
from audio_features import FEATURES


class BasicPitchModel(nn.Module):
    version = 2

    def __init__(self):
        super().__init__()
        self.pitch_features = nn.Sequential(
            nn.Conv2d(8, 8, (3, 39), padding=(1, 19)), nn.BatchNorm2d(8), nn.ReLU()
        )
        self.pitch_head = nn.Conv2d(8, 1, (5, 5), padding=(2, 2))
        self.note_head = nn.Sequential(
            nn.Conv2d(1, 32, (7, 7), stride=(1, 3), padding=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 1, (7, 3), padding=(3, 1)),
        )
        self.onset_features = nn.Sequential(
            nn.Conv2d(8, 32, (5, 5), stride=(1, 3), padding=(2, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.onset_head = nn.Conv2d(33, 1, (3, 3), padding=(1, 1))

    def forward(self, x, return_logits=False):
        if x.ndim != 4 or x.shape[1] != 8 or x.shape[-1] != FEATURES.n_pitch_bins:
            raise ValueError("Expected input (batch, 8, time, 264)")
        pitch = self.pitch_head(self.pitch_features(x))
        note = self.note_head(torch.sigmoid(pitch))
        onset = self.onset_head(
            torch.cat((self.onset_features(x), torch.sigmoid(note)), dim=1)
        )
        outputs = {"onset": onset[:, 0], "pitch": pitch[:, 0], "note": note[:, 0]}
        return (
            outputs
            if return_logits
            else {k: torch.sigmoid(v) for k, v in outputs.items()}
        )


def compute_loss_components(outputs, labels, from_logits=True):
    """Masked BCE with onset class means balanced to preserve sparse starts."""
    losses = {}
    for key in ("onset", "pitch", "note"):
        prediction, target = outputs[key], labels[key]
        if prediction.shape != target.shape:
            raise ValueError(
                f"{key} shapes differ: {prediction.shape} vs {target.shape}"
            )
        valid = labels["mask"].unsqueeze(-1).expand_as(target)
        if not valid.any():
            raise ValueError("Batch contains no valid audio frames")
        element_loss = (
            F.binary_cross_entropy_with_logits
            if from_logits
            else F.binary_cross_entropy
        )(prediction, target, reduction="none")
        if key == "onset":
            positives, negatives = valid & (target > 0.5), valid & (target <= 0.5)
            loss = prediction.sum() * 0
            if positives.any():
                loss = loss + 0.95 * element_loss[positives].mean()
            if negatives.any():
                loss = loss + 0.05 * element_loss[negatives].mean()
            losses[key] = loss
        else:
            losses[key] = element_loss[valid].mean()
    return losses


def compute_loss(outputs, labels, from_logits=True):
    return sum(compute_loss_components(outputs, labels, from_logits).values())
