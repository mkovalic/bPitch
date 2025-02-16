import torch
from model import total_loss, class_balanced_bce_loss, standard_bce_loss

# Dummy predictions and ground truth
yo_true = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.float32)  # Ground truth (onset)
yo_pred = torch.tensor([[[0.1, 0.9], [0.8, 0.2]]], dtype=torch.float32)  # Predictions

yn_true = torch.tensor([[[1, 0], [0, 1]]], dtype=torch.float32)  # Ground truth (notes)
yn_pred = torch.tensor([[[0.9, 0.1], [0.2, 0.8]]], dtype=torch.float32)  # Predictions

yp_true = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.float32)  # Ground truth (pitches)
yp_pred = torch.tensor([[[0.2, 0.8], [0.7, 0.3]]], dtype=torch.float32)  # Predictions

# Compute individual losses
loss_yo = class_balanced_bce_loss(yo_true, yo_pred, pos_weight=0.95, neg_weight=0.05)
loss_yn = standard_bce_loss(yn_true, yn_pred)
loss_yp = standard_bce_loss(yp_true, yp_pred)

# Total loss
total = total_loss(yo_true, yo_pred, yn_true, yn_pred, yp_true, yp_pred)

print(f"Loss Yo (Onset): {loss_yo.item():.4f}")
print(f"Loss Yn (Notes): {loss_yn.item():.4f}")
print(f"Loss Yp (Pitch): {loss_yp.item():.4f}")
print(f"Total Loss: {total.item():.4f}")

