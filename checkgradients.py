# Example using the full model
import torch
from model import BasicPitchModel, total_loss, class_balanced_bce_loss

# Initialize model and dummy data
model = BasicPitchModel()
inputs = torch.randn(4, 8, 224, 908)  # Batch size 4, 8 harmonics
yo_true = torch.randint(0, 2, (4, 1, 218, 433)).float()
yn_true = torch.randint(0, 2, (4, 1, 218, 433)).float()
yp_true = torch.randint(0, 2, (4, 1, 218, 433)).float()

# Forward pass
yo_pred, yp_pred, yn_pred = model(inputs)

# Compute loss
loss = total_loss(yo_true, yo_pred, yn_true, yn_pred, yp_true, yp_pred)
print(f"Loss: {loss.item()}")

# Backward pass
loss.backward()

# Check gradients
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Gradient for {name}: {param.grad.norm()}")

perfect_pred = torch.tensor([[[1, 0], [0, 1]]], dtype=torch.float32)
perfect_loss = total_loss(perfect_pred, perfect_pred, perfect_pred, perfect_pred, perfect_pred, perfect_pred)
print(f"Perfect Prediction Loss: {perfect_loss.item()}")

worst_pred = 1 - perfect_pred
worst_loss = total_loss(perfect_pred, worst_pred, perfect_pred, worst_pred, perfect_pred, worst_pred)
print(f"Worst Prediction Loss: {worst_loss.item()}")

