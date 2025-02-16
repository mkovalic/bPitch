import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import BasicPitchModel, total_loss  # Import the model and loss function from model.py
from MaestroDataset import MaestroDataset
import torch.nn.functional as F

# Define the training loop
def train_model(model, dataloader, optimizer, num_epochs, device='mps'):

    print(f"Using device: {device}")
    model = model.to(device)  # Move model to GPU/CPU

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0  # For tracking loss

        for batch_idx, (inputs, labels) in enumerate(dataloader):
            # Move data to GPU/CPU
            inputs = inputs.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}

            # Extract and resize labels to match model outputs
            yo_true = F.interpolate(labels['onset'].unsqueeze(1), size=(218, 433), mode='bilinear', align_corners=False).squeeze(1).unsqueeze(1)
            yn_true = F.interpolate(labels['note'].unsqueeze(1), size=(218, 433), mode='bilinear', align_corners=False).squeeze(1).unsqueeze(1)
            yp_true = F.interpolate(labels['pitch'].unsqueeze(1), size=(218, 433), mode='bilinear', align_corners=False).squeeze(1).unsqueeze(1)

            # Zero the gradients for the optimizer
            optimizer.zero_grad()

            # Forward pass: compute model predictions
            yo_pred, yp_pred, yn_pred = model(inputs)

            # Debugging: Ensure shape compatibility
            if yo_pred.shape != yo_true.shape or yp_pred.shape != yp_true.shape or yn_pred.shape != yn_true.shape:
                print(f"Shape Mismatch - Yo: {yo_pred.shape} vs {yo_true.shape}, "
                      f"Yp: {yp_pred.shape} vs {yp_true.shape}, "
                      f"Yn: {yn_pred.shape} vs {yn_true.shape}")
                continue

            # Compute loss
            loss = total_loss(yo_true, yo_pred, yn_true, yn_pred, yp_true, yp_pred)

            # Backward pass: compute gradients
            loss.backward()

            # Optimization step: update model weights
            optimizer.step()

            # Track the running loss
            running_loss += loss.item()

            # Optionally log progress for every batch
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")

        # Calculate average loss per epoch
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

        # Optionally, save model checkpoints after each epoch
        torch.save(model.state_dict(), f"basic_pitch_epoch_{epoch + 1}.pth")

def main():
    # Hyperparameters
    batch_size = 4  # Adjust based on your system's capacity
    num_epochs = 50
    learning_rate = 0.001
    with open("config.json", "r") as f:
        config = json.load(f)
    data_dir = config.get("data_dir", "./data/maestro")  # Default to ./data/maestro if missing

    # Initialize dataset and dataloader
    train_dataset = MaestroDataset(data_dir)  # Replace with your dataset path
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Initialize the model
    model = BasicPitchModel()

    # Set up optimizer (Adam)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    """""
    # Check device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    # **Verify the forward pass here**
    sample_input, sample_labels = next(iter(train_dataloader))
    sample_input = sample_input.to(device)
    sample_labels = {k: v.to(device) for k, v in sample_labels.items()}
    yo_pred, yp_pred, yn_pred = model(sample_input)
    


    # Verify outputs
    print(f"Yo Shape: {yo_pred.shape}, Yp Shape: {yp_pred.shape}, Yn Shape: {yn_pred.shape}")
    print(f"Yo Range: Min = {yo_pred.min().item()}, Max = {yo_pred.max().item()}")
    print(f"Yp Range: Min = {yp_pred.min().item()}, Max = {yp_pred.max().item()}")
    print(f"Yn Range: Min = {yn_pred.min().item()}, Max = {yn_pred.max().item()}")

    # Check loss computation for a single batch
    yo_true = sample_labels['onset'].to(device)
    yn_true = sample_labels['note'].to(device)
    yp_true = sample_labels['pitch'].to(device)

    yo_true_resized = F.interpolate(yo_true.unsqueeze(1), size=(218, 433), mode='bilinear', align_corners=False).squeeze(1)
    yn_true_resized = F.interpolate(yn_true.unsqueeze(1), size=(218, 433), mode='bilinear', align_corners=False).squeeze(1)
    yp_true_resized = F.interpolate(yp_true.unsqueeze(1), size=(218, 433), mode='bilinear', align_corners=False).squeeze(1)

    # Add a channel dimension to the target tensors
    yo_true_resized = yo_true_resized.unsqueeze(1)
    yn_true_resized = yn_true_resized.unsqueeze(1)
    yp_true_resized = yp_true_resized.unsqueeze(1)

    loss = total_loss(yo_true_resized, yo_pred, yn_true_resized, yn_pred, yp_true_resized, yp_pred)
    print(f"Sample Loss: {loss.item()}")
    """""

    # Call the training loop
    train_model(model, train_dataloader, optimizer, num_epochs)

if __name__ == '__main__':
    main()
