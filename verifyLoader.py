import torch
import json
from torch.utils.data import DataLoader
from MaestroDataset import MaestroDataset
import matplotlib.pyplot as plt


# Initialize dataset and DataLoader
with open("config.json", "r") as f:
    config = json.load(f)
data_dir = config.get("data_dir", "./data/maestro")  # Default to ./data/maestro if missing
dataset = MaestroDataset(data_dir)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

def verifyInstall():
    ## Verify DataLoader initialization
    print(f"Number of samples in the dataset: {len(dataset)}")
    print(f"Number of batches in DataLoader: {len(dataloader)}")

def oneBatch():
    # Fetch a batch
    for batch_idx, (audio_batch, label_batch) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"  Audio Batch Shape: {audio_batch.shape}")  # Should match (batch_size, n_harmonics, time_steps, freq_bins)
        print(f"  Onset Labels Shape: {label_batch['onset'].shape}")  # Should match (batch_size, time_steps, freq_bins)
        print(f"  Pitch Labels Shape: {label_batch['pitch'].shape}")
        print(f"  Note Labels Shape: {label_batch['note'].shape}")
        assert audio_batch.shape[-1] == label_batch['onset'].shape[1], "Time frames don't match!"
        assert audio_batch.shape[2] == label_batch['onset'].shape[2], "Frequency bins don't match!"

        print("Audio and labels are aligned!")

        break  # Check one batch only

def valueRange():
    # Fetch a single batch
    audio_batch, label_batch = next(iter(dataloader))

    # Check audio data range
    print(f"Audio Data Range: {audio_batch.min().item()} to {audio_batch.max().item()}")

    # Check label values
    for label_name in ['onset', 'pitch', 'note']:
        unique_values = torch.unique(label_batch[label_name])
        print(f"{label_name.capitalize()} Unique Values: {unique_values}")

def mismatch():
    audio_batch, label_batch = next(iter(dataloader))
    for i in range(audio_batch.size(0)):  # Loop through batch
        onset = label_batch['onset'][i]
        pitch = label_batch['pitch'][i]
        note = label_batch['note'][i]

        # Ensure onsets align with notes
        onset_note_mismatch = ((onset == 1) & (note == 0)).sum().item()
        print(f"Sample {i}: Onset-Note Mismatch Count: {onset_note_mismatch}")

        # Ensure onsets align with pitches
        onset_pitch_mismatch = ((onset == 1) & (pitch == 0)).sum().item()
        print(f"Sample {i}: Onset-Pitch Mismatch Count: {onset_pitch_mismatch}")

        if onset_note_mismatch > 0 or onset_pitch_mismatch > 0:
            print(f"Sample {i}: Problem detected. Visualizing...")
            plt.figure(figsize=(10, 4))
            plt.imshow(onset.numpy().T, aspect='auto', origin='lower')
            plt.title(f"Onset Matrix for Sample {i}")
            plt.colorbar()
            plt.show()

            plt.figure(figsize=(10, 4))
            plt.imshow(note.numpy().T, aspect='auto', origin='lower')
            plt.title(f"Note Matrix for Sample {i}")
            plt.colorbar()
            plt.show()

def mismatch_rate():
    audio_batch, label_batch = next(iter(dataloader))
    total_onsets = (label_batch['onset'] == 1).sum().item()
    
    onset_note_mismatch = ((label_batch['onset'] == 1) & (label_batch['note'] == 0)).sum().item()
    onset_pitch_mismatch = ((label_batch['onset'] == 1) & (label_batch['pitch'] == 0)).sum().item()

    # Calculate mismatch rates
    note_mismatch_rate = (onset_note_mismatch / total_onsets) * 100 if total_onsets > 0 else 0
    pitch_mismatch_rate = (onset_pitch_mismatch / total_onsets) * 100 if total_onsets > 0 else 0

    print(f"Note Mismatch Rate: {note_mismatch_rate:.2f}%")
    print(f"Pitch Mismatch Rate: {pitch_mismatch_rate:.2f}%")
    #mismatch_rate_with_tolerance(label_batch)
    mismatch_rate_per_onset(label_batch, tolerance=3)

def mismatch_rate_with_tolerance(label_batch, tolerance=3):
    onset = label_batch['onset']
    pitch = label_batch['pitch']
    note = label_batch['note']

    # Convert to binary tensors (0 or 1) before applying bitwise OR
    onset = (onset > 0).to(torch.bool)  # Convert to boolean tensor
    pitch = (pitch > 0).to(torch.bool)  # Convert to boolean tensor
    note = (note > 0).to(torch.bool)  # Convert to boolean tensor

    # Extend onset frames by tolerance
    extended_onset = onset.clone()
    for t in range(1, tolerance + 1):
        extended_onset[:-t] |= onset[t:]  # Bitwise OR operation for alignment
        extended_onset[t:] |= onset[:-t]  # Bitwise OR operation for alignment

    # Recalculate mismatches
    onset_note_mismatch = ((extended_onset == 1) & (note == 0)).sum().item()
    onset_pitch_mismatch = ((extended_onset == 1) & (pitch == 0)).sum().item()
    total_onsets = (onset == 1).sum().item()

    note_mismatch_rate = (onset_note_mismatch / total_onsets) * 100 if total_onsets > 0 else 0
    pitch_mismatch_rate = (onset_pitch_mismatch / total_onsets) * 100 if total_onsets > 0 else 0

    print(f"Note Mismatch Rate (with tolerance): {note_mismatch_rate:.2f}%")
    print(f"Pitch Mismatch Rate (with tolerance): {pitch_mismatch_rate:.2f}%")

import torch

def mismatch_rate_per_onset(labels, tolerance=1):
    """
    Calculates mismatch rates by checking each original onset
    for a matching label (note or pitch) within ±tolerance frames.
    
    Args:
        labels (dict): Dictionary containing 'onset', 'note', 'pitch' tensors.
                       Each should have shape (batch_size, time_steps, freq_bins).
        tolerance (int): Number of frames to allow on either side of the onset frame.
    
    Returns:
        None (prints mismatch rates)
    """
    onset = (labels['onset'] > 0)  # boolean tensor of shape (B, T, F)
    note = (labels['note']  > 0)
    pitch = (labels['pitch'] > 0)
    
    B, T, F = onset.shape
    mismatch_onset_note = 0
    mismatch_onset_pitch = 0
    total_onsets = 0

    # Loop over the batch
    for b in range(B):
        # Collect all (time, freq) coordinates where onset == 1
        # Example: shape (#onsets_in_sample, 2)
        onset_coords = torch.nonzero(onset[b], as_tuple=False)
        
        # For each onset coordinate, check if there's a matching note/pitch
        # within ±tolerance frames at that same frequency bin.
        for (t, f) in onset_coords:
            total_onsets += 1

            # Define time window around the onset
            t_start = max(0, t - tolerance)
            t_end   = min(T, t + tolerance + 1)  # +1 for inclusive slice
            
            # Check note mismatch
            if not note[b, t_start:t_end, f].any():
                mismatch_onset_note += 1

            # Check pitch mismatch
            if not pitch[b, t_start:t_end, f].any():
                mismatch_onset_pitch += 1

    # Calculate mismatch percentages
    note_mismatch_rate = (mismatch_onset_note / total_onsets * 100) if total_onsets > 0 else 0.0
    pitch_mismatch_rate = (mismatch_onset_pitch / total_onsets * 100) if total_onsets > 0 else 0.0
    
    print(f"Note Mismatch Rate (±{tolerance} frames): {note_mismatch_rate:.2f}%")
    print(f"Pitch Mismatch Rate (±{tolerance} frames): {pitch_mismatch_rate:.2f}%")


def dummyLoop():
    for batch_idx, (audio_batch, label_batch) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"  Audio Shape: {audio_batch.shape}")
        print(f"  Onset Labels Shape: {label_batch['onset'].shape}")
        print(f"  Pitch Labels Shape: {label_batch['pitch'].shape}")
        print(f"  Note Labels Shape: {label_batch['note'].shape}")



#oneBatch()
#valueRange()
mismatch_rate()
#mismatch()
#dummyLoop()


"""# Test __getitem__ with a specific index
audio_data, labels = dataset[0]

# Print outputs for inspection
print(f"Audio Data Shape: {audio_data.shape}")
print(f"Onset Labels Shape: {labels['onset'].shape}")
print(f"Pitch Labels Shape: {labels['pitch'].shape}")
print(f"Note Labels Shape: {labels['note'].shape}")"""