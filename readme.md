# Music transcription (audio to midi) project

This repository contains source code for my attempt at recreating [Spotify’s Basic Pitch](https://basicpitch.spotify.com/) music transcription model. The original Basic Pitch is built in TensorFlow, whereas **this** version is implemented in **PyTorch** and is currently trained only on the **MAESTRO** dataset. The model was trained on Apple silicon but can be modifed to run on CPU/GPU

---

## Overview

- **Goal**: Accurately transcribe piano audio (onsets, pitches, and note durations) from the MAESTRO dataset into MIDI.
- **Difference from Spotify’s Basic Pitch**:
  - Uses **PyTorch** instead of TensorFlow.
  - Is trained solely on the **MAESTRO dataset** (instead of multiple or augmented datasets).
---

## Features

1. **Harmonic CQT (HCQT) Preprocessing**  
   Converts raw audio into an HCQT representation with configurable harmonics.
2. **MIDI Label Generation**  
   Uses PrettyMIDI to parse MIDI files and align note events with the audio frames.
3. **PyTorch Training Pipeline**  
   Includes a dataset loader (`MaestroDataset`) and training scripts (to be expanded).
4. **Onset/Pitch/Note Outputs**  
   The model aims to predict **onset**, **pitch**, and **full note** activations.

---

## Installation

1. **Clone the repo** (or download the source code):
   ```
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```
2. **Set up a Python environment** (recommended):
    ```
    python -m venv ml_env
    source ml_env/bin/activate  # or ml_env\Scripts\activate on Windows
    ```
3. **Install dependencies:**
    ```
    pip install -r requirements.txt
    ```
---
## Data
- This project uses the MAESTRO dataset (v3) for training and validation. This dataset contains roughly 200 hours of paried audio and midi piano recordings.
    1. Download the MAESTRO dataset.
    2. Place the extracted audio and MIDI files in a folder, e.g. data/maestro-v3.0.0.
    3. Configure the path in config.json.   
    **Example `config.json`**
    ```
    {
        "data_dir": "<your-path>/data/maestro-v3.0.0/"
    }
    ```
---
## Troubleshooting
- **JSON Decode Error**
    - Check your `config.json` syntax (no trailing commas or comments).
- **Onset–Note Mismatches**
    - This is a known challenge in aligning short frames with exact note starts. We have a tolerance-based mismatch check in `verifyLoader.py`.
- **Data Folder Not Found**
    - Make sure `data_dir` is set properly in `config.json` or as an env var.
---
## Acknowledgments
- **Spotify’s Basic Pitch** for the original idea and inspiration.
- **MAESTRO** dataset creators for providing high-quality piano recordings and aligned MIDI.
- **Librosa**, **PrettyMIDI**, and the **PyTorch** community for their awesome tools and libraries.