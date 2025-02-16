import os
import json
import pandas as pd
import librosa
import pretty_midi

def validate_maestro_dataset(data_dir, metadata_file="maestro-v3.0.0.csv"):
    """
    Validates the integrity of the MAESTRO dataset.

    Args:
        data_dir (str): Path to the MAESTRO dataset directory.
        metadata_file (str): Name of the metadata file.

    Returns:
        None
    """
    metadata_path = os.path.join(data_dir, metadata_file)
    
    # Load metadata
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found at {metadata_path}")
        return
    
    metadata = pd.read_csv(metadata_path)
    print(f"Loaded metadata for {len(metadata)} samples.")
    
    missing_files = []
    inaccessible_files = []
    
    for _, row in metadata.iterrows():
        audio_path = os.path.join(data_dir, row["audio_filename"])
        midi_path = os.path.join(data_dir, row["midi_filename"])
        
        # Check if files exist
        if not os.path.exists(audio_path):
            missing_files.append(audio_path)
        if not os.path.exists(midi_path):
            missing_files.append(midi_path)
        
        # Test loading files
        try:
            librosa.load(audio_path, sr=None)
        except Exception as e:
            inaccessible_files.append((audio_path, str(e)))
        
        try:
            pretty_midi.PrettyMIDI(midi_path)
        except Exception as e:
            inaccessible_files.append((midi_path, str(e)))
    
    # Check dataset splits
    split_counts = metadata['split'].value_counts()
    print("\nDataset Split Counts:")
    print(split_counts)
    
    # Report results
    if missing_files:
        print(f"\nMissing Files ({len(missing_files)}):")
        for path in missing_files:
            print(f"  {path}")
    else:
        print("\nAll files are present.")
    
    if inaccessible_files:
        print(f"\nInaccessible Files ({len(inaccessible_files)}):")
        for path, error in inaccessible_files:
            print(f"  {path}: {error}")
    else:
        print("\nAll files are accessible.")

# Run validation
if __name__ == "__main__":
    # Load config.json
    with open("config.json", "r") as f:
        config = json.load(f)
    data_dir = config.get("data_dir", "./data/maestro")  # Default to ./data/maestro if missing

    validate_maestro_dataset(data_dir)
