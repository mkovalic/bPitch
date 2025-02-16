import os
import torch
from torch.utils.data import Dataset
import librosa
import pretty_midi
import numpy as np
import pandas as pd



class MaestroLoader(Dataset):
    def __init__(self, data_dir, sample_rate=22050, n_harmonics=8, n_freq_bins=224):
        """
        Args:
            data_dir (str): Path to the directory containing MAESTRO audio and MIDI files.
            sample_rate (int): Target sample rate for audio files.
            n_harmonics (int): Number of harmonics for the Harmonic CQT.
            n_freq_bins (int): Number of frequency bins for pitch-related matrices.
        """
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.n_harmonics = n_harmonics
        self.n_freq_bins = n_freq_bins

        # Get list of audio and MIDI file paths
        self.data_paths = self._get_data_paths()
    
    def _get_data_paths(self):
        """Retrieve paths of paired audio and MIDI files from metadata."""
        csv_path = os.path.join(self.data_dir, "maestro-v3.0.0.csv")
        metadata = pd.read_csv(csv_path)
        data_paths = []

        for _, row in metadata.iterrows():
            audio_path = os.path.join(self.data_dir, row["audio_filename"])
            midi_path = os.path.join(self.data_dir, row["midi_filename"])
            if os.path.exists(audio_path) and os.path.exists(midi_path):
                data_paths.append((audio_path, midi_path))
        return data_paths


    def _load_and_preprocess_audio(self, audio_path, max_length=22050 * 10):
        """Load and preprocess audio by converting it to a Harmonic CQT representation."""
        try:
            audio, _ = librosa.load(audio_path, sr=self.sample_rate)
        except Exception as e:
            raise ValueError(f"Error loading audio file {audio_path}: {e}")

        # Check if the audio is long enough for CQT computation
        if len(audio) < self.sample_rate:  # e.g., less than 1 second
            raise ValueError(f"Audio file {audio_path} is too short for processing.")
        
        audio, _ = librosa.load(audio_path, sr=self.sample_rate)
        if len(audio) > max_length:
            audio = audio[:max_length]
        else:
            audio = np.pad(audio, (0, max_length - len(audio)), mode='constant')

        # Compute Harmonic CQT (HCQT) with harmonics
        hop_length = 512  # Default hop length; adjust if needed
        hcqt = librosa.cqt(audio, sr=self.sample_rate, n_bins=self.n_freq_bins, bins_per_octave=12, hop_length=hop_length)
        print(f"Audio Time Frames from CQT: {hcqt.shape[1]}")  # Check time frames here
        hcqt = np.abs(hcqt)

        # Stack harmonics to capture harmonic structure
        harmonic_stack = [hcqt]
        for harmonic in range(2, self.n_harmonics + 1):
            harmonic_cqt = librosa.cqt(audio, sr=self.sample_rate, n_bins=self.n_freq_bins, bins_per_octave=12, hop_length=librosa.time_to_samples(1 / harmonic, sr=self.sample_rate))
            harmonic_stack.append(np.abs(harmonic_cqt))

        return np.stack(harmonic_stack, axis=0)

    def _generate_labels_from_midi(self, midi_path, audio_shape):
        """Generate onset, pitch, and note event matrices from the MIDI file."""
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        n_time_frames = audio_shape[1]  # Use time frames from audio as reference

        # Create empty matrices for onset, pitch, and note events
        onset_matrix = np.zeros((n_time_frames, self.n_freq_bins))
        pitch_matrix = np.zeros((n_time_frames, self.n_freq_bins))
        note_matrix = np.zeros((n_time_frames, self.n_freq_bins))

        # Calculate time intervals
        times = np.linspace(0, midi_data.get_end_time(), n_time_frames)

        # Populate matrices based on MIDI note information
        for note in midi_data.instruments[0].notes:  # Assuming a single instrument (piano)
            start_frame = np.searchsorted(times, note.start)
            end_frame = np.searchsorted(times, note.end)
            pitch_bin = note.pitch - pretty_midi.note_number_to_hz(21)  # Lowest note is A0 (MIDI 21)

            # Ensure pitch_bin is within valid range
            if 0 <= pitch_bin < self.n_freq_bins:
                pitch_matrix[start_frame:end_frame, pitch_bin] = 1
                note_matrix[start_frame:end_frame, pitch_bin] = 1
                onset_matrix[start_frame, pitch_bin] = 1  # Mark onset at start frame

        return onset_matrix, pitch_matrix, note_matrix

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data_paths)

    def __getitem__(self, idx):
        """Return a single sample consisting of audio data and labels."""
        audio_path, midi_path = self.data_paths[idx]

        # Load and preprocess audio
        audio_data = self._load_and_preprocess_audio(audio_path)
        print(f"Audio Time Frames in __getitem__: {audio_data.shape[2]}")
        
        # Generate labels from MIDI
        onset_matrix, pitch_matrix, note_matrix = self._generate_labels_from_midi(midi_path, audio_data.shape)
        print(f"Label Time Frames in __getitem__: {onset_matrix.shape[0]}")

        # Convert to PyTorch tensors
        audio_data = torch.tensor(audio_data, dtype=torch.float32)
        onset_matrix = torch.tensor(onset_matrix, dtype=torch.float32)
        pitch_matrix = torch.tensor(pitch_matrix, dtype=torch.float32)
        note_matrix = torch.tensor(note_matrix, dtype=torch.float32)

        return audio_data, {'onset': onset_matrix, 'pitch': pitch_matrix, 'note': note_matrix}

