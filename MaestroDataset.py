import os
import torch
from torch.utils.data import Dataset
import librosa
import pretty_midi
import numpy as np

class MaestroDataset(Dataset):
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
        """Retrieve paths of paired audio and MIDI files in the dataset directory."""
        data_paths = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.wav'):
                    audio_path = os.path.join(root, file)
                    midi_path = audio_path.replace('.wav', '.midi')
                    if os.path.exists(midi_path):
                        data_paths.append((audio_path, midi_path))
        return data_paths

    def _load_and_preprocess_audio(self, audio_path, max_length=22050 * 10):
        """
        Load and preprocess audio by converting it to a Harmonic CQT representation.

        Args:
            audio_path (str): Path to the audio file.
            max_length (int): Maximum length of audio in samples (default: 10 seconds).

        Returns:
            np.ndarray: HCQT representation of the audio, shape (n_harmonics, n_bins, time_steps).
        """
        try:
            # Load audio file with target sample rate
            audio, _ = librosa.load(audio_path, sr=self.sample_rate)
        except Exception as e:
            raise ValueError(f"Error loading audio file {audio_path}: {e}")

        # Truncate or pad the audio to max_length
        if len(audio) > max_length:
            audio = audio[:max_length]
        else:
            audio = np.pad(audio, (0, max_length - len(audio)), mode='constant')

        # Parameters for HCQT
        hop_length = 243  # ~11 ms hop size
        fmin = librosa.note_to_hz('C0')  # Minimum frequency (C0, 32.7 Hz)
        bins_per_octave = 36  # 3 bins per semitone
        n_bins = self.n_freq_bins  # Total bins (252 for 7 octaves)

        # Compute the HCQT (fundamental + harmonics)
        hcqt = librosa.cqt(
            audio,
            sr=self.sample_rate,
            hop_length=hop_length,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            fmin=fmin,
            window='hann',
            pad_mode='reflect'
        )
        hcqt = np.abs(hcqt)

        # Stack harmonics (fundamental + 7 harmonics + 1 sub-harmonic)
        harmonic_stack = [hcqt]
        #for i, harmonic in enumerate(harmonic_stack):
            #print(f"Harmonic {i} Shape: {harmonic.shape}")

        for harmonic in range(2, self.n_harmonics + 1):
            harmonic_cqt = librosa.cqt(
                audio,
                sr=self.sample_rate,
                hop_length=hop_length,
                n_bins=n_bins,
                bins_per_octave=bins_per_octave,
                fmin=fmin * harmonic,  # Shift frequency for harmonics
                window='hann',
                pad_mode='reflect'
            )
            harmonic_stack.append(np.abs(harmonic_cqt))

        return np.stack(harmonic_stack, axis=0)
    
    def _generate_labels_from_midi(self, midi_path, audio_shape):
        """
        Generate onset, pitch, and note event matrices from the MIDI file.

        Args:
            midi_path (str): Path to the MIDI file.
            audio_shape (tuple): Shape of the audio HCQT (n_harmonics, n_bins, time_frames).
            
        Returns:
            onset_matrix, pitch_matrix, note_matrix
        """
        midi_data = pretty_midi.PrettyMIDI(midi_path)

        # audio_shape is (n_harmonics, n_freq_bins, n_time_frames)
        n_time_frames = audio_shape[2]
        n_freq_bins = self.n_freq_bins  # e.g., 224

        onset_matrix = np.zeros((n_time_frames, n_freq_bins), dtype=np.float32)
        pitch_matrix = np.zeros((n_time_frames, n_freq_bins), dtype=np.float32)
        note_matrix  = np.zeros((n_time_frames, n_freq_bins), dtype=np.float32)

        # Time grid in seconds for each HCQT frame
        hop_length = 243  # ~11 ms
        times = np.arange(n_time_frames) * (hop_length / self.sample_rate)

        # We'll use the same fmin/bins_per_octave that were used in _load_and_preprocess_audio
        fmin = librosa.note_to_hz('C0')       # ~16.35 Hz
        bins_per_octave = 36                 # 3 bins per semitone

        # Iterate over all notes in the first instrument (assuming piano)
        for note in midi_data.instruments[0].notes:
            note_start = note.start  # in seconds
            note_end   = note.end
            midi_pitch = note.pitch  # integer MIDI pitch, e.g. 60 = Middle C

            # 1) Convert MIDI pitch to frequency
            f_note = librosa.midi_to_hz(midi_pitch)

            # 2) Map frequency to bin index
            #    pitch_bin_float = bins_per_octave * log2(f_note / fmin)
            pitch_bin_float = bins_per_octave * np.log2(f_note / fmin)
            pitch_bin = int(round(pitch_bin_float))

            # 3) Clamp to valid freq bin range
            if pitch_bin < 0 or pitch_bin >= n_freq_bins:
                # This note lies outside the frequency range of the CQT
                continue

            # Convert note’s start/end times in seconds -> frame indices
            start_frame = np.searchsorted(times, note_start)
            end_frame   = np.searchsorted(times, note_end)

            # Ensure note spans at least one frame
            if start_frame == end_frame:
                end_frame = min(start_frame + 1, n_time_frames - 1)

            # Clamp to valid frame range
            start_frame = min(max(start_frame, 0), n_time_frames - 1)
            end_frame   = min(max(end_frame, 0),   n_time_frames - 1)

            # Mark pitch, note, and onset
            pitch_matrix[start_frame:end_frame, pitch_bin] = 1.0
            note_matrix[start_frame:end_frame, pitch_bin]  = 1.0
            onset_matrix[start_frame, pitch_bin]           = 1.0

        return onset_matrix, pitch_matrix, note_matrix
    
    """
    def _generate_labels_from_midi(self, midi_path, audio_shape):
        
        Generate onset, pitch, and note event matrices from the MIDI file.

        Args:
            midi_path (str): Path to the MIDI file.
            audio_shape (tuple): Shape of the audio HCQT (batch_size, harmonics, time_frames, freq_bins).

        Returns:
            onset_matrix, pitch_matrix, note_matrix: Matrices representing the labels for onsets, pitches, and notes.

        midi_data = pretty_midi.PrettyMIDI(midi_path)
        n_time_frames = audio_shape[2]  # Use the number of time frames from the audio HCQT.
        n_freq_bins = self.n_freq_bins  # Number of frequency bins.

        # Create empty matrices for onset, pitch, and note events
        onset_matrix = np.zeros((n_time_frames, n_freq_bins))
        pitch_matrix = np.zeros((n_time_frames, n_freq_bins))
        note_matrix = np.zeros((n_time_frames, n_freq_bins))

        # Calculate time intervals from audio
        hop_length = 243  # Using 11 ms hop size (22050 Hz * 0.011 sec)
        times = np.arange(n_time_frames) * (hop_length / self.sample_rate)

        # Debugging: Print details
        #print(f"Calculated Times Array: {times[:10]} ... {times[-10:]}")
        #print(f"MIDI File Duration: {midi_data.get_end_time()} seconds")
        #print(f"Number of Time Frames: {n_time_frames}")

        # Populate matrices based on MIDI note information
        for note in midi_data.instruments[0].notes:  # Assuming a single instrument (e.g., piano)
            start_frame = np.searchsorted(times, note.start)
            end_frame = np.searchsorted(times, note.end)
            pitch_bin = note.pitch - 21  # Map MIDI pitch to bin (A0 is MIDI 21)

            # Ensure pitch_bin and frames are within valid ranges
            if 0 <= pitch_bin < n_freq_bins:
                # Ensure note spans at least one frame
                if start_frame == end_frame:
                    end_frame = min(start_frame + 1, n_time_frames - 1)

                # Clamp frame indices to avoid out-of-bounds errors
                start_frame = min(max(start_frame, 0), n_time_frames - 1)
                end_frame = min(max(end_frame, 0), n_time_frames - 1)

                # Update pitch, note, and onset matrices
                pitch_matrix[start_frame:end_frame, pitch_bin] = 1
                note_matrix[start_frame:end_frame, pitch_bin] = 1
                onset_matrix[start_frame, pitch_bin] = 1  # Mark onset only at the start frame

        #Debugging: Print sample MIDI note times and alignment
        print("Sample MIDI Notes:")
        for note in midi_data.instruments[0].notes[:10]:  # Print first 10 notes
            print(f"Note Start: {note.start}, Note End: {note.end}, Pitch: {note.pitch}")

        print(f"Onset Matrix Sum: {onset_matrix.sum()}")
        print(f"Pitch Matrix Sum: {pitch_matrix.sum()}")
        print(f"Note Matrix Sum: {note_matrix.sum()}")

        return onset_matrix, pitch_matrix, note_matrix
        """


    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data_paths)

    def __getitem__(self, idx):
        """Return a single sample consisting of audio data and labels."""
        audio_path, midi_path = self.data_paths[idx]

        # Load and preprocess audio
        audio_data = self._load_and_preprocess_audio(audio_path)
        
        # Generate labels from MIDI
        onset_matrix, pitch_matrix, note_matrix = self._generate_labels_from_midi(midi_path, audio_data.shape)

        # Convert to PyTorch tensors
        audio_data = torch.tensor(audio_data, dtype=torch.float32)
        onset_matrix = torch.tensor(onset_matrix, dtype=torch.float32)
        pitch_matrix = torch.tensor(pitch_matrix, dtype=torch.float32)
        note_matrix = torch.tensor(note_matrix, dtype=torch.float32)

        return audio_data, {'onset': onset_matrix, 'pitch': pitch_matrix, 'note': note_matrix}

