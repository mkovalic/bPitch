import numpy as np
import scipy.signal

class NoteCreator:
    def __init__(self, threshold_onset=0.5, threshold_note=0.3, tolerance_frames=11, min_duration_ms=120, sample_rate=22050, hop_length=243):
        """
        Initialize the NoteCreator with thresholds and other parameters.

        Args:
            threshold_onset (float): Threshold for peak-picking Yo.
            threshold_note (float): Threshold for note creation in Yn.
            tolerance_frames (int): Tolerance for Yn falling below the threshold.
            min_duration_ms (float): Minimum duration for a note (in milliseconds).
            sample_rate (int): Audio sample rate.
            hop_length (int): Hop length used for the model.
        """
        self.threshold_onset = threshold_onset
        self.threshold_note = threshold_note
        self.tolerance_frames = tolerance_frames
        self.min_duration_frames = int((min_duration_ms / 1000) * sample_rate / hop_length)
        self.sample_rate = sample_rate
        self.hop_length = hop_length

    def create_notes(self, yo, yn, yp):
        """
        Create note events from Yo, Yn, and Yp.

        Args:
            yo (np.ndarray): Onset posteriorgram (shape: time_steps, frequency_bins).
            yn (np.ndarray): Note event posteriorgram (shape: time_steps, frequency_bins).
            yp (np.ndarray): Pitch posteriorgram (shape: time_steps, frequency_bins).

        Returns:
            list: List of notes as (start_time, end_time, pitch).
            list: List of multi-pitch estimates as (time, pitch).
        """
        notes = []
        time_steps, freq_bins = yo.shape

        # Step 1: Peak-picking in Yo
        onset_candidates = []
        for f in range(freq_bins):
            peaks, _ = scipy.signal.find_peaks(yo[:, f], height=self.threshold_onset)
            onset_candidates.extend([(t, f) for t in peaks])

        # Sort onset candidates by time in descending order
        onset_candidates.sort(reverse=True, key=lambda x: x[0])

        # Step 2: Create notes from Yo
        for t0, f in onset_candidates:
            if yn[t0, f] < self.threshold_note:
                continue

            # Track forward in time through Yn
            t1 = t0
            while t1 < time_steps and yn[t1, f] >= self.threshold_note:
                t1 += 1

            # Ensure the note meets minimum duration
            if t1 - t0 >= self.min_duration_frames:
                start_time = t0 * self.hop_length / self.sample_rate
                end_time = t1 * self.hop_length / self.sample_rate
                pitch = f
                notes.append((start_time, end_time, pitch))

                # Set Yn values to 0 for this note
                yn[t0:t1, f] = 0

        # Step 3: Additional notes from Yn
        remaining_candidates = np.argwhere(yn > self.threshold_note)
        remaining_candidates = sorted(remaining_candidates, key=lambda x: yn[x[0], x[1]], reverse=True)

        for t, f in remaining_candidates:
            if yn[t, f] < self.threshold_note:
                continue

            # Track forward and backward in time through Yn
            t_start, t_end = t, t
            while t_start > 0 and yn[t_start, f] >= self.threshold_note:
                t_start -= 1
            while t_end < time_steps and yn[t_end, f] >= self.threshold_note:
                t_end += 1

            # Ensure the note meets minimum duration
            if t_end - t_start >= self.min_duration_frames:
                start_time = t_start * self.hop_length / self.sample_rate
                end_time = t_end * self.hop_length / self.sample_rate
                pitch = f
                notes.append((start_time, end_time, pitch))

                # Set Yn values to 0 for this note
                yn[t_start:t_end, f] = 0

        # Step 4: Multi-pitch estimation from Yp
        pitch_candidates = np.argwhere(yp > self.threshold_note)
        multi_pitch = [(t * self.hop_length / self.sample_rate, f) for t, f in pitch_candidates]

        return notes, multi_pitch

