"""Decode fixed-resolution probability grids into MIDI-pitched note events."""

import math
import numpy as np
from scipy.signal import find_peaks
from audio_features import FEATURES


class NoteCreator:
    def __init__(
        self,
        threshold_onset=0.5,
        threshold_note=0.3,
        tolerance_frames=11,
        min_duration_ms=120,
        sample_rate=FEATURES.sample_rate,
        hop_length=FEATURES.hop_length,
        min_midi=FEATURES.min_midi,
    ):
        self.threshold_onset = threshold_onset
        self.threshold_note = threshold_note
        self.tolerance_frames = tolerance_frames
        self.min_duration_frames = max(
            1, math.ceil(min_duration_ms / 1000 * sample_rate / hop_length)
        )
        self.frame_seconds = hop_length / sample_rate
        self.min_midi = min_midi

    def create_notes(self, yo, yn, yp=None):
        yo, yn = np.asarray(yo), np.asarray(yn)
        if yo.shape != yn.shape or yo.ndim != 2 or yo.shape[1] != FEATURES.n_notes:
            raise ValueError("Onset and note grids must both be (time, 88)")
        if yp is not None and np.asarray(yp).shape != (len(yn), FEATURES.n_pitch_bins):
            raise ValueError("Pitch grid must be (time, 264)")
        active = yn >= self.threshold_note
        consumed = np.zeros_like(active)
        notes = []

        def trace(seed, pitch, direction):
            last, gaps, position = seed, 0, seed + direction
            while 0 <= position < len(yn) and not consumed[position, pitch]:
                if active[position, pitch]:
                    last, gaps = position, 0
                else:
                    gaps += 1
                    if gaps > self.tolerance_frames:
                        break
                position += direction
            return last

        def emit(first, last, pitch):
            consumed[first : last + 1, pitch] = True
            if last - first + 1 >= self.min_duration_frames:
                notes.append(
                    (
                        first * self.frame_seconds,
                        (last + 1) * self.frame_seconds,
                        int(pitch + self.min_midi),
                    )
                )

        candidates = []
        for pitch in range(yo.shape[1]):
            # Padding includes real onsets at the first/last frame.
            peaks, _ = find_peaks(np.pad(yo[:, pitch], 1), height=self.threshold_onset)
            candidates.extend((int(t - 1), pitch) for t in peaks)
        for first, pitch in sorted(candidates, reverse=True):
            if active[first, pitch] and not consumed[first, pitch]:
                emit(first, trace(first, pitch, 1), pitch)
        while True:
            remaining = np.where(active & ~consumed, yn, -1)
            flat = int(np.argmax(remaining))
            seed, pitch = np.unravel_index(flat, yn.shape)
            if remaining[seed, pitch] < self.threshold_note:
                break
            emit(trace(seed, pitch, -1), trace(seed, pitch, 1), pitch)
        multi_pitch = []
        if yp is not None:
            for frame, row in enumerate(np.asarray(yp)):
                peaks, _ = find_peaks(np.pad(row, 1), height=self.threshold_note)
                multi_pitch.extend(
                    (frame * self.frame_seconds, self.min_midi + (p - 1) / 3)
                    for p in peaks
                )
        return sorted(notes), multi_pitch


def notes_to_midi(notes, path, duration=None):
    import pretty_midi

    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    for start, end, pitch in notes:
        end = min(end, duration) if duration is not None else end
        if end > start:
            instrument.notes.append(
                pretty_midi.Note(
                    velocity=100, pitch=int(pitch), start=float(start), end=float(end)
                )
            )
    midi.instruments.append(instrument)
    midi.write(str(path))
    return midi
