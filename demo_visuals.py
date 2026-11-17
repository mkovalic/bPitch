"""Compact waveform and piano-roll data for the local and static demos."""

import numpy as np


def visualization(audio, sample_rate, notes):
    samples = np.asarray(audio, dtype=np.float64)
    count = min(640, len(samples))
    if count == 0:
        raise ValueError("Cannot draw empty audio")
    # As in harmonixset-ui, use every sample and normalize peak and RMS together.
    edges = np.linspace(0, len(samples), count + 1, dtype=int)
    peaks = np.maximum.reduceat(np.abs(samples), edges[:-1])
    rms = np.sqrt(np.add.reduceat(samples * samples, edges[:-1]) / np.diff(edges))
    kernel = [1, 2, 3, 2, 1]
    def smooth(series):
        return np.convolve(np.pad(series, 2), kernel, mode="valid") / weights

    weights = np.convolve(np.pad(np.ones(count), 2), kernel, mode="valid")
    peaks = smooth(peaks)
    scale = peaks.max() or 1.0
    return {
        "duration": len(samples) / sample_rate,
        "peaks": (peaks / scale).round(4).tolist(),
        "rms": (smooth(rms) / scale).round(4).tolist(),
        "events": [[float(start), float(end), int(pitch)] for start, end, pitch in notes],
    }
