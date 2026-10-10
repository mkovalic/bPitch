"""Shared audio/annotation coordinates for training and inference."""

from dataclasses import asdict, dataclass
import librosa
import numpy as np
import soundfile as sf
import warnings


@dataclass(frozen=True)
class FeatureConfig:
    version: int = 2
    sample_rate: int = 22050
    hop_length: int = 256
    window_seconds: float = 2.0
    min_midi: int = 21
    n_notes: int = 88
    bins_per_semitone: int = 3
    harmonics: tuple = (0.5, 1, 2, 3, 4, 5, 6, 7)

    @property
    def n_samples(self):
        return round(self.sample_rate * self.window_seconds)

    @property
    def n_frames(self):
        return 1 + self.n_samples // self.hop_length

    @property
    def n_pitch_bins(self):
        return self.n_notes * self.bins_per_semitone

    @property
    def frame_seconds(self):
        return self.hop_length / self.sample_rate

    def to_dict(self):
        result = asdict(self)
        result["harmonics"] = list(self.harmonics)
        return result


FEATURES = FeatureConfig()


def pad_audio(audio, config=FEATURES):
    audio = np.asarray(audio, dtype=np.float32)
    valid_samples = min(len(audio), config.n_samples)
    padded = np.zeros(config.n_samples, dtype=np.float32)
    padded[:valid_samples] = audio[:valid_samples]
    mask = np.arange(config.n_frames) * config.hop_length < valid_samples
    return padded, mask


def load_window(path, start_sample=0, config=FEATURES):
    """Seek in the source WAV and resample only the requested window."""
    with sf.SoundFile(str(path)) as source:
        start = round(start_sample * source.samplerate / config.sample_rate)
        source.seek(min(start, len(source)))
        count = round(config.n_samples * source.samplerate / config.sample_rate)
        audio = source.read(count, dtype="float32", always_2d=True).mean(axis=1)
        source_rate = source.samplerate
    if source_rate != config.sample_rate and len(audio):
        audio = librosa.resample(
            audio, orig_sr=source_rate, target_sr=config.sample_rate
        )
    return pad_audio(audio, config)


def compute_hcqt(audio, config=FEATURES):
    """One log-normalized CQT shifted into harmonic channels: (H,T,F)."""
    if len(audio) != config.n_samples:
        raise ValueError("Expected a padded training window")
    if not np.any(audio):
        return np.zeros(
            (len(config.harmonics), config.n_frames, config.n_pitch_bins), np.float32
        )
    bins_per_octave = 12 * config.bins_per_semitone
    fmin = float(librosa.midi_to_hz(config.min_midi)) / 2
    # Leave bandwidth for the highest CQT filter below Nyquist.
    n_bins = (
        int(np.floor(bins_per_octave * np.log2(config.sample_rate / (2 * fmin)))) - 1
    )
    # Centered CQT deliberately zero-pads the lowest octaves of short windows.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="n_fft=.*too large.*",
            category=UserWarning,
            module="librosa.core.spectrum",
        )
        magnitude = np.abs(
            librosa.cqt(
                audio,
                sr=config.sample_rate,
                hop_length=config.hop_length,
                fmin=fmin,
                n_bins=n_bins,
                bins_per_octave=bins_per_octave,
                window="hann",
                pad_mode="constant",
            )
        )
    db = librosa.amplitude_to_db(magnitude, ref=np.max, top_db=80)
    normalized = np.clip((db + 80) / 80, 0, 1).astype(np.float32)
    channels = []
    for harmonic in config.harmonics:
        shift = bins_per_octave + round(bins_per_octave * np.log2(harmonic))
        channel = np.zeros((config.n_pitch_bins, config.n_frames), dtype=np.float32)
        width = min(config.n_pitch_bins, n_bins - shift)
        channel[:width] = normalized[shift : shift + width, : config.n_frames]
        channels.append(channel.T)
    return np.stack(channels)


def labels_for_notes(notes, start_sample, mask, config=FEATURES):
    """Rasterize overlaps; notes already sounding at crop start get no new onset."""
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (config.n_frames,):
        raise ValueError("Invalid frame mask")
    note_grid = np.zeros((config.n_frames, config.n_notes), dtype=np.float32)
    onset_grid = np.zeros_like(note_grid)
    pitch_grid = np.zeros((config.n_frames, config.n_pitch_bins), dtype=np.float32)
    start_time = start_sample / config.sample_rate
    end_time = start_time + config.window_seconds
    for note in notes:
        index = note.pitch - config.min_midi
        if (
            not 0 <= index < config.n_notes
            or note.end <= start_time
            or note.start >= end_time
        ):
            continue
        first = int(np.ceil((note.start - start_time) / config.frame_seconds - 1e-8))
        last = int(np.ceil((note.end - start_time) / config.frame_seconds - 1e-8))
        first = min(max(first, 0), config.n_frames - 1)
        last = min(max(last, first + 1), config.n_frames)
        note_grid[first:last, index] = 1
        # MAESTRO supplies notes: use their centers as pitch-contour proxies.
        pitch_grid[first:last, index * config.bins_per_semitone] = 1
        if note.start >= start_time:
            onset_grid[first, index] = 1
    return {
        "onset": onset_grid * mask[:, None],
        "pitch": pitch_grid * mask[:, None],
        "note": note_grid * mask[:, None],
        "mask": mask,
    }
