"""Self-contained audio and piano-roll inspection report."""

import base64
from html import escape
import io
import textwrap
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import soundfile as sf
from audio_features import FEATURES
from note_creator import notes_to_midi


def wav_url(audio):
    buffer = io.BytesIO()
    sf.write(
        buffer,
        np.asarray(audio, dtype=np.float32),
        FEATURES.sample_rate,
        format="WAV",
        subtype="PCM_16",
    )
    return "data:audio/wav;base64," + base64.b64encode(buffer.getvalue()).decode()


def create_report(
    output_dir,
    audio,
    predicted,
    reference=None,
    metrics=None,
    title="Transcription inspection",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    duration = len(audio) / FEATURES.sample_rate
    midi = notes_to_midi(predicted, output_dir / "prediction.mid", duration)
    synthesized = midi.synthesize(fs=FEATURES.sample_rate)
    synthesized = np.pad(synthesized, (0, max(0, len(audio) - len(synthesized))))[
        : len(audio)
    ]
    if len(synthesized) and np.max(np.abs(synthesized)) > 0:
        synthesized = 0.8 * synthesized / np.max(np.abs(synthesized))
    sf.write(output_dir / "original.wav", audio, FEATURES.sample_rate)
    sf.write(output_dir / "prediction.wav", synthesized, FEATURES.sample_rate)
    rolls = [("Predicted notes", predicted, "#137c66")]
    if reference is not None:
        notes_to_midi(reference, output_dir / "reference.mid", duration)
        rolls.insert(0, ("Reference MIDI", reference, "#60738c"))
    fig, axes = plt.subplots(
        len(rolls) + 1,
        1,
        figsize=(12, 2.2 + 2.6 * len(rolls)),
        sharex=True,
        gridspec_kw={"height_ratios": [0.6] + [1] * len(rolls)},
    )
    times = np.arange(len(audio)) / FEATURES.sample_rate
    axes[0].plot(times[::16], audio[::16], color="#536171", linewidth=0.5)
    axes[0].set_ylabel("Audio")
    all_notes = predicted + (reference or [])
    low = max(21, min((n[2] for n in all_notes), default=48) - 4)
    high = min(108, max((n[2] for n in all_notes), default=84) + 4)
    for ax, (label, notes, color) in zip(axes[1:], rolls):
        for start, end, pitch in notes:
            ax.add_patch(
                Rectangle(
                    (start, pitch - 0.4), end - start, 0.8, facecolor=color, alpha=0.85
                )
            )
        ax.set_ylim(low - 0.5, high + 0.5)
        ticks = [p for p in range(24, 109, 12) if low <= p <= high]
        ax.set_yticks(ticks, [f"C{p // 12 - 1}" for p in ticks])
        ax.set_ylabel(label)
        ax.grid(alpha=0.18)
    axes[-1].set_xlabel("Seconds")
    axes[-1].set_xlim(0, max(duration, 0.01))
    fig.suptitle(textwrap.fill(title, 90), fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "piano_roll.png", dpi=140)
    plt.close(fig)
    image = (
        "data:image/png;base64,"
        + base64.b64encode((output_dir / "piano_roll.png").read_bytes()).decode()
    )
    score = (
        ""
        if metrics is None
        else (
            f'<p class="score">Note onset F1 <strong>{metrics["f1"]:.1%}</strong> · '
            f"{metrics['matched']} matched · {metrics['predicted']} predicted · {metrics['reference']} reference</p>"
        )
    )
    reference_info = (
        "Reference MIDI is unavailable for this audio."
        if reference is None
        else (
            "The reference uses the same audio excerpt and clips notes at its boundaries."
        )
    )
    html = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)}</title><style>
body{{font-family:system-ui,sans-serif;background:#f4f6f8;color:#1b2937;margin:0}}
main{{max-width:1120px;margin:40px auto;padding:32px;background:white;border-radius:16px}}
h1{{font-size:26px;margin:0 0 8px;overflow-wrap:anywhere}}p{{line-height:1.5;color:#546474}}
.players{{display:flex;gap:24px;flex-wrap:wrap;margin:24px 0}}.players section{{flex:1;min-width:280px}}
audio{{width:100%}}img{{width:100%;height:auto}}.score{{background:#eef7f3;padding:14px;border-radius:8px}}
a{{color:#137c66}}small{{display:block;color:#657281;margin-top:16px}}
</style><main><h1>{escape(title)}</h1><p>{duration:.2f}-second excerpt · piano transcription</p>
{score}<div class="players"><section><p>Original recording</p><audio controls src="{wav_url(audio)}"></audio></section>
<section><p>Predicted MIDI — simple tone synthesis</p><audio controls src="{wav_url(synthesized)}"></audio></section></div>
<img alt="Audio waveform and predicted/reference piano rolls" src="{image}">
<p>{reference_info}</p><a href="prediction.mid" download>Download predicted MIDI</a>
<small>Two-second model windows are combined on their shared time grid. This model is trained on piano;
pitch bends and pedal reconstruction are outside this demo.</small></main></html>'''
    path = output_dir / "index.html"
    path.write_text(html)
    return path
