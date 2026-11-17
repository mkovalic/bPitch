"""Serve the piano demo and transcribe short WAV files on this computer."""

import base64
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import tempfile
import threading

import librosa
import numpy as np
import soundfile as sf
import torch

from audio_features import FEATURES
from demo_visuals import visualization
from inspection_report import synthesize_midi, wav_url
from model import BasicPitchModel
from note_creator import NoteCreator, notes_to_midi
from project_config import load_checkpoint
from transcribe import transcribe_samples

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs" / "expanded"
VIEWER = RUN / "viewer"
PORT = 8766
MAX_BYTES = 32 * 1024 * 1024


def read_audio(body):
    try:
        with sf.SoundFile(BytesIO(body)) as source:
            if source.format not in {"WAV", "WAVEX"}:
                raise ValueError("Choose a WAV recording.")
            if source.frames <= 0 or source.frames > source.samplerate * 60:
                raise ValueError("Choose a recording between 0 and 60 seconds long.")
            if source.channels > 2 or not 8000 <= source.samplerate <= 192000:
                raise ValueError("Choose a mono or stereo WAV from 8 to 192 kHz.")
            samples = source.read(dtype="float32", always_2d=True).mean(axis=1)
            if not np.isfinite(samples).all():
                raise ValueError("Choose a WAV containing finite audio samples.")
            return librosa.resample(samples, orig_sr=source.samplerate,
                                    target_sr=FEATURES.sample_rate)
    except sf.LibsndfileError as exc:
        raise ValueError("Could not read this WAV file.") from exc


class Handler(SimpleHTTPRequestHandler):
    def reply(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status, value):
        self.reply(status, json.dumps(value).encode(), "application/json")

    def do_GET(self):
        if self.path.split("?")[0] in {"/", "/index.html"}:
            page = (VIEWER / "index.html").read_bytes()
            self.reply(200, page.replace(b'data-inference="off"', b'data-inference="on"', 1),
                       "text/html; charset=utf-8")
        else:
            super().do_GET()

    def do_POST(self):
        address = f"127.0.0.1:{PORT}"
        if self.path != "/transcribe":
            return self.send_error(404)
        if self.headers.get("Host") != address or self.headers.get("Origin") != "http://" + address:
            return self.send_json(403, {"error": "Open the demo on this computer."})
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            size = 0
        if not 0 < size <= MAX_BYTES:
            return self.send_json(413, {"error": "Choose a WAV file under 32 MB."})
        if not self.server.inference_lock.acquire(blocking=False):
            return self.send_json(429, {"error": "A transcription is running. Try again shortly."})
        try:
            audio = read_audio(self.rfile.read(size))
            notes = transcribe_samples(self.server.model, self.server.decoder, audio, "cpu", 4)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "output.mid"
                midi = notes_to_midi(notes, path)
                encoded = base64.b64encode(path.read_bytes()).decode()
            synthesized = synthesize_midi(midi, len(audio))
            result = {"midi": "data:audio/midi;base64," + encoded,
                      "audio": wav_url(synthesized), "notes": len(notes),
                      "visualization": visualization(audio, FEATURES.sample_rate, notes),
                      "prediction_visualization": visualization(synthesized, FEATURES.sample_rate, notes)}
        except ValueError as exc:
            return self.send_json(400, {"error": str(exc)})
        finally:
            self.server.inference_lock.release()
        self.send_json(200, result)


if __name__ == "__main__":
    if not (VIEWER / "index.html").exists():
        raise SystemExit("Generate the examples first with build_demo.py.")
    torch.set_num_threads(4)
    checkpoint = load_checkpoint(RUN / "best.pt")
    model = BasicPitchModel()
    model.load_state_dict(checkpoint["model"])
    thresholds = json.loads((RUN / "decoder.json").read_text())["thresholds"]
    with ThreadingHTTPServer(("127.0.0.1", PORT), partial(Handler, directory=str(VIEWER))) as server:
        server.model = model.eval()
        server.decoder = NoteCreator(**thresholds)
        server.inference_lock = threading.Lock()
        print(f"Piano to MIDI: http://127.0.0.1:{PORT}/", flush=True)
        server.serve_forever()
