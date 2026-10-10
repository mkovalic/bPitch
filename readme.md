# torchPitch — piano audio to MIDI

A PyTorch piano adaptation of Spotify’s [Basic Pitch paper](https://arxiv.org/abs/2203.09893) and [released architecture](https://github.com/spotify/basic-pitch/blob/main/basic_pitch/models.py).

The corrected version-2 pipeline uses a shared audio/label time grid, manifest-based dataset splits, cached features, validation-based checkpoints, note-event evaluation, and MIDI export. Legacy checkpoints such as `basic_pitch_epoch_1.pth` use a different architecture and training targets, so version 2 requires a fresh training run.

The expanded run reaches **68.81% note F1** on 160 fixed two-second held-out windows, compared with **62.09%** for the initial corrected run. Including note endings gives **31.57% F1**. The results viewer compares both runs and provides four selectable audio/MIDI examples with duration-error diagnostics.

## Setup

Python 3.11 is the tested runtime. The existing environment is `ml_env`.

```sh
python3.11 -m venv ml_env
ml_env/bin/python -m pip install -r requirements.txt
cp config_template.json config.json
```

Set `data_dir` in `config.json` to the extracted [MAESTRO v3 dataset](https://magenta.tensorflow.org/datasets/maestro) directory. Paths resolve relative to the config file. Dataset audio, trained checkpoints, feature caches, and generated reports are local artifacts and are excluded from Git; the manifests and aggregate result summaries are included.

## Data and tensor contract

The reproducible demo manifest selects **50 train / 10 validation / 10 test recordings** from MAESTRO’s official splits, preserving recording and composition separation. With seed 42, it samples **32 / 16 / 16 two-second windows per recording**, distributed throughout the music. This gives **1,600 / 160 / 160 windows**. Evaluation windows are fixed; training windows are a fixed, stratified random sample for this initial demo.

| Representation | Shape |
|---|---|
| Model input | `(batch, 8, 173, 264)` — harmonics, time, pitch |
| Pitch contours | `(batch, 173, 264)` — three bins per semitone |
| Notes and onsets | `(batch, 173, 88)` — one bin per piano key |
| Valid-audio mask | `(batch, 173)` |

Audio is mono at 22,050 Hz with a 256-sample hop. All heads preserve time. Note bin 0 is A0 (MIDI 21); bin 87 is C8 (MIDI 108). Contour bin `3 * (MIDI - 21)` is that note’s center. MAESTRO supplies note labels, so these centers serve as contour proxies; this training set does not provide continuous pitch-bend targets.

Preprocessing computes one CQT, applies an 80 dB log normalization, and shifts it into harmonic channels `[0.5, 1, 2, 3, 4, 5, 6, 7]`. Frequencies above Nyquist are zero-filled. WAV reads seek directly to each requested excerpt; MIDI is parsed once per recording per worker. Versioned, atomic caches store float16 features and binary targets. The same quantized features are returned on cache hits and misses.

Labels include notes crossing either crop boundary, add onsets only for notes beginning inside the window, and mask padded audio. Binary labels are never resized. The 16,780-parameter model follows the released shallow network’s heads, strides, and padding. Fixed preprocessing normalization replaces its trainable input normalization. This is a piano adaptation; broader instrument coverage requires additional training data.

## Verify and prepare

```sh
ml_env/bin/python prepare_data.py
ml_env/bin/python validateDataset.py
ml_env/bin/python -m unittest discover -s tests -v
ml_env/bin/python verifyLoader.py
```

`prepare_data.py --cache` optionally builds all feature caches in advance. Ordinary training fills missing entries automatically. Existing manifests are checked against the requested seed/counts; `--overwrite` explicitly regenerates a changed selection. Regenerating a manifest invalidates resume/evaluation compatibility with earlier runs.

The tests cover known tones from A0 through C8, chords, short notes, crop boundaries, padded frames, shape errors, gradient flow, cache regeneration, MIDI round trips, and checkpoint round trips.

## Overfit gate and training

First verify that two real clips can be memorized:

```sh
ml_env/bin/python overfit.py --steps 2000
```

This command exits unsuccessfully if training-clip note-onset F1 cannot reach 0.85. Its metrics and checkpoint live in `runs/overfit/`. This is a pipeline check using training clips; its score measures memorization.

Then train the demo subset:

```sh
ml_env/bin/python train.py
```

Defaults are Adam, learning rate 0.001, batch size 4, up to 50 epochs, and early stopping after eight epochs without validation-loss improvement. Device selection prefers Apple MPS, then CUDA, then CPU; `--device cpu` overrides it. Onset loss balances positive/negative class means with weights 0.95/0.05. All losses ignore padded frames.

`runs/demo/` contains `best.pt`, `last.pt`, `history.json`, and the resolved config. Checkpoints contain the model/optimizer state, feature version, seed, selected manifest, and manifest digest. They load with PyTorch’s restricted weights-only loader.

```sh
# Continue the same run; --epochs is the total desired epoch count.
ml_env/bin/python train.py --resume runs/demo/last.pt --epochs 75

# Start a new run from scratch without overwriting the demo.
ml_env/bin/python train.py --run-dir runs/retrain

# Fast pipeline smoke check, including validation and checkpoint writing.
ml_env/bin/python train.py --run-dir runs/smoke-new --epochs 1 --workers 0 --max-train-batches 1 --max-val-batches 1
```

`MaestroLoader.py` is a compatibility alias. `earlystop_train.py` delegates to the supported trainer, eliminating the competing loading/training implementations.

## Training monitoring

TensorBoard logging is enabled by default. Start its local dashboard in another terminal:

```sh
ml_env/bin/tensorboard --logdir runs --host 127.0.0.1 --port 6006
```

Open <http://127.0.0.1:6006/>. Each run writes to its own `runs/<name>/tensorboard/` directory, so the dashboard can compare experiments. Event files remain available after training exits. Scalars use optimizer updates as the horizontal axis; `progress/epoch` records the corresponding epoch. Resuming restores this step counter and purges summaries beyond the resumed checkpoint.

The dashboard records:

- Train/validation total loss and separate onset, note, and pitch losses every epoch.
- Sampled batch losses, learning rate, and gradient norm before clipping (first batch and every 25 batches).
- Epoch duration, examples per second, and time waiting for the DataLoader. Wait time measures foreground waiting, including worker startup; it is not total background preprocessing CPU time. Batch processing includes device transfers and model/optimizer work; it is not a GPU kernel profiler.
- Note-event precision/recall/F1, with and without offset matching, on the same 16 evenly spaced validation windows each epoch.
- Reference audio, simple synthesis of predicted MIDI, reference/predicted piano rolls, and confidence histograms for one fixed validation clip at epoch 1 and every five epochs. Preview MIDI files are saved under `runs/<name>/preview/`.
- Configuration, feature coordinates, manifest digest, preview window indices, and decoder settings.

The preview decoder is fixed throughout training at onset 0.995, note 0.3, minimum duration 40 ms, and gap tolerance 11 frames, based on the earlier validation calibration. This small validation preview is a progress indicator; it does not replace full validation/test evaluation. Checkpoint selection still uses loss over all validation windows, and training monitoring never reads the test split. Perform final threshold tuning with `performance_metrics.py --tune` after training.

The `observability` section in `config.json` controls logging interval, preview size, media interval, and fixed monitoring thresholds. `--no-tensorboard` disables event logging and the extra previews. No model or optimizer hyperparameters change when logging is enabled. Epoch metrics are also retained in `history.json` and checkpoints.

To import an older run's saved loss history once:

```sh
ml_env/bin/python observability.py --import-run runs/demo
```

Importing only includes measurements the run saved: the old total-loss curves and any saved test results. Legacy optimizer-step counts are inferred from full epochs; missing head losses, timings, and gradients are not reconstructed.

## Expanded training run

`configs/expanded.json` selects 200 training recordings and 6,400 two-second training windows with the same optimizer, batch size, learning rate, seed, and early-stopping settings as the demo. Its `holdout_manifest` preserves the exact 160 validation and 160 test windows from `manifests/demo.json`, so changing training size does not change the evaluation material. The new manifest records the original holdout manifest's digest; split membership, composition separation, and frozen window bounds are checked during preparation.

```sh
ml_env/bin/python prepare_data.py --config configs/expanded.json
ml_env/bin/python train.py --config configs/expanded.json
```

This starts from fresh weights and writes to `runs/expanded/`, retaining the existing `runs/demo/` checkpoint and results. Missing training features are cached during the first epoch; later epochs reuse them. After training, tune on validation and evaluate the same test clips:

```sh
ml_env/bin/python performance_metrics.py --checkpoint runs/expanded/best.pt --tune
ml_env/bin/python transcribe.py --checkpoint runs/expanded/best.pt --output runs/expanded/inspection
```

## Tune and evaluate

```sh
ml_env/bin/python performance_metrics.py --checkpoint runs/demo/best.pt --tune
```

Thresholds are selected using the 160 validation windows, then frozen for the 160 test windows. Results are written to `decoder.json` and `test_metrics.json`. Note-event precision/recall/F1 use `mir_eval` matching: 50 ms onset tolerance and 50-cent pitch tolerance, with and without the 20% duration offset tolerance. Counts aggregate across examples without broadcasting between them. Frame scores are also reported.

These demo metrics cover selected two-second held-out windows. Reference note events are clipped to each window; full-recording benchmark evaluation is a separate scope. The test split is used for reporting, not checkpoint selection or threshold tuning.

## MIDI and audio inspection

```sh
# An eight-second excerpt from the first held-out recording.
ml_env/bin/python transcribe.py --checkpoint runs/demo/best.pt

# A user-supplied recording, optionally paired with reference MIDI.
ml_env/bin/python transcribe.py --checkpoint runs/demo/best.pt --audio example.wav --reference example.mid --output runs/my-example

# Reproduce the two-second training example used by the overfit check.
ml_env/bin/python transcribe.py --checkpoint runs/overfit/best.pt --example train --duration 2 --output runs/overfit/inspection
```

Inference combines overlapping model windows on the same frame grid, decodes note events, and exports MIDI with actual MIDI note numbers. The decoder tolerates short probability dropouts and handles repeated notes without mutating its input. Threshold defaults follow the paper; validation selects onset/note confidence, minimum duration, and gap tolerance. The initial duration/gap defaults are 120 ms and 11 frames; the validation search also includes shorter notes for fast piano passages.

Each output directory includes `prediction.mid`, original and synthesized WAV audio, `events.json`, `piano_roll.png`, and a self-contained `index.html` with audio players and predicted/reference rolls. Predicted audio uses simple tone synthesis for inspection. MIDI velocity is fixed; pedal reconstruction and pitch bends are outside this demo.

Open the generated HTML directly, or serve reports locally:

```sh
ml_env/bin/python -m http.server 8765 --bind 127.0.0.1 --directory runs
```

Then visit `http://127.0.0.1:8765/demo/inspection/`.

## Results viewer

Build the expanded model's results viewer after calibration and test evaluation:

```sh
ml_env/bin/python build_demo.py
ml_env/bin/python -m http.server 8765 --bind 127.0.0.1 --directory runs/expanded/viewer
```

Visit `http://127.0.0.1:8765/`. If an older inspection server is using port 8765, stop that server first or choose another port.

The viewer includes a clip selector, original and predicted audio, reference/predicted piano rolls, MIDI downloads, and the previous-versus-expanded test metrics. Four eight-second excerpts illustrate sparse playing, chords, fast playing, and note-ending errors. Texture selection uses reference MIDI only and chooses distinct held-out recordings; the final clip preserves the earlier Rachmaninoff example as a known limitation. The builder verifies that both runs use identical validation/test windows and complete test metrics, then uses the expanded checkpoint's frozen validation decoder.

Clip scores describe eight-second overlapping-window inference and are displayed separately from the aggregate 160 two-second test-window scores. Duration-error bars show the largest ending errors among pitch/onset-matched notes, using the same tolerances as evaluation. `runs/expanded/viewer/results.json` stores the comparison, selection details, clip scores, and diagnostics. The generated viewer runs locally without a JavaScript build step or extra dependencies.

The expanded run stopped after 28 epochs and selected epoch 20. Test note precision/recall/F1 are **59.57% / 81.46% / 68.81%**, with **31.57% F1** including note endings. The baseline note F1 was **62.09%** on the same windows.

Portable aggregate results are saved in [`expanded_results.json`](expanded_results.json) and [`demo_results.json`](demo_results.json). Rebuilding the viewer requires both trained runs and the MAESTRO audio/MIDI files; follow the training and calibration commands above to produce them in a fresh checkout.

## Verified demo run

The initial corrected run used seed 42 and Apple MPS. Early stopping ended training at epoch 18; epoch 10 had the best validation loss (0.123714). The two-clip memorization check reached 89.66% note-onset F1 after 400 updates.

Validation selected onset confidence 0.995, note confidence 0.3, minimum duration 40 ms, and gap tolerance 11 frames. On all 160 held-out test windows, note-event precision/recall/F1 without offset matching were 58.78% / 65.78% / **62.09%**. Including offset matching gave **27.59% F1**. Note endings remain the main quality limitation. The overfit score and test score describe different tasks and should be presented separately.

Detailed artifacts are in `runs/overfit/overfit_metrics.json`, `runs/demo/decoder.json`, `runs/demo/test_metrics.json`, and `runs/demo/inspection/`. The original checkpoint remains untouched. To retrain from scratch, choose a fresh output directory with `--run-dir`; to continue, resume `runs/demo/last.pt`.
