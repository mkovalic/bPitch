"""Gate larger training by memorizing two real training clips."""

import argparse
import json
from pathlib import Path
import time
import torch
from MaestroDataset import MaestroDataset
from model import BasicPitchModel, compute_loss
from performance_metrics import reference_notes, tune_thresholds, score_predictions
from project_config import choose_device, read_config, seed_everything
from train import save_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--target-f1", type=float, default=0.85)
    parser.add_argument("--run-dir", default="runs/overfit")
    args = parser.parse_args()
    config = read_config(args.config)
    seed_everything(config["seed"])
    torch.set_num_threads(4)
    device = choose_device(args.device)
    dataset = MaestroDataset(
        config["data_dir"], config["manifest"], "train", config["cache_dir"]
    )
    indices = []
    samples = []
    # Pick audible clips containing actual note starts.
    for index in range(len(dataset)):
        sample = dataset[index]
        if sample[1]["onset"].sum() >= 2:
            indices.append(index)
            samples.append(sample)
        if len(samples) == 2:
            break
    if len(samples) != 2:
        raise RuntimeError("Need two clips with note starts")
    inputs = torch.stack([s[0] for s in samples]).to(device)
    labels = {
        k: torch.stack([s[1][k] for s in samples]).to(device) for k in samples[0][1]
    }
    model = BasicPitchModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    history = []
    passed = False
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(model(inputs, return_logits=True), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            model.eval()
            with torch.inference_mode():
                outputs = {k: v.cpu().numpy() for k, v in model(inputs).items()}
            examples = [
                {
                    "outputs": {k: v[b] for k, v in outputs.items()},
                    "reference": reference_notes(dataset, index),
                }
                for b, index in enumerate(indices)
            ]
            thresholds, score = tune_thresholds(examples)
            metrics = score_predictions(examples, thresholds)
            history.append({"step": step, "loss": loss.item(), "note_f1": score})
            print(
                f"step={step} loss={loss.item():.5f} training-note-F1={score:.4f} elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
            if score >= args.target_f1:
                passed = True
                break
    result = {
        "passed": passed,
        "target_f1": args.target_f1,
        "indices": indices,
        "steps": step,
        "metrics": metrics,
        "thresholds": thresholds,
        "history": history,
        "scope": "Training clips only; memorization check, not a generalization score.",
    }
    (run_dir / "overfit_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    (run_dir / "decoder.json").write_text(
        json.dumps({"thresholds": thresholds}, indent=2) + "\n"
    )
    save_checkpoint(
        run_dir / "best.pt", model, optimizer, 0, float(loss.item()), config, history, 0
    )
    if not passed:
        raise SystemExit(
            "Overfit gate failed; inspect the pipeline before larger training."
        )
    print(f"Overfit gate passed: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
