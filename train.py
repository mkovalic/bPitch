"""Train the corrected model with held-out validation and resumable checkpoints."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time
import shutil
import torch
from torch.utils.data import DataLoader
from audio_features import FEATURES
from MaestroDataset import MaestroDataset
from model import BasicPitchModel, compute_loss_components
from project_config import (
    choose_device,
    load_checkpoint,
    manifest_digest,
    read_config,
    seed_everything,
)


def make_loader(dataset, config, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=shuffle,
        num_workers=config["num_workers"],
        persistent_workers=config["num_workers"] > 0,
        generator=torch.Generator().manual_seed(config["seed"]),
    )


def run_epoch(
    model, loader, device, optimizer=None, max_batches=None, observer=None, start_step=0
):
    model.train(optimizer is not None)
    totals = {key: 0.0 for key in ("total", "onset", "pitch", "note")}
    examples, batches, wait_seconds, gradient_norm_sum = 0, 0, 0.0, 0.0
    started = time.perf_counter()
    previous = started
    with torch.set_grad_enabled(optimizer is not None):
        for step, (inputs, labels) in enumerate(loader):
            ready = time.perf_counter()
            wait = ready - previous
            wait_seconds += wait
            inputs = inputs.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}
            if optimizer:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs, return_logits=True)
            components = compute_loss_components(outputs, labels)
            loss = sum(components.values())
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; training stopped")
            if optimizer:
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 5.0, error_if_nonfinite=True
                )
                optimizer.step()
                gradient_norm_sum += float(gradient_norm)
            values = dict(
                zip(
                    totals,
                    torch.stack([loss, *components.values()]).detach().cpu().tolist(),
                )
            )
            for key, value in values.items():
                totals[key] += value * len(inputs)
            examples += len(inputs)
            batches += 1
            processing_seconds = time.perf_counter() - ready
            if observer and optimizer:
                observer.log_batch(
                    start_step + batches,
                    batches,
                    values,
                    float(gradient_norm),
                    optimizer.param_groups[0]["lr"],
                    wait,
                    processing_seconds,
                )
            if optimizer and (step == 0 or (step + 1) % 50 == 0):
                print(
                    f"  batch {step + 1}/{len(loader)} loss={values['total']:.5f} elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
            if max_batches and step + 1 >= max_batches:
                break
            previous = time.perf_counter()
    if not examples:
        raise RuntimeError("No batches processed")
    elapsed = time.perf_counter() - started
    return {
        "loss": totals["total"] / examples,
        "loss_components": {
            key: totals[key] / examples for key in ("onset", "pitch", "note")
        },
        "batches": batches,
        "seconds": elapsed,
        "loader_wait_seconds": wait_seconds,
        "examples_per_second": examples / elapsed,
        "gradient_norm": gradient_norm_sum / batches if optimizer else None,
    }


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    best_loss,
    config,
    history,
    bad_epochs,
    global_step=None,
):
    payload = {
        "model_version": model.version,
        "features": FEATURES.to_dict(),
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_loss,
        "config": config,
        "manifest_sha256": manifest_digest(config["manifest"]),
        "manifest": json.loads(Path(config["manifest"]).read_text()),
        "history": history,
        "bad_epochs": bad_epochs,
    }
    if global_step is not None:
        payload["global_step"] = global_step
    temporary = Path(str(path) + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--run-dir")
    parser.add_argument("--resume")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable event logging and validation previews",
    )
    args = parser.parse_args()
    config = read_config(args.config)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.run_dir:
        config["run_dir"] = str(Path(args.run_dir).resolve())
    elif args.resume:
        config["run_dir"] = str(Path(args.resume).resolve().parent)
    if args.workers is not None:
        config["num_workers"] = args.workers
    if args.no_tensorboard:
        config["observability"]["enabled"] = False
    seed_everything(config["seed"])
    torch.set_num_threads(4)
    device = choose_device(args.device)
    run_dir = Path(config["run_dir"])
    if (run_dir / "last.pt").exists() and not args.resume:
        raise FileExistsError(
            f"{run_dir} already contains a run. Use --resume or a new --run-dir."
        )
    train = MaestroDataset(
        config["data_dir"], config["manifest"], "train", config["cache_dir"]
    )
    validation = MaestroDataset(
        config["data_dir"], config["manifest"], "validation", config["cache_dir"]
    )
    train_loader, val_loader = (
        make_loader(train, config, True),
        make_loader(validation, config),
    )
    model = BasicPitchModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    first_epoch, best_loss, history, bad_epochs, global_step = 0, float("inf"), [], 0, 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        if checkpoint["manifest_sha256"] != manifest_digest(config["manifest"]):
            raise ValueError("Resume manifest differs from checkpoint")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        first_epoch, best_loss = checkpoint["epoch"] + 1, checkpoint["best_val_loss"]
        history, bad_epochs = checkpoint["history"], checkpoint["bad_epochs"]
        global_step = checkpoint.get("global_step", first_epoch * len(train_loader))
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.resume and not (run_dir / "best.pt").exists():
        prior_best = Path(args.resume).resolve().parent / "best.pt"
        if not prior_best.exists():
            raise FileNotFoundError(
                "Resuming into a new directory requires the prior best.pt alongside the checkpoint"
            )
        shutil.copy2(prior_best, run_dir / "best.pt")
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(
        f"Device={device}; parameters={sum(p.numel() for p in model.parameters())}; train={len(train)} validation={len(validation)}",
        flush=True,
    )
    observer_context = nullcontext(None)
    if config["observability"]["enabled"]:
        from observability import TrainingObserver

        observer_context = TrainingObserver(
            run_dir, config, device, global_step if args.resume else None
        )
        print(f"TensorBoard events: {run_dir / 'tensorboard'}", flush=True)
    with observer_context as observer:
        for epoch in range(first_epoch, config["epochs"]):
            seed_everything(config["seed"] + epoch)
            train_loader.generator.manual_seed(config["seed"] + epoch)
            print(f"Epoch {epoch + 1}/{config['epochs']}", flush=True)
            training = run_epoch(
                model,
                train_loader,
                device,
                optimizer,
                args.max_train_batches,
                observer,
                global_step,
            )
            validation_stats = run_epoch(
                model, val_loader, device, max_batches=args.max_val_batches
            )
            global_step += training["batches"]
            train_loss, val_loss = training["loss"], validation_stats["loss"]
            improved = val_loss < best_loss - 1e-4
            if improved:
                best_loss, bad_epochs = val_loss, 0
            else:
                bad_epochs += 1
            row = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "gradient_norm": training["gradient_norm"],
                "best_validation_loss": best_loss,
                "bad_epochs": bad_epochs,
            }
            for split, stats in (("train", training), ("validation", validation_stats)):
                for key in (
                    "loss_components",
                    "seconds",
                    "loader_wait_seconds",
                    "examples_per_second",
                ):
                    row[f"{split}_{key}"] = stats[key]
            if observer:
                row["validation_preview"] = observer.validation_preview(
                    model, validation, device, epoch + 1, global_step
                )
                observer.log_epoch(row)
            history.append(row)
            if improved:
                save_checkpoint(
                    run_dir / "best.pt",
                    model,
                    optimizer,
                    epoch,
                    best_loss,
                    config,
                    history,
                    bad_epochs,
                    global_step,
                )
            save_checkpoint(
                run_dir / "last.pt",
                model,
                optimizer,
                epoch,
                best_loss,
                config,
                history,
                bad_epochs,
                global_step,
            )
            (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
            print(
                f"  train={train_loss:.5f} validation={val_loss:.5f} best={best_loss:.5f}",
                flush=True,
            )
            if observer:
                print(
                    f"  validation preview ({row['validation_preview']['windows']} windows): note F1={row['validation_preview']['note_events']['f1']:.4f}",
                    flush=True,
                )
            if bad_epochs >= config["patience"]:
                print("Early stopping: validation loss stopped improving.", flush=True)
                break
    print(f"Checkpoint: {run_dir / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
