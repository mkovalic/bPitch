"""Check real selected windows against the shared model/label contract."""

import argparse
import torch
from MaestroDataset import MaestroDataset
from model import BasicPitchModel, compute_loss
from project_config import read_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--windows", type=int, default=2)
    args = parser.parse_args()
    config = read_config(args.config)
    model = BasicPitchModel().eval()
    torch.set_num_threads(2)
    for split in ("train", "validation", "test"):
        dataset = MaestroDataset(
            config["data_dir"], config["manifest"], split, config["cache_dir"]
        )
        for index in range(min(args.windows, len(dataset))):
            audio, labels = dataset[index]
            if not torch.all(labels["onset"] <= labels["note"]):
                raise AssertionError("Onset without note activity")
            targets = {k: v[None] for k, v in labels.items()}
            with torch.inference_mode():
                outputs = model(audio[None], return_logits=True)
                loss = compute_loss(outputs, targets)
            print(
                f"{split}[{index}]: audio={tuple(audio.shape)} note={tuple(labels['note'].shape)} "
                f"onsets={int(labels['onset'].sum())} valid={int(labels['mask'].sum())} loss={loss:.4f}"
            )
    print("All selected windows satisfy the contract.")


if __name__ == "__main__":
    main()
