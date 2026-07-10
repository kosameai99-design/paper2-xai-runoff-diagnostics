#!/usr/bin/env python3
"""Generate dynamic-input IG for a trained NeuralHydrology run."""

from __future__ import annotations

import argparse
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from captum.attr import IntegratedGradients
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nh-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--basin-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--period", choices=["validation", "test"], default="test")
    parser.add_argument("--n-steps", type=int, default=20)
    parser.add_argument("--mode", choices=["deterministic", "historical"], default="deterministic")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=595126)
    parser.add_argument("--basins", nargs="*")
    return parser.parse_args()


def model_forward_factory(model, dynamic_inputs: list[str], static: torch.Tensor | None):
    def forward(x_dynamic: torch.Tensor) -> torch.Tensor:
        data = {
            "x_d": {
                name: x_dynamic[:, :, index : index + 1]
                for index, name in enumerate(dynamic_inputs)
            }
        }
        if static is not None:
            source_batch = static.shape[0]
            target_batch = x_dynamic.shape[0]
            if source_batch and target_batch != source_batch and target_batch % source_batch == 0:
                data["x_s"] = static.repeat(target_batch // source_batch, 1)
            else:
                data["x_s"] = static
        return model(data)["y_hat"][:, -1, 0]

    return forward


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(args.nh_root.resolve()))
    from neuralhydrology.evaluation.tester import RegressionTester
    from neuralhydrology.utils.config import Config

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = Config(args.run_dir / "config.yml")
    cfg.update_config(
        {
            "data_dir": str(args.data_dir),
            "train_basin_file": str(args.basin_file),
            "validation_basin_file": str(args.basin_file),
            "test_basin_file": str(args.basin_file),
            "device": args.device,
            "num_workers": 0,
            "verbose": 0,
        },
        dev_mode=True,
    )
    tester = RegressionTester(cfg, args.run_dir, period=args.period, init_model=True)
    tester._load_weights(epoch=args.epoch)
    model = tester.model
    basins = [str(value).zfill(3) for value in (args.basins or tester.basins)]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for basin in basins:
        dataset = tester._get_dataset(basin)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=0, collate_fn=dataset.collate_fn)
        for batch_index, data in enumerate(loader):
            for key in data:
                if key.startswith("x_d"):
                    data[key] = {name: value.to(tester.device) for name, value in data[key].items()}
                elif not key.startswith("date"):
                    data[key] = data[key].to(tester.device)
            data = model.pre_model_hook(data, is_train=False)
            x_dynamic = torch.stack([value.squeeze(-1) for value in data["x_d"].values()], dim=-1)
            x_dynamic = x_dynamic.to(tester.device).requires_grad_(True)
            forward = model_forward_factory(model, list(cfg.dynamic_inputs), data.get("x_s"))

            if args.mode == "historical":
                model.train()
                context = nullcontext()
            else:
                model.eval()
                context = torch.backends.cudnn.flags(enabled=False) if hasattr(torch.backends, "cudnn") else nullcontext()
            with context:
                attr = IntegratedGradients(forward).attribute(x_dynamic, n_steps=args.n_steps)
            np.save(args.output_dir / f"{basin}_batch{batch_index}.npy", attr.detach().cpu().numpy())
        print(f"{basin}: {len(loader)} batches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

