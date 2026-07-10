#!/usr/bin/env python3
"""Audit historical and deterministic IG behavior for the Paper 2 CudaLSTM.

The historical evaluation code switched the model to training mode so cuDNN
would permit recurrent backward passes. Because the trained model has output
dropout, this audit compares repeated historical attributions with an eval-mode
implementation that disables cuDNN only while Captum evaluates the path.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients
from scipy.stats import spearmanr
from torch.utils.data import DataLoader


GROUPS = {
    "Precipitation": ["hist_pr"],
    "Air temperature": ["hist_tas"],
    "Snow physics": ["sd", "snowc", "sf"],
    "Radiation": ["ssr"],
    "Soil temperature": ["stl1"],
    "Soil water": ["swvl1", "swvl_deep"],
    "Wind": ["wind_speed"],
    "Potential evapotranspiration": ["PET"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nh-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--n-steps", type=int, default=20)
    parser.add_argument("--basins", nargs="+", default=["001", "107", "123"])
    parser.add_argument("--historical-repeats", type=int, default=3)
    parser.add_argument("--deterministic-repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=595126)
    return parser.parse_args()


def configure_imports(nh_root: Path) -> None:
    sys.path.insert(0, str(nh_root.resolve()))


def build_tester(args: argparse.Namespace):
    from neuralhydrology.evaluation.tester import RegressionTester
    from neuralhydrology.utils.config import Config

    cfg = Config(args.run_dir / "config.yml")
    basin_file = args.nh_root / "data" / "japan_set.txt"
    cfg.update_config(
        {
            "data_dir": str(args.nh_root / "data" / "sawada"),
            "train_basin_file": str(basin_file),
            "validation_basin_file": str(basin_file),
            "test_basin_file": str(basin_file),
            "device": "cpu",
            "batch_size": 1,
            "num_workers": 0,
            "verbose": 0,
        },
        dev_mode=True,
    )
    tester = RegressionTester(cfg=cfg, run_dir=args.run_dir, period="test", init_model=True)
    tester._load_weights(epoch=args.epoch)
    return cfg, tester


def move_and_prepare_batch(tester, dataset, sample_index: int) -> dict:
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=[sample_index],
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )
    data = next(iter(loader))
    for key in data:
        if key.startswith("x_d"):
            data[key] = {name: value.to(tester.device) for name, value in data[key].items()}
        elif not key.startswith("date"):
            data[key] = data[key].to(tester.device)
    return tester.model.pre_model_hook(data, is_train=False)


def make_forward(model, cfg, data):
    static = data.get("x_s")

    def forward(x_dynamic: torch.Tensor) -> torch.Tensor:
        dynamic = {
            variable: x_dynamic[:, :, index : index + 1]
            for index, variable in enumerate(cfg.dynamic_inputs)
        }
        model_input = {"x_d": dynamic}
        if static is not None:
            outer_batch = static.shape[0]
            forward_batch = x_dynamic.shape[0]
            if outer_batch and forward_batch != outer_batch and forward_batch % outer_batch == 0:
                model_input["x_s"] = static.repeat(forward_batch // outer_batch, 1)
            else:
                model_input["x_s"] = static
        return model(model_input)["y_hat"][:, -1, 0]

    return forward


def group_importance(attr: np.ndarray, variables: list[str]) -> dict[str, float]:
    variable_index = {name: idx for idx, name in enumerate(variables)}
    grouped = {}
    for group, members in GROUPS.items():
        indices = [variable_index[name] for name in members]
        grouped[group] = float(np.abs(attr[:, indices]).sum())
    return grouped


def normalized(values: dict[str, float]) -> dict[str, float]:
    total = sum(values.values())
    if total <= 0:
        return {key: math.nan for key in values}
    return {key: value / total for key, value in values.items()}


def run_ig(model, forward, dynamic, mode: str, seed: int, n_steps: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if mode == "historical_train":
        model.train()
        context = nullcontext()
    elif mode == "deterministic_eval":
        model.eval()
        context = torch.backends.cudnn.flags(enabled=False) if hasattr(torch.backends, "cudnn") else nullcontext()
    else:
        raise ValueError(mode)

    with context:
        attr, convergence_delta = IntegratedGradients(forward).attribute(
            dynamic,
            n_steps=n_steps,
            return_convergence_delta=True,
        )
    return attr.detach().cpu().numpy()[0], float(convergence_delta.detach().cpu().numpy().reshape(-1)[0])


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    result = spearmanr(left, right)
    return float(result.statistic if hasattr(result, "statistic") else result[0])


def summarize(records: pd.DataFrame, group_columns: list[str], args: argparse.Namespace) -> dict:
    historical = records[records["mode"] == "historical_train"]
    deterministic = records[records["mode"] == "deterministic_eval"]
    sample_keys = ["basin_id", "sample_index"]
    historical_pair_rho = []
    deterministic_repeat_diff = []
    historical_vs_deterministic_rho = []
    historical_vs_deterministic_share_diff = []

    for _, sample in records.groupby(sample_keys):
        hist = sample[sample["mode"] == "historical_train"].sort_values("repeat")
        det = sample[sample["mode"] == "deterministic_eval"].sort_values("repeat")
        for i, j in itertools.combinations(range(len(hist)), 2):
            historical_pair_rho.append(safe_spearman(hist.iloc[i][group_columns], hist.iloc[j][group_columns]))
        if len(det) >= 2:
            deterministic_repeat_diff.append(
                float(np.max(np.abs(det.iloc[0][group_columns].to_numpy() - det.iloc[1][group_columns].to_numpy())))
            )
        hist_mean = hist[group_columns].mean(axis=0).to_numpy()
        det_mean = det[group_columns].mean(axis=0).to_numpy()
        historical_vs_deterministic_rho.append(safe_spearman(hist_mean, det_mean))
        historical_vs_deterministic_share_diff.append(float(np.max(np.abs(hist_mean - det_mean))))

    hist_global = historical.groupby("repeat")[group_columns].mean()
    det_global = deterministic.groupby("repeat")[group_columns].mean()
    pooled_rho = safe_spearman(hist_global.mean(axis=0).to_numpy(), det_global.mean(axis=0).to_numpy())
    pooled_max_share_diff = float(
        np.max(np.abs(hist_global.mean(axis=0).to_numpy() - det_global.mean(axis=0).to_numpy()))
    )
    deterministic_closure = deterministic["convergence_delta_abs"].median()

    summary = {
        "epoch": args.epoch,
        "n_steps": args.n_steps,
        "n_samples": int(records.groupby(sample_keys).ngroups),
        "historical_pairwise_spearman_median": float(np.nanmedian(historical_pair_rho)),
        "deterministic_repeat_max_abs_share_diff": float(np.nanmax(deterministic_repeat_diff)),
        "historical_vs_deterministic_sample_spearman_median": float(
            np.nanmedian(historical_vs_deterministic_rho)
        ),
        "historical_vs_deterministic_sample_max_share_diff": float(
            np.nanmax(historical_vs_deterministic_share_diff)
        ),
        "historical_vs_deterministic_pooled_spearman": pooled_rho,
        "historical_vs_deterministic_pooled_max_share_diff": pooled_max_share_diff,
        "historical_convergence_delta_abs_median": float(historical["convergence_delta_abs"].median()),
        "deterministic_convergence_delta_abs_median": float(deterministic_closure),
    }
    summary["thresholds"] = {
        "pooled_spearman_min": 0.95,
        "pooled_max_share_diff_max": 0.03,
        "deterministic_repeat_max_abs_share_diff_max": 1e-6,
        "deterministic_convergence_delta_abs_median_max": 0.05,
    }
    summary["release_ready"] = bool(
        pooled_rho >= 0.95
        and pooled_max_share_diff <= 0.03
        and summary["deterministic_repeat_max_abs_share_diff"] <= 1e-6
        and deterministic_closure <= 0.05
    )
    return summary


def main() -> int:
    args = parse_args()
    configure_imports(args.nh_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg, tester = build_tester(args)
    model = tester.model
    records = []

    for basin_id in args.basins:
        dataset = tester._get_dataset(str(basin_id).zfill(3))
        indices = sorted({0, len(dataset) // 2, len(dataset) - 1})
        for sample_index in indices:
            data = move_and_prepare_batch(tester, dataset, sample_index)
            dynamic = torch.stack([value.squeeze(-1) for value in data["x_d"].values()], dim=-1)
            dynamic = dynamic.to(tester.device).requires_grad_(True)
            forward = make_forward(model, cfg, data)
            date_value = data.get("date")
            date_text = str(np.asarray(date_value).reshape(-1)[-1]) if date_value is not None else ""

            modes = [
                ("historical_train", args.historical_repeats),
                ("deterministic_eval", args.deterministic_repeats),
            ]
            for mode, repeats in modes:
                for repeat in range(repeats):
                    attr, delta = run_ig(
                        model=model,
                        forward=forward,
                        dynamic=dynamic,
                        mode=mode,
                        seed=args.seed + repeat,
                        n_steps=args.n_steps,
                    )
                    shares = normalized(group_importance(attr, list(cfg.dynamic_inputs)))
                    records.append(
                        {
                            "basin_id": str(basin_id).zfill(3),
                            "sample_index": sample_index,
                            "date": date_text,
                            "mode": mode,
                            "repeat": repeat,
                            "seed": args.seed + repeat,
                            "convergence_delta": delta,
                            "convergence_delta_abs": abs(delta),
                            **shares,
                        }
                    )

    records_df = pd.DataFrame(records)
    group_columns = list(GROUPS)
    summary = summarize(records_df, group_columns, args)
    records_df.to_csv(args.output_dir / "ig_determinism_sample_results.csv", index=False)
    (args.output_dir / "ig_determinism_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["release_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

