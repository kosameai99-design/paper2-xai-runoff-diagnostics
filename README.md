# Model-Consistent XAI Diagnostics for Runoff Prediction

This repository contains version 1.0.0 of the analysis software supporting model-consistent Integrated Gradients diagnostics for CudaLSTM runoff predictions across 135 Japanese basins.

## What is included

- Epoch-explicit Integrated Gradients runner.
- Group A-F attribution aggregation and preprocessing scripts.
- Closure, random-seed stability, ablation, and keep-only response checks.
- Reproduction scripts for Figures 3-5 and summary tables.
- Parameterized builders for geospatial Figures 1, 2, and 6 and Supporting Information figures.
- CudaLSTM and EA-LSTM sensitivity configurations.

Model checkpoints, derived attribution tables, validation results, and rendered figures are not included in this software repository. A citable companion archive of derived results will be released separately.

## Reproduction levels

Figures 3-5 and the compact validation summaries can be reproduced after obtaining and extracting the companion derived-results package. Figures 1, 2, and 6 additionally require the provider-controlled basin and geospatial inputs listed in [`docs/DATA_ACCESS.md`](docs/DATA_ACCESS.md). Full model retraining requires the original provider-controlled observations.

## Environment

```bash
conda env create -f environment.yml
conda activate runoff-xai-diagnostics
```

## Verify the release

```bash
python code/release_tools/verify_release.py --code-only
python code/release_tools/verify_release.py --data-root /path/to/extracted_dataset
```

## Canonical lineage

- Primary model: NeuralHydrology CudaLSTM, seed 595126.
- Reported test metrics and Integrated Gradients analyses use Epoch 20.
- Epoch 18 has the minimum validation loss but was not used for the reported test/IG pipeline.
- Historical test IG was saved under `epoch_000` because of a filename-parsing bug; the run log confirms that `model_epoch020.pt` was loaded.

## License

Code is licensed under the BSD 3-Clause License. The companion derived-results archive uses a separate data license. Third-party data remain subject to their providers' terms.
