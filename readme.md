# IRASD-GST: Interference-Resilient Anomalous Sound Detection based on Semantic Embedding Trees

This repository contains the code, pre-trained checkpoints, and evaluation results for the paper:

**"A Lightweight and Interference-Resilient Sound Anomaly Detection Model for Industrial Machines Based on Semantic Embedding Trees"** (PeerJ Computer Science, manuscript #125199v1, under revision).

The project implements a lightweight (6.9M parameters) two-stage system for industrial machine sound anomaly detection that remains robust under complex background interference (human speech, footsteps, environmental noise). A semantic embedding tree constructed from audio-tag text labels guides a contrastive learning objective, and a CNN–BiLSTM event recognition module produces a temporal mask that suppresses interference frames before anomaly scoring.

## Repository Structure

```
.
├── models.py                        # Model definitions: AudioDataset, EventDetectionModel,
│                                    #   MachineAnalysisModel, and their loss functions
├── semantic_tree_builder.py         # LLM-assisted construction of the semantic tree (Algorithm 1)
├── semantic_tree_embedding.py       # Semantic vector space construction (STCL contrastive training)
├── generate_synthetic_dataset.py    # Synthetic dataset generation with machine-ID-wise
│                                    #   10-fold splits and zero-shot interference protocol
├── train_crossval.py                # 10-fold cross-validation training & evaluation driver
├── evaluate.py                      # Standalone evaluation script for trained checkpoints
├── effect_of_ratio.py               # Interference-ratio (noise intensity) sweep analysis
├── semantic_tree.json               # Final semantic tree ontology used in the paper
├── fold_assignments.json            # Exact machine-ID-wise fold assignments (leakage control)
├── visualize_framework.py           # Figure: overall framework illustration
├── visualize_tree.py                # Figure: semantic tree subset illustration
├── visualize_result.py              # Figure: t-SNE / metric visualizations
├── visualize_dataset_stats.py       # Dataset statistics visualization
├── checkpoints/
│   ├── best_event_model.pth         # Event recognition stage weights (fold 5)
│   └── best_machine_model.pth       # Machine analysis stage weights (fold 5)
├── results/
│   ├── 01_anomaly_auc_by_device.csv
│   ├── 02_anomaly_metrics_by_fold.csv
│   ├── 03_event_metrics_by_fold.csv
│   ├── 04_anomaly_metrics_by_category.csv
│   ├── 05_anomaly_metrics_by_ratio.csv
│   └── overall_statistics.csv
├── DATA_DICTIONARY.md               # Column-level description of all result CSVs
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/hackdong/IRASD-GST.git
cd IRASD-GST
pip install -r requirements.txt
```

Python 3.10+ and PyTorch 2.0 are recommended. All experiments in the paper were run on a GeForce RTX3090 GPU (CUDA 11.7); inference latency was additionally measured on an Intel N100 CPU.

## Data Preparation

The datasets are **not** included in this repository. Please download the public source datasets and organize them as follows:

```
dataset/
├── MIMII/                # MIMII dataset (fan, gearbox, pump, slider, valve)
├── ESC-50-master/        # ESC-50 dataset
└── UrbanSound8K/         # UrbanSound8K dataset
```

Then generate the synthetic corpus (29,700 development + 3,300 held-out test samples, machine-ID-wise 10-fold splits, disjoint noise pools, ~20% unseen interference categories in the test partition):

```bash
python generate_synthetic_dataset.py
```

During synthesis, ambient sounds are mixed with machine sounds with 50% probability, and the overlay ratio (interference gain relative to the machine sound) is randomly drawn from {0.50, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0}.

## Training and Evaluation

```bash
# 10-fold cross-validation training and evaluation
python train_crossval.py

# standalone evaluation of the released checkpoints
python evaluate.py

# interference-ratio sweep analysis
python effect_of_ratio.py
```

`fold_assignments.json` records the exact machine-ID-wise product-ID assignment of every fold so the split protocol can be reproduced or audited.

## Checkpoints

Pre-trained weights from fold 5 of the cross-validation run are released under `checkpoints/`:

- `best_event_model.pth` — interference event recognition stage (~9.8 MB)
- `best_machine_model.pth` — machine analysis / anomaly scoring stage (~17 MB)

## Results

Aggregate results over the 10 folds (mean ± SD), corresponding to the revised manuscript:

| Metric | Value |
|---|---|
| Anomaly detection Precision | 0.954 ± 0.009 |
| Anomaly detection Recall | 0.968 ± 0.009 |
| Anomaly detection F1 | 0.961 ± 0.010 |
| Average AUC / pAUC (Table 1) | 88.71 ± 1.8 / 79.47 ± 2.2 |
| Event recognition F1 / temporal IoU | 0.72 ± 0.02 / 0.77 ± 0.02 |
| Zero-shot interference subset F1 | 0.932 |
| DCASE 2020 Task 2 transfer (AUC / pAUC) | 84.62 ± 2.4 / 78.35 ± 2.9 |
| Parameters / inference time | 6.93M / 0.034 s (RTX3090), 0.28 s (Intel N100) per 10-s clip |

Per-fold, per-device, per-category, and per-ratio breakdowns are provided in `results/`; see `DATA_DICTIONARY.md` for column definitions.

## License

This project is licensed under the terms of the LICENSE file included in this repository.
