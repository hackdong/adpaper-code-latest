# Data Dictionary

Column-level description of every CSV file under `results/`. All metrics are aggregated over the 10-fold machine-ID-wise cross-validation run; `std_*` columns report the standard deviation across folds.

## results/01_anomaly_auc_by_device.csv

| Column | Description |
|---|---|
| `device_type` | Machine type (fan, gearbox, pump, slider, valve) |
| `samples` | Number of held-out test samples of this machine type |
| `auc` | Area under the ROC curve for anomaly detection on this machine type |
| `std_auc` | Standard deviation of AUC across folds |
| `pauc` | Partial AUC at p = 0.1 for this machine type |
| `std_pauc` | Standard deviation of pAUC across folds |
| `overall_mean_auc` / `overall_std_auc` | Mean / SD of AUC across all machine types |
| `overall_mean_pauc` / `overall_std_pauc` | Mean / SD of pAUC across all machine types |

## results/02_anomaly_metrics_by_fold.csv

| Column | Description |
|---|---|
| `fold` | Fold index (0–9) |
| `samples` | Number of held-out test samples in this fold |
| `precision` / `recall` / `f1` | Anomaly detection precision / recall / F1-score of this fold (decision threshold 0.5) |
| `mean_precision` / `std_precision` | Mean / SD of precision across the 10 folds |
| `mean_recall` / `std_recall` | Mean / SD of recall across the 10 folds |
| `mean_f1` / `std_f1` | Mean / SD of F1-score across the 10 folds |

## results/03_event_metrics_by_fold.csv

| Column | Description |
|---|---|
| `fold` | Fold index (0–9) |
| `samples` | Number of test samples with an interference event in this fold |
| `precision` / `recall` / `f1` | Interference event recognition precision / recall / F1-score of this fold |
| `mean_iou` | Mean temporal Intersection-over-Union of the predicted 1D event mask (threshold 0.5) in this fold |
| `mean_precision` / `std_precision` | Mean / SD of event precision across folds |
| `mean_recall` / `std_recall` | Mean / SD of event recall across folds |
| `mean_f1` / `std_f1` | Mean / SD of event F1-score across folds |
| `mean_iou_overall` / `std_iou_overall` | Mean / SD of the temporal IoU across folds |

## results/04_anomaly_metrics_by_category.csv

| Column | Description |
|---|---|
| `overlay_category` | Interference sound category (e.g., dog_bark, footsteps) |
| `samples` | Number of test samples containing this interference category |
| `precision` / `recall` / `f1` | Anomaly detection metrics on samples containing this category |
| `std_precision` / `std_recall` / `std_f1` | Standard deviation across folds |

## results/05_anomaly_metrics_by_ratio.csv

| Column | Description |
|---|---|
| `overlay_ratio` | Interference-to-machine gain ratio; 1.0 corresponds to approximately 0 dB SNR |
| `samples` | Number of test samples synthesized at this ratio |
| `precision` / `recall` / `f1` | Anomaly detection metrics at this ratio |
| `std_precision` / `std_recall` / `std_f1` | Standard deviation across folds |

## results/overall_statistics.csv

| Column | Description |
|---|---|
| `total_samples` | Total number of held-out test samples (3,300) |
| `num_folds` | Number of cross-validation folds (10) |
| `event_f1` / `event_f1_std` | Event recognition F1 (mean / SD) |
| `event_iou` / `event_iou_std` | Event mask temporal IoU (mean / SD) |
| `device_accuracy` / `device_accuracy_std` | Machine-type classification accuracy (mean / SD) |
| `anomaly_precision` / `anomaly_precision_std` | Anomaly detection precision (mean / SD) |
| `anomaly_recall` / `anomaly_recall_std` | Anomaly detection recall (mean / SD) |
| `anomaly_f1` / `anomaly_f1_std` | Anomaly detection F1-score (mean / SD) |

## fold_assignments.json

| Field | Description |
|---|---|
| `universe_product_ids` | All product IDs per machine type appearing in the synthetic corpus |
| `folds.fold_k.test_product_ids` | Product IDs held out for testing in fold k |
| `folds.fold_k.train_product_ids` | Product IDs used for training in fold k |

No product ID appears in both the training and test partitions of the same fold (machine-ID-wise split).

## semantic_tree.json

Hierarchical ontology of the audio-tag labels. Each node contains:

| Field | Description |
|---|---|
| `label` | Node name (category or leaf audio tag) |
| `phrases` | Text phrases associated with the node, embedded with `paraphrase-MiniLM-L6-v2` |
| `children` | Child nodes (empty for leaves) |
