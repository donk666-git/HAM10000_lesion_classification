# Deep Learning — HAM10000 Skin Lesion Classification

InceptionResNetV2 + Soft-Attention + Clinical Metadata fusion for 7-class skin lesion classification on the [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) dataset.

## Table of Contents

- [Background](#background)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Dataset](#dataset)
- [Usage](#usage)
- [Outputs](#outputs)
- [Training Results](#training-results)

## Background

Automated skin lesion classification is a challenging task due to high inter-class visual similarity and severe class imbalance (e.g., melanocytic nevi outnumber dermatofibroma by ~55:1). This project combines a pretrained InceptionResNetV2 backbone with a spatial soft-attention mechanism and clinical metadata (age, sex, body site) to improve diagnostic accuracy, particularly for minority classes.

Inspired by the HAM10000_EVALUATE Refactored baseline (traditional handcrafted-feature + SVM/ANN pipeline), this deep learning counterpart explores whether learned features from a CNN backbone can outperform the manual feature-engineering approach on the same dataset.

## Architecture

```
Image (299×299) ──► InceptionResNetV2 ──► [B, C, H, W]
                                         │
                                         ▼
                                   Soft-Attention (16 heads)
                                   ├─ 1×1 conv → spatial softmax
                                   ├─ multi-head average → attention map
                                   └─ concat(x, x ⊙ attn_map) → [B, 2C, H, W]
                                         │
                                         ▼
                                   AdaptiveAvgPool2d → [B, 2C]
                                                          +
Metadata (age + sex + localization) ──► MLP → [B, 64] ──► [B, 2C+64]
                                                          │
                                                          ▼
                                                  FC(2C+64 → 512 → 7)
```

| Component | Detail |
|-----------|--------|
| Backbone | InceptionResNetV2 (timm, pretrained on ImageNet, features-only, last feature block) |
| Soft-Attention | 16-head 1×1 conv spatial attention, learnable scaling parameter γ initialized to 0 |
| Metadata MLP | 19-dim → 128 → 64 with BatchNorm + ReLU + Dropout(0.3) |
| Classifier | 2C+64-dim → 512 → 7 with BatchNorm + ReLU + Dropout(0.5) |
| Loss | Weighted CrossEntropyLoss (inverse-frequency class weights) |

### Metadata encoding

Metadata (19-dim) is constructed from three clinical attributes:

- **Age**: StandardScaler normalized, median imputation for missing values (1-dim)
- **Sex**: one-hot encoded (female / male / unknown, 3-dim)
- **Localization**: one-hot encoded across 15 body-site categories (15-dim)

## Project Structure

```text
Deep learning/
├── README.md
├── train.py                                          # Main training script
├── preprocess.ipynb                                  # Image preprocessing / enhancement
├── test_1epoch.ipynb                                 # Quick single-epoch test
├── test_1epoch_soft-attention.ipynb                  # Single-epoch test with attention
└── outputs_inceptionresnetv2_softattention_metadata_weighted/
    ├── best_model.pth                                # Best checkpoint (val F1-macro)
    ├── last_checkpoint.pth                           # Latest checkpoint (resumable)
    ├── metrics.csv                                   # Final validation metrics
    ├── per_class_metrics.csv                         # Per-class precision/recall/F1
    ├── confusion_matrix.csv / .png                   # Confusion matrix
    ├── loss_curve.png                                # Train/val loss curves
    ├── metric_curve.png                              # Val metric curves
    ├── training_history.csv                          # Per-epoch metrics
    ├── predictions.csv                               # Per-sample predictions + probs
    ├── class_weights.csv                             # Computed class weights
    ├── metadata_info.json                            # Metadata encoding config
    └── train_split.csv / val_split.csv               # Train/val split record
```

## Dependencies

- Python ≥ 3.8
- PyTorch ≥ 1.12
- torchvision
- [timm](https://github.com/huggingface/pytorch-image-models)
- numpy, pandas
- scikit-learn
- Pillow
- matplotlib, seaborn
- tqdm

```bash
pip install torch torchvision timm numpy pandas scikit-learn Pillow matplotlib seaborn tqdm
```

## Dataset

[HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) — 10,015 dermatoscopic images across 7 diagnostic categories.

### Data layout

Images are expected under `{root}/{dx}/enhanced/{image_id}.jpg`, where `{dx}` is the diagnosis label. A `metadata.csv` file must be present at `{root}/metadata.csv` with columns: `lesion_id`, `image_id`, `dx`, `dx_type`, `age`, `sex`, `localization`, `dataset`.

### Classes

| Abbrev. | Full Name | Description |
|---------|-----------|-------------|
| akiec | Actinic Keratoses | Intraepithelial carcinoma |
| bcc | Basal Cell Carcinoma | Common skin cancer |
| bkl | Benign Keratosis | Seborrheic keratosis, solar lentigo |
| df | Dermatofibroma | Benign fibrous lesion |
| mel | Melanoma | Malignant melanoma |
| nv | Melanocytic Nevi | Benign moles (majority class) |
| vasc | Vascular Lesions | Angiomas, pyogenic granulomas |

### Split

90% train / 10% validation, stratified by class label (random seed 42).

## Usage

### Training

```bash
python train.py \
  --root /path/to/ham10000_data \
  --image-size 299 \
  --batch-size 16 \
  --epochs 30 \
  --lr 1e-4 \
  --patience 7 \
  --seed 42
```

### Resume from checkpoint

```bash
python train.py \
  --resume outputs_inceptionresnetv2_softattention_metadata_weighted/last_checkpoint.pth \
  --epochs 40
```

### Full argument list

| Argument | Default | Description |
|----------|---------|-------------|
| `--root` | `./` | Root data directory |
| `--metadata` | `metadata.csv` | Metadata filename under root |
| `--output_dir` | `outputs_inceptionresnetv2_softattention_metadata_weighted` | Output directory |
| `--image_size` | 299 | Input image size (square) |
| `--batch_size` | 16 | Batch size |
| `--epochs` | 30 | Max training epochs |
| `--patience` | 7 | Early stopping patience |
| `--lr` | 1e-4 | Initial learning rate |
| `--weight_decay` | 1e-4 | AdamW weight decay |
| `--val_ratio` | 0.1 | Validation split ratio |
| `--num_workers` | 4 | DataLoader workers |
| `--seed` | 42 | Random seed |
| `--resume` | None | Path to checkpoint for resume |

### Training configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW (lr=1e-4, β=(0.9, 0.999), weight_decay=1e-4) |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=3, monitor=val_f1_macro) |
| Loss | Weighted CrossEntropyLoss |
| Batch size | 16 |
| Image size | 299×299 |
| Max epochs | 30 |
| Early stopping | patience=7 (monitor val_f1_macro) |
| Augmentation | Random flips (H+V), rotation (±30°), ColorJitter (brightness=0.15, contrast=0.15, saturation=0.10, hue=0.03) |
| Normalization | ImageNet stats (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) |

## Outputs

All run artifacts are saved under `--output_dir`:

| Artifact | Contents |
|----------|----------|
| `best_model.pth` | Model weights + metadata config at best val F1-macro |
| `last_checkpoint.pth` | Full checkpoint with optimizer/scheduler state (resumable) |
| `metrics.csv` | Final validation metrics (accuracy, balanced accuracy, precision/recall/F1) |
| `per_class_metrics.csv` | Per-class precision, recall, F1-score, support |
| `confusion_matrix.csv/png` | Raw and visualized confusion matrix |
| `loss_curve.png` | Train and validation loss over epochs |
| `metric_curve.png` | Validation accuracy, balanced accuracy, F1-macro over epochs |
| `training_history.csv` | Per-epoch train + val metrics and LR |
| `predictions.csv` | Per-sample predictions with class probabilities |
| `class_weights.csv` | Computed inverse-frequency class weights |
| `metadata_info.json` | Age median, sex/localization category lists |
| `train_split.csv` / `val_split.csv` | Dataset split record |

## Training Results

Best epoch: **18/30** (early stopping at epoch 25, patience=7).

| Metric | Value |
|--------|-------|
| Accuracy | **89.02%** |
| Balanced Accuracy | **88.28%** |
| Precision (macro) | 83.84% |
| Recall (macro) | 88.28% |
| F1 (macro) | **85.93%** |

### Per-class

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| nv | 0.951 | 0.924 | **0.937** | 671 |
| vasc | 0.929 | 0.929 | **0.929** | 14 |
| df | 0.857 | 1.000 | **0.923** | 12 |
| bcc | 0.870 | 0.922 | **0.895** | 51 |
| bkl | 0.800 | 0.836 | **0.818** | 110 |
| akiec | 0.778 | 0.848 | **0.812** | 33 |
| mel | 0.684 | 0.721 | **0.702** | 111 |

### Summary

- 整体准确率 89%，宏平均 F1 为 85.9%，说明多标签不平衡条件下模型表现稳定。
- 多数类别 (nv) 表现最好 (F1=0.937)；样本极少的 df 和 vasc 受益于类别权重损失函数，召回率分别达到满分 100% 和 92.9%。
- 黑色素瘤 (mel) 表现最弱 (F1=0.702)，这是皮肤镜分类的常见难点——黑色素瘤与良性痣 (nv, bkl) 视觉特征高度重叠，仅靠单一 CNN 骨干难以进一步提升。
- 模型在第 18 轮达到最优后，验证 F1 在 0.84~0.86 之间波动不再提升，说明当前架构在此数据规模下的提升空间已近上限，后续可以考虑更大的输入尺寸、更强的数据增强或引入 ensemble。

### Comparison with referenced paper

| Backbone                              | Soft-Attention | Weighted Loss  |   Accuracy | Balanced Acc / Recall |  Precision |         F1 |
| ------------------------------------- | -------------- | -------------- | ---------: | --------------------: | ---------: | ---------: |
| PNASNet-5-Large / ensemble-style DNNs | No             | Not main focus |          - | 0.76 validation score |          - |          - |
| InceptionResNetV2                     | Yes            | Yes            |       0.90 |           0.81 recall |       0.86 |       0.86 |
| InceptionResNetV2                     | Yes            | Yes            | **0.8902** |            **0.8828** | **0.8384** | **0.8593** |