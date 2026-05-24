# Deep Learning — HAM10000 皮肤 lesion 七分类

基于 InceptionResNetV2 + Soft-Attention + 临床元数据融合的皮肤镜图像分类模型，在 HAM10000 数据集上实现 7 类病变诊断。

## 项目结构

```text
Deep learning/
├── README.md
├── train.py                                          # 训练脚本
├── preprocess.ipynb                                  # 图像预处理 / 增强
├── test_1epoch.ipynb                                 # 单 epoch 快速验证
├── test_1epoch_soft-attention.ipynb                  # 单 epoch 验证（含 attention）
└── outputs_inceptionresnetv2_softattention_metadata_weighted/
    ├── best_model.pth                                # 最优模型（按 val F1-macro）
    ├── last_checkpoint.pth                           # 最新 checkpoint（可恢复训练）
    ├── metrics.csv                                   # 最终验证指标
    ├── per_class_metrics.csv                         # 各类别 precision/recall/F1
    ├── confusion_matrix.csv / .png                   # 混淆矩阵
    ├── loss_curve.png                                # 训练/验证 loss 曲线
    ├── metric_curve.png                              # 验证指标曲线
    ├── training_history.csv                          # 每 epoch 指标
    ├── predictions.csv                               # 逐样本预测结果
    ├── class_weights.csv                             # 类别权重
    ├── metadata_info.json                            # 元数据编码配置
    └── train_split.csv / val_split.csv               # 训练/验证划分记录
```

## Pipeline

整体架构：**InceptionResNetV2 + Soft-Attention + Metadata + Weighted Loss**，90% 训练 / 10% 验证。

```
preprocess.ipynb
  图像裁剪、增强质量 → 输出 {dx}/enhanced/{image_id}.jpg
      │
      ▼
train.py
  │
  ├── 数据准备
  │   [1] 读取 metadata.csv
  │   [2] 构建 enhanced 图像路径，检查缺失
  │   [3] 标签编码 → 7 类
  │   [4] 元数据处理：年龄标准化 + 性别/部位 one-hot → 19 维向量
  │   [5] 按频率倒数计算类别权重
  │   [6] 90/10 分层划分 → DataLoader
  │
  ├── 模型构建
  │   [7] InceptionResNetV2(features-only, ImageNet 预训练)
  │       + Soft-Attention(16头, γ=0 初始化)
  │       + Metadata MLP(19→64)
  │       → 拼接 → FC(2C+64 → 512 → 7)
  │
  ├── 验证单轮
  │   [8] forward/backward 各跑 1 epoch，确认无报错
  │
  ├── 正式训练
  │   [9] 最多 30 epochs
  │       · 每轮：train → validate → ReduceLROnPlateau（按 val F1-macro）
  │       · F1 提升 → 保存 best_model.pth
  │       · 每轮   → 保存 last_checkpoint.pth（含 optimizer/scheduler）
  │       · patience=7 无提升 → early stopping
  │
  └── 最终输出
      [10] 加载最优模型 → 全量评估
            → metrics.csv / confusion_matrix.png
            → per_class_metrics.csv / predictions.csv
            → loss_curve.png / metric_curve.png
            → training_history.csv
```

### 一些设计

- **γ 初始化为 0**：Attention 输出在训练初期被屏蔽，保持 backbone 稳定，逐步学习有效后再接入
- **加权 CrossEntropyLoss**：缓解 nv:df ≈ 55:1 的类别不平衡问题
- **ReduceLROnPlateau 按 F1-macro 调度**：相比按 loss 降 LR，更直接优化不平衡数据下的核心指标


## 模型架构

```
图像 (299×299) ──► InceptionResNetV2 ──► [B, C, H, W]
                                         │
                                         ▼
                                   Soft-Attention (16 头)
                                   ├─ 1×1 conv → spatial softmax
                                   ├─ 多头平均 → attention map
                                   └─ concat(x, x ⊙ attn_map) → [B, 2C, H, W]
                                         │
                                         ▼
                                   AdaptiveAvgPool2d → [B, 2C]
                                                          +
元数据 (年龄+性别+部位) ──► MLP → [B, 64] ──► [B, 2C+64]
                                                          │
                                                          ▼
                                                  FC(2C+64 → 512 → 7)
```

| 组件 | 说明 |
|------|------|
| Backbone | InceptionResNetV2（timm，ImageNet 预训练，仅取 features） |
| Soft-Attention | 16 头 1×1 conv 空间 attention，可学习缩放参数 γ（初始化为 0） |
| Metadata MLP | 19 维 → 128 → 64，BatchNorm + ReLU + Dropout(0.3) |
| 分类器 | 2C+64 维 → 512 → 7，BatchNorm + ReLU + Dropout(0.5) |
| Loss | 加权 CrossEntropyLoss（按类别频率倒数） |

### 元数据编码

- **年龄**: StandardScaler 标准化，缺失值用中位数填充（1 维）
- **性别**: one-hot（女 / 男 / 未知，3 维）
- **部位**: one-hot（15 类身体部位，15 维）

## 数据集

[HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) — 10,015 张皮肤镜图像，7 个诊断类别。

### 数据存放

图像路径：`{root}/{dx}/enhanced/{image_id}.jpg`，其中 `{dx}` 为诊断标签。元数据文件 `{root}/metadata.csv` 需包含列：`lesion_id`, `image_id`, `dx`, `dx_type`, `age`, `sex`, `localization`, `dataset`。

### 类别

| 缩写 | 中文名 | 说明 |
|------|--------|------|
| akiec | 日光性角化病 | 表皮内癌 |
| bcc | 基底细胞癌 | 常见皮肤癌 |
| bkl | 良性角化病 | 脂溢性角化、日光性雀斑样痣 |
| df | 皮肤纤维瘤 | 良性纤维性病变 |
| mel | 黑色素瘤 | 恶性黑色素瘤 |
| nv | 色素痣 | 良性痣（多数类） |
| vasc | 血管性病变 | 血管瘤、化脓性肉芽肿 |

### 划分

90% 训练 / 10% 验证，按类别分层（seed=42）。

## 依赖

```bash
pip install torch torchvision timm numpy pandas scikit-learn Pillow matplotlib seaborn tqdm
```

## 使用

### 训练

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

### 恢复训练

```bash
python train.py \
  --resume outputs_inceptionresnetv2_softattention_metadata_weighted/last_checkpoint.pth \
  --epochs 40
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--root` | `./` | 数据根目录 |
| `--metadata` | `metadata.csv` | 元数据文件名 |
| `--output_dir` | `outputs_inceptionresnetv2_softattention_metadata_weighted` | 输出目录 |
| `--image_size` | 299 | 输入图像尺寸 |
| `--batch_size` | 16 | batch size |
| `--epochs` | 30 | 最大训练轮数 |
| `--patience` | 7 | early stopping 轮数 |
| `--lr` | 1e-4 | 初始学习率 |
| `--weight_decay` | 1e-4 | AdamW weight decay |
| `--val_ratio` | 0.1 | 验证集比例 |
| `--num_workers` | 4 | DataLoader 线程数 |
| `--seed` | 42 | 随机种子 |
| `--resume` | None | checkpoint 路径 |

### 训练配置

| 超参数 | 值 |
|--------|----|
| 优化器 | AdamW (lr=1e-4, β=(0.9, 0.999), weight_decay=1e-4) |
| 学习率调度 | ReduceLROnPlateau (factor=0.5, patience=3, 监控 val F1-macro) |
| Loss | 加权 CrossEntropyLoss |
| Batch size | 16 |
| 图像尺寸 | 299×299 |
| 最大 epoch | 30 |
| Early stopping | patience=7（监控 val F1-macro） |
| 数据增强 | 随机翻转（水平+垂直），旋转（±30°），ColorJitter |
| 归一化 | ImageNet 均值/方差 |

## 训练结果

最优 epoch：**18/30**（第 25 轮触发 early stopping，patience=7）。

| 指标 | 值 |
|------|----|
| Accuracy | **89.02%** |
| Balanced Accuracy | **88.28%** |
| Precision (macro) | 83.84% |
| Recall (macro) | 88.28% |
| F1 (macro) | **85.93%** |

### 各类别指标

| 类别 | Precision | Recall | F1 | 样本数 |
|------|-----------|--------|----|--------|
| nv | 0.951 | 0.924 | **0.937** | 671 |
| vasc | 0.929 | 0.929 | **0.929** | 14 |
| df | 0.857 | 1.000 | **0.923** | 12 |
| bcc | 0.870 | 0.922 | **0.895** | 51 |
| bkl | 0.800 | 0.836 | **0.818** | 110 |
| akiec | 0.778 | 0.848 | **0.812** | 33 |
| mel | 0.684 | 0.721 | **0.702** | 111 |

### 小结

- 整体准确率 89%，宏平均 F1 85.9%，在类别不平衡条件下表现稳定。
- 多数类 nv 效果最好（F1=0.937）；样本极少的 df 和 vasc 因加权 loss 召回率分别达到 100% 和 92.9%。
- 黑色素瘤 mel 表现最弱（F1=0.702），与良性痣（nv, bkl）视觉特征高度重叠，是皮肤镜分类的常见难点。
- 第 18 轮后验证 F1 在 0.84~0.86 波动不再提升，当前架构在此数据规模下接近上限。后续可尝试更大输入尺寸、更强数据增强或 ensemble。
