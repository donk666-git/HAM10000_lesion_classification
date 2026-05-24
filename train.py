# train_inceptionresnetv2_softattention_metadata_weighted.py

# ============================================================
# Step 1. Import libraries and set paths
# ============================================================

from pathlib import Path
import os
import random
import argparse
import json

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from torchvision import transforms
import timm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


# ============================================================
# Step 2. Set random seed
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ============================================================
# Step 3. Dataset definition
# ============================================================

class HAMMetadataDataset(Dataset):
    def __init__(self, dataframe, metadata_array, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.metadata_array = metadata_array
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(row["image_path"]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        meta = torch.tensor(self.metadata_array[idx], dtype=torch.float32)
        label = torch.tensor(row["label"], dtype=torch.long)

        image_id = row["image_id"]
        true_label_text = row["dx"]

        return image, meta, label, image_id, true_label_text


# ============================================================
# Step 4. Soft-Attention module
# ============================================================

class SoftAttention(nn.Module):
    """
    Soft-Attention module applied on the final CNN feature map.

    Input:
        x: [B, C, H, W]

    Output:
        out: [B, 2C, H, W]
        attn_map: [B, 1, H, W]
    """

    def __init__(self, in_channels, num_heads=16):
        super().__init__()

        self.num_heads = num_heads

        self.attention_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_heads,
            kernel_size=1,
            bias=True
        )

        # Learnable scaling parameter.
        # Initialized as 0 to keep early training stable.
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape

        attn = self.attention_conv(x)              # [B, K, H, W]
        attn = attn.view(B, self.num_heads, -1)    # [B, K, H*W]
        attn = torch.softmax(attn, dim=-1)
        attn = attn.view(B, self.num_heads, H, W)  # [B, K, H, W]

        attn_map = attn.mean(dim=1, keepdim=True)  # [B, 1, H, W]

        attended = x * attn_map
        attended = self.gamma * attended

        # Concatenate original feature map and attended feature map
        out = torch.cat([x, attended], dim=1)       # [B, 2C, H, W]

        return out, attn_map


# ============================================================
# Step 5. InceptionResNetV2 + Soft-Attention + Metadata model
# ============================================================

class InceptionResNetV2SoftAttentionMetadata(nn.Module):
    def __init__(self, metadata_dim, num_classes=7, pretrained=True):
        super().__init__()

        self.backbone = timm.create_model(
            "inception_resnet_v2",
            pretrained=pretrained,
            features_only=True,
            out_indices=(-1,)
        )

        feature_channels = self.backbone.feature_info.channels()[-1]

        self.soft_attention = SoftAttention(
            in_channels=feature_channels,
            num_heads=16
        )

        self.image_pool = nn.AdaptiveAvgPool2d(1)

        self.metadata_mlp = nn.Sequential(
            nn.Linear(metadata_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

        self.classifier = nn.Sequential(
            nn.Linear(feature_channels * 2 + 64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(512, num_classes)
        )

    def forward(self, image, metadata, return_attention=False):
        feature_map = self.backbone(image)[-1]         # [B, C, H, W]

        feature_map, attn_map = self.soft_attention(feature_map)

        image_feat = self.image_pool(feature_map)
        image_feat = image_feat.flatten(1)

        meta_feat = self.metadata_mlp(metadata)

        feat = torch.cat([image_feat, meta_feat], dim=1)
        logits = self.classifier(feat)

        if return_attention:
            return logits, attn_map

        return logits


# ============================================================
# Step 6. Metrics
# ============================================================

def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
    }


# ============================================================
# Step 7. Train one epoch
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for images, metas, labels, _, _ in tqdm(loader, desc="Training"):
        images = images.to(device, non_blocking=True)
        metas = metas.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        logits = model(images, metas)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)

    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = avg_loss

    return metrics


# ============================================================
# Step 8. Evaluate
# ============================================================

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    all_image_ids = []

    for images, metas, labels, image_ids, _ in tqdm(loader, desc="Validation"):
        images = images.to(device, non_blocking=True)
        metas = metas.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images, metas)
        loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        total_loss += loss.item() * images.size(0)

        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())
        all_image_ids.extend(image_ids)

    avg_loss = total_loss / len(loader.dataset)

    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = avg_loss

    return (
        metrics,
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
        all_image_ids
    )


# ============================================================
# Step 9. Plot and save training curves
# ============================================================

def save_training_curves(history_df, output_dir):
    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="train_loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["val_accuracy"], label="val_accuracy")
    plt.plot(
        history_df["epoch"],
        history_df["val_balanced_accuracy"],
        label="val_balanced_accuracy"
    )
    plt.plot(history_df["epoch"], history_df["val_f1_macro"], label="val_f1_macro")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.title("Validation Metrics")
    plt.tight_layout()
    plt.savefig(output_dir / "metric_curve.png", dpi=300)
    plt.close()


# ============================================================
# Step 10. Save final evaluation files
# ============================================================

def save_final_outputs(
    output_dir,
    val_metrics,
    y_true,
    y_pred,
    y_probs,
    image_ids,
    classes,
    idx2label
):
    num_classes = len(classes)

    # metrics.csv
    pd.DataFrame([val_metrics]).to_csv(
        output_dir / "metrics.csv",
        index=False
    )

    # predictions.csv
    pred_df = pd.DataFrame({
        "image_id": image_ids,
        "true_label": [idx2label[i] for i in y_true],
        "pred_label": [idx2label[i] for i in y_pred],
        "correct": y_true == y_pred
    })

    for i, cls in idx2label.items():
        pred_df[f"prob_{cls}"] = y_probs[:, i]

    pred_df.to_csv(output_dir / "predictions.csv", index=False)

    # per_class_metrics.csv
    report = classification_report(
        y_true,
        y_pred,
        target_names=classes,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(output_dir / "per_class_metrics.csv")

    # confusion_matrix.csv
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_df.to_csv(output_dir / "confusion_matrix.csv")

    # confusion_matrix.png
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - InceptionResNetV2 + Soft-Attention + Metadata + Weighted Loss")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300)
    plt.close()


# ============================================================
# Step 11. Main function
# ============================================================

def main(args):
    set_seed(args.seed)

    root = Path(args.root)
    meta_path = root / args.metadata
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    label2idx = {c: i for i, c in enumerate(classes)}
    idx2label = {i: c for c, i in label2idx.items()}
    num_classes = len(classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("Device:", device)
    print("Root:", root)
    print("Metadata:", meta_path)
    print("Output:", output_dir)
    print("=" * 80)

    # ========================================================
    # Step 12. Read metadata.csv
    # ========================================================

    df = pd.read_csv(meta_path)

    required_cols = [
        "lesion_id",
        "image_id",
        "dx",
        "dx_type",
        "age",
        "sex",
        "localization",
        "dataset"
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column in metadata.csv: {col}")

    print("Original dataframe shape:", df.shape)
    print("Class distribution:")
    print(df["dx"].value_counts())

    # ========================================================
    # Step 13. Build image paths
    # ========================================================

    def get_image_path(row):
        cls = row["dx"]
        image_id = row["image_id"]
        path = root / cls / "enhanced" / f"{image_id}.jpg"

        if path.exists():
            return str(path)

        return None

    df["image_path"] = df.apply(get_image_path, axis=1)

    missing_images = df["image_path"].isna().sum()
    print("Missing images:", missing_images)

    if missing_images > 0:
        missing_df = df[df["image_path"].isna()][["image_id", "dx", "image_path"]]
        missing_df.to_csv(output_dir / "missing_images.csv", index=False)

        raise FileNotFoundError(
            f"{missing_images} images are missing. "
            f"Check {output_dir / 'missing_images.csv'}."
        )

    df = df.reset_index(drop=True)

    # ========================================================
    # Step 14. Encode labels
    # ========================================================

    df["label"] = df["dx"].map(label2idx)

    if df["label"].isna().any():
        raise ValueError("Some labels are not in the predefined class list.")

    df["label"] = df["label"].astype(int)

    # ========================================================
    # Step 15. 90% train / 10% validation split
    # ========================================================

    train_df, val_df = train_test_split(
        df,
        test_size=args.val_ratio,
        stratify=df["dx"],
        random_state=args.seed
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    print("Train size:", len(train_df))
    print("Val size:", len(val_df))

    print("Train distribution:")
    print(train_df["dx"].value_counts())

    print("Val distribution:")
    print(val_df["dx"].value_counts())

    train_df.to_csv(output_dir / "train_split.csv", index=False)
    val_df.to_csv(output_dir / "val_split.csv", index=False)

    # ========================================================
    # Step 16. Process metadata: age + sex + localization
    # ========================================================

    train_df = train_df.copy()
    val_df = val_df.copy()

    age_median = train_df["age"].median()

    train_df["age"] = train_df["age"].fillna(age_median)
    val_df["age"] = val_df["age"].fillna(age_median)

    for col in ["sex", "localization"]:
        train_df[col] = train_df[col].fillna("unknown").astype(str)
        val_df[col] = val_df[col].fillna("unknown").astype(str)

    scaler = StandardScaler()
    train_age = scaler.fit_transform(train_df[["age"]])
    val_age = scaler.transform(val_df[["age"]])

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    train_cat = encoder.fit_transform(train_df[["sex", "localization"]])
    val_cat = encoder.transform(val_df[["sex", "localization"]])

    train_meta = np.concatenate([train_age, train_cat], axis=1).astype("float32")
    val_meta = np.concatenate([val_age, val_cat], axis=1).astype("float32")

    metadata_dim = train_meta.shape[1]

    print("Metadata dim:", metadata_dim)
    print("Metadata categories:", encoder.categories_)

    metadata_info = {
        "metadata_dim": int(metadata_dim),
        "age_median": float(age_median),
        "sex_categories": encoder.categories_[0].tolist(),
        "localization_categories": encoder.categories_[1].tolist()
    }

    with open(output_dir / "metadata_info.json", "w", encoding="utf-8") as f:
        json.dump(metadata_info, f, indent=2, ensure_ascii=False)

    # ========================================================
    # Step 17. Compute class weights
    # ========================================================

    class_counts = train_df["label"].value_counts().sort_index().values
    N = class_counts.sum()
    C = num_classes

    class_weights = N / (C * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    print("Class weights:")
    for cls, count, weight in zip(classes, class_counts, class_weights):
        print(f"{cls:6s} count={count:5d}, weight={weight.item():.4f}")

    weights_df = pd.DataFrame({
        "class": classes,
        "count": class_counts,
        "weight": class_weights.numpy()
    })
    weights_df.to_csv(output_dir / "class_weights.csv", index=False)

    # ========================================================
    # Step 18. Image transforms
    # ========================================================

    train_tfms = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.03
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # ========================================================
    # Step 19. DataLoader
    # ========================================================

    train_dataset = HAMMetadataDataset(
        train_df,
        train_meta,
        transform=train_tfms
    )

    val_dataset = HAMMetadataDataset(
        val_df,
        val_meta,
        transform=val_tfms
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Quick batch check
    batch = next(iter(train_loader))
    images, metas, labels, image_ids, true_labels_text = batch

    print("Batch image shape:", images.shape)
    print("Batch metadata shape:", metas.shape)
    print("Batch label shape:", labels.shape)

    # ========================================================
    # Step 20. Create model, loss, optimizer, scheduler
    # ========================================================

    model = InceptionResNetV2SoftAttentionMetadata(
        metadata_dim=metadata_dim,
        num_classes=num_classes,
        pretrained=True
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3
    )

    # Forward test
    model.eval()
    with torch.no_grad():
        images = images.to(device)
        metas = metas.to(device)
        labels = labels.to(device)

        logits, attn_map = model(images, metas, return_attention=True)
        loss = criterion(logits, labels)

    print("Forward test:")
    print("Logits shape:", logits.shape)
    print("Attention map shape:", attn_map.shape)
    print("Loss:", loss.item())

   # ========================================================
# Resume from checkpoint if specified
# ========================================================

    start_epoch = 1
    best_f1 = -1.0
    no_improve = 0
    history = []

    if args.resume:
        print("\nLoading checkpoint from:", args.resume)
        checkpoint = torch.load(args.resume, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        print("Loaded model weights.")

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("Loaded optimizer state.")

        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print("Loaded scheduler state.")

        best_f1 = checkpoint.get("best_f1", -1.0)
        no_improve = checkpoint.get("no_improve", 0)
        start_epoch = checkpoint.get("epoch", 0) + 1

        hist_path = output_dir / "training_history.csv"
        if hist_path.exists():
            existing_history = pd.read_csv(hist_path)
            history = existing_history.to_dict("records")
            print(f"Loaded existing training history. Last epoch in CSV: {len(history)}.")
        elif "history" in checkpoint:
            history = checkpoint["history"]
            print(f"Loaded training history from checkpoint. Last epoch: {len(history)}.")
        else:
            history = []

        print(f"Resume from epoch {start_epoch}")
        print(f"best_f1 = {best_f1:.4f}")
        print(f"no_improve = {no_improve}/{args.patience}")

        if start_epoch > args.epochs:
            print(f"start_epoch ({start_epoch}) > args.epochs ({args.epochs}). Consider increasing --epochs.")
            args.epochs = max(args.epochs, start_epoch + args.patience)
            print(f"Auto-increased --epochs to {args.epochs} so training can proceed.")

    # ========================================================
    # Step 21. Formal training
    # ========================================================

    best_f1 = checkpoint.get("best_f1", -1.0) if args.resume else -1.0
    no_improve = checkpoint.get("no_improve", 0) if args.resume else 0
    if not args.resume:
        history = []

    for epoch in range(start_epoch, args.epochs + 1):
        print("\n" + "=" * 80)
        print(f"Epoch {epoch}/{args.epochs}")
        print("=" * 80)

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        val_metrics, y_true, y_pred, y_probs, image_ids = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step(val_metrics["f1_macro"])

        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "lr": optimizer.param_groups[0]["lr"]
        }

        history.append(row)

        print("Train metrics:", train_metrics)
        print("Val metrics:", val_metrics)
        print("LR:", optimizer.param_groups[0]["lr"])

        history_df = pd.DataFrame(history)
        history_df.to_csv(output_dir / "training_history.csv", index=False)

        current_f1 = val_metrics["f1_macro"]

        if current_f1 > best_f1:
            best_f1 = current_f1
            no_improve = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_f1": best_f1,
                    "classes": classes,
                    "label2idx": label2idx,
                    "idx2label": idx2label,
                    "metadata_dim": metadata_dim,
                    "age_median": age_median,
                    "class_weights": class_weights,
                    "image_size": args.image_size,
                },
                output_dir / "best_model.pth"
            )

            print(f"Saved best model. val_f1_macro = {best_f1:.4f}")

            save_final_outputs(
                output_dir=output_dir,
                val_metrics=val_metrics,
                y_true=y_true,
                y_pred=y_pred,
                y_probs=y_probs,
                image_ids=image_ids,
                classes=classes,
                idx2label=idx2label
            )

        else:
            no_improve += 1
            print(f"No improvement: {no_improve}/{args.patience}")


        # Save last checkpoint every epoch for true resume
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_f1": best_f1,
                "no_improve": no_improve,
                "history": history,
                "classes": classes,
                "label2idx": label2idx,
                "idx2label": idx2label,
                "metadata_dim": metadata_dim,
                "age_median": age_median,
                "class_weights": class_weights,
                "image_size": args.image_size,
            },
            output_dir / "last_checkpoint.pth"
        )


        if no_improve >= args.patience:
            print("Early stopping triggered.")
            break

    # ========================================================
    # Step 22. Load best model and save final outputs again
    # ========================================================

    print("\nLoading best model for final evaluation...")

    checkpoint = torch.load(output_dir / "best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_metrics, y_true, y_pred, y_probs, image_ids = evaluate(
        model,
        val_loader,
        criterion,
        device
    )

    print("Final validation metrics:")
    print(val_metrics)

    save_final_outputs(
        output_dir=output_dir,
        val_metrics=val_metrics,
        y_true=y_true,
        y_pred=y_pred,
        y_probs=y_probs,
        image_ids=image_ids,
        classes=classes,
        idx2label=idx2label
    )

    history_df = pd.read_csv(output_dir / "training_history.csv")
    save_training_curves(history_df, output_dir)

    print("\nDone.")
    print("Outputs saved to:", output_dir)


# ============================================================
# Argument parser
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=str,
        default="./",
        help="Root directory of preprocessing data."
    )

    parser.add_argument(
        "--metadata",
        type=str,
        default="metadata.csv",
        help="Metadata csv filename under root."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs_inceptionresnetv2_softattention_metadata_weighted",
        help="Output directory."
    )

    parser.add_argument(
        "--image_size",
        type=int,
        default=299,
        help="Input image size."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Maximum number of epochs."
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=7,
        help="Early stopping patience."
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate."
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay."
    )

    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation ratio."
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader workers."
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed."
    )

    args = parser.parse_args()
    main(args)