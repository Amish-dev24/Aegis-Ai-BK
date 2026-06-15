"""
Violence Detection Model Training Script
=========================================
Trains a lightweight Conv3D model to classify video clips as violent / non-violent.

Dataset: RWF-2000 (2000 clips: 1000 violent, 1000 non-violent)
Model:   Small Conv3D (runs on CPU, ~2-4 hours training)
Output:  violence_model.onnx (for integration with Aegis AI backend)

STEP 1: Download the dataset
-----------------------------
Option A (Kaggle - easiest):
  1. Go to: https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset
  2. Download and extract to: E:/violence_dataset/
  3. Structure should be:
     E:/violence_dataset/
       Violence/        (1000 videos)
       NonViolence/     (1000 videos)

Option B (RWF-2000 from GitHub):
  1. Go to: https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection
  2. Follow download instructions
  3. Extract to: E:/RWF-2000/
     E:/RWF-2000/
       train/
         Fight/       (800 videos)
         NonFight/    (800 videos)
       val/
         Fight/       (200 videos)
         NonFight/    (200 videos)

STEP 2: Run this script
------------------------
  cd E:/Aegis_Ai_Bk
  python scripts/train_violence_model.py --dataset E:/violence_dataset

STEP 3: Model will be saved to
  models/violence_model.pt
  models/violence_model.onnx
"""

import argparse
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# ============================================================
# CONFIG
# ============================================================
NUM_FRAMES = 16        # frames per clip
FRAME_SIZE = 64        # resize frames to 64x64
BATCH_SIZE = 4         # small batch for CPU
EPOCHS = 20
LEARNING_RATE = 0.001
TRAIN_SPLIT = 0.8      # 80% train, 20% val


# ============================================================
# DATASET
# ============================================================
class ViolenceDataset(Dataset):
    """Load video clips and extract fixed-length frame sequences."""

    def __init__(self, video_paths, labels, num_frames=NUM_FRAMES, frame_size=FRAME_SIZE):
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.frame_size = frame_size

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        frames = self._extract_frames(video_path)

        # Normalize to [0, 1] and convert to tensor (C, T, H, W)
        frames = frames.astype(np.float32) / 255.0
        # frames shape: (T, H, W, C) -> (C, T, H, W)
        frames = np.transpose(frames, (3, 0, 1, 2))

        return torch.from_numpy(frames), torch.tensor(label, dtype=torch.long)

    def _extract_frames(self, video_path):
        """Extract exactly num_frames from the video, evenly spaced."""
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            return np.zeros((self.num_frames, self.frame_size, self.frame_size, 3), dtype=np.uint8)

        # Pick evenly spaced frame indices
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

        frames = []
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (self.frame_size, self.frame_size))
                frames.append(frame)
            else:
                # Duplicate last frame if read fails
                if frames:
                    frames.append(frames[-1].copy())
                else:
                    frames.append(np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8))

        cap.release()
        return np.array(frames[:self.num_frames])


# ============================================================
# MODEL - Lightweight Conv3D
# ============================================================
class ViolenceClassifier(nn.Module):
    """
    Small Conv3D model for binary violence classification.
    Input: (batch, 3, 16, 64, 64)
    Output: (batch, 2)  [non-violent, violent]
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: (3, 16, 64, 64) -> (32, 8, 32, 32)
            nn.Conv3d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            # Block 2: (32, 8, 32, 32) -> (64, 4, 16, 16)
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            # Block 3: (64, 4, 16, 16) -> (128, 2, 8, 8)
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            # Block 4: (128, 2, 8, 8) -> (256, 1, 4, 4)
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 1 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x


# ============================================================
# FIND DATASET
# ============================================================
def find_videos(dataset_path):
    """
    Auto-detect dataset structure and return (video_paths, labels).
    Supports:
      - Violence/ + NonViolence/
      - Fight/ + NonFight/
      - train/Fight/ + train/NonFight/ (RWF-2000)
    """
    dataset_path = Path(dataset_path)
    video_paths = []
    labels = []

    # Try different folder structures
    structures = [
        # (violent_folder, non_violent_folder)
        ("Violence", "NonViolence"),
        ("Fight", "NonFight"),
        ("train/Fight", "train/NonFight"),
        ("violent", "non-violent"),
        ("violence", "nonviolence"),
    ]

    found = False
    for violent_dir, nonviolent_dir in structures:
        v_path = dataset_path / violent_dir
        nv_path = dataset_path / nonviolent_dir
        if v_path.exists() and nv_path.exists():
            print(f"Found dataset structure: {violent_dir}/ + {nonviolent_dir}/")

            # Collect violent videos
            for ext in ["*.mp4", "*.avi", "*.mov", "*.mkv"]:
                for f in v_path.glob(ext):
                    video_paths.append(str(f))
                    labels.append(1)  # violent

            # Collect non-violent videos
            for ext in ["*.mp4", "*.avi", "*.mov", "*.mkv"]:
                for f in nv_path.glob(ext):
                    video_paths.append(str(f))
                    labels.append(0)  # non-violent

            found = True
            break

    if not found:
        print(f"\nERROR: Could not find dataset at {dataset_path}")
        print("Expected folder structure:")
        print(f"  {dataset_path}/Violence/    (violent video clips)")
        print(f"  {dataset_path}/NonViolence/ (non-violent video clips)")
        print("\nDownload from: https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset")
        sys.exit(1)

    return video_paths, labels


# ============================================================
# TRAINING
# ============================================================
def train():
    parser = argparse.ArgumentParser(description="Train violence detection model")
    parser.add_argument("--dataset", required=True, help="Path to dataset folder")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help=f"Training epochs (default: {EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--output", default="./models", help="Output directory for model files")
    args = parser.parse_args()

    print("=" * 60)
    print("AEGIS AI - Violence Detection Model Training")
    print("=" * 60)

    # Find videos
    print(f"\nSearching for videos in: {args.dataset}")
    video_paths, labels = find_videos(args.dataset)

    violent_count = sum(labels)
    nonviolent_count = len(labels) - violent_count
    print(f"Found {len(video_paths)} videos ({violent_count} violent, {nonviolent_count} non-violent)")

    if len(video_paths) < 20:
        print("ERROR: Not enough videos. Need at least 20.")
        sys.exit(1)

    # Shuffle and split
    combined = list(zip(video_paths, labels, strict=False))
    random.seed(42)
    random.shuffle(combined)
    video_paths, labels = zip(*combined, strict=False)

    split_idx = int(len(video_paths) * TRAIN_SPLIT)
    train_paths, val_paths = video_paths[:split_idx], video_paths[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]

    print(f"Train: {len(train_paths)} videos | Val: {len(val_paths)} videos")

    # Create datasets
    print("\nLoading datasets...")
    train_dataset = ViolenceDataset(train_paths, train_labels)
    val_dataset = ViolenceDataset(val_paths, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Create model
    device = torch.device("cpu")
    model = ViolenceClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    print(f"Device: {device}")
    print(f"\nStarting training for {args.epochs} epochs...")
    print("-" * 60)

    best_val_acc = 0.0
    best_epoch = 0
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # ---- Train ----
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (frames, targets) in enumerate(train_loader):
            frames, targets = frames.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

            # Progress indicator
            if (batch_idx + 1) % 10 == 0:
                print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}", end="\r")

        train_acc = 100.0 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for frames, targets in val_loader:
                frames, targets = frames.to(device), targets.to(device)
                outputs = model(frames)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_acc = 100.0 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        scheduler.step(avg_val_loss)

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1:2d}/{args.epochs} | "
              f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.1f}% | "
              f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.1f}% | "
              f"Time: {epoch_time:.0f}s")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch + 1,
                "val_acc": val_acc,
                "train_acc": train_acc,
            }, str(output_dir / "violence_model.pt"))
            print(f"  >> New best model saved (val_acc: {val_acc:.1f}%)")

    print("-" * 60)
    print(f"Training complete! Best val accuracy: {best_val_acc:.1f}% (epoch {best_epoch})")

    # ---- Export to ONNX ----
    print("\nExporting to ONNX...")
    checkpoint = torch.load(str(output_dir / "violence_model.pt"), map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dummy_input = torch.randn(1, 3, NUM_FRAMES, FRAME_SIZE, FRAME_SIZE)
    onnx_path = str(output_dir / "violence_model.onnx")

    torch.onnx.export(
        model, dummy_input, onnx_path,
        opset_version=17,
        input_names=["video_clip"],
        output_names=["prediction"],
        dynamic_axes={"video_clip": {0: "batch"}},
    )
    print(f"ONNX model saved to: {onnx_path}")

    # Verify ONNX
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        test_output = session.run(None, {"video_clip": dummy_input.numpy()})
        print(f"ONNX verification passed! Output shape: {test_output[0].shape}")
    except ImportError:
        print("onnxruntime not installed — skipping verification")

    print(f"\n{'=' * 60}")
    print(f"DONE! Model files saved in: {output_dir}/")
    print("  - violence_model.pt   (PyTorch checkpoint)")
    print("  - violence_model.onnx (ONNX for fast inference)")
    print(f"  - Best accuracy: {best_val_acc:.1f}%")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    train()
