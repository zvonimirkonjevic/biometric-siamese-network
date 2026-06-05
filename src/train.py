import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from torchinfo import summary

from model import SiameseNetwork
from dataset import FingerprintPairDataset
from training import Trainer

BATCH_SIZE = 32
TRAIN_PAIRS = 20_000
TEST_PAIRS = 4_000
IMG_SIZE = 96
EMB_DIM = 128
EPOCHS = 20
LEARNING_RATE = 1e-3

DATA_ROOT = Path(__file__).parent.parent / "data"
ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb1, emb2, labels):
        d = F.pairwise_distance(emb1, emb2)
        genuine_loss = (1 - labels) * d.pow(2)
        impostor_loss = labels * F.relu(self.margin - d).pow(2)
        return (genuine_loss + impostor_loss).mean()


def siamese_adapter(model, batch, func):
    img1, img2, labels = batch
    return func(model(img1), model(img2), labels)


def _resolve_paths(df: pd.DataFrame) -> pd.DataFrame:
    base = Path(__file__).parent
    df = df.copy()
    df["path"] = df["path"].apply(lambda p: str((base / p).resolve()))
    return df


def main():
    train_real_df = _resolve_paths(pd.read_csv(DATA_ROOT / "train_real.csv"))
    test_real_df = _resolve_paths(pd.read_csv(DATA_ROOT / "test_real.csv"))
    train_altered_df = _resolve_paths(pd.read_csv(DATA_ROOT / "train_altered.csv"))
    test_altered_df = _resolve_paths(pd.read_csv(DATA_ROOT / "test_altered.csv"))

    print(f"Train subjects: {train_real_df['subject'].nunique()}  |  Test subjects: {test_real_df['subject'].nunique()}")

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(degrees=10),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    train_dataset = FingerprintPairDataset(train_real_df, train_altered_df, num_pairs=TRAIN_PAIRS, transform=train_transform)
    test_dataset = FingerprintPairDataset(test_real_df, test_altered_df, num_pairs=TEST_PAIRS, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Train: {len(train_loader)} batches  |  Test: {len(test_loader)} batches")

    model = SiameseNetwork(emb_dim=EMB_DIM, head_dropout=0.3, img_size=IMG_SIZE)
    summary(model, (BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE))

    loss_fn = ContrastiveLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    save_path = ARTIFACTS_DIR / "siamese_net.pt"

    trainer = Trainer()
    trainer.set_adapter(siamese_adapter)

    trainer.fit(
        model,
        train_loader,
        loss_fn,
        optimizer,
        epochs=EPOCHS,
        valid_loader=test_loader,
        scheduler=scheduler,
        patience=3,
        checkpoint_path=save_path,
    )

    print(f"Best model saved to {save_path}")


if __name__ == "__main__":
    main()
