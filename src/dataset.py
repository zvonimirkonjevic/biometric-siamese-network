import os
import re
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

FINGER_LABELS = {
    "index_finger": 0,
    "little_finger": 1,
    "middle_finger": 2,
    "ring_finger": 3,
    "thumb_finger": 4,
}

ALTERATION_LABELS = {
    "CR": "Central Rotation",
    "Obl": "Obliteration",
    "Zcut": "Z-cut",
}


def parse_real_filename(filename: str, real_dir: Path) -> dict:
    pattern = r"(\d+)__([MF])_(Left|Right)_(index|little|middle|ring|thumb)_finger\.BMP$"
    match = re.match(pattern, filename)
    if match:
        subject, gender, hand, finger = match.groups()
        return {
            "subject": int(subject),
            "gender": gender,
            "hand": hand,
            "finger": f"{finger}_finger",
            "label": FINGER_LABELS[f"{finger}_finger"],
            "filename": filename,
            "path": str(real_dir / filename),
        }
    raise ValueError(f"Filename '{filename}' does not match the expected pattern for real fingerprints.")


def parse_altered_filename(filename: str, difficulty: str, altered_dir: Path) -> dict:
    pattern = r"(\d+)__([MF])_(Left|Right)_(index|little|middle|ring|thumb)_finger_(CR|Obl|Zcut)\.BMP$"
    match = re.match(pattern, filename)
    if match:
        subject, gender, hand, finger, alteration = match.groups()
        return {
            "subject": int(subject),
            "gender": gender,
            "hand": hand,
            "finger": f"{finger}_finger",
            "label": FINGER_LABELS[f"{finger}_finger"],
            "alteration": alteration,
            "difficulty": difficulty,
            "filename": filename,
            "path": str(altered_dir / difficulty / filename),
        }
    raise ValueError(f"Filename '{filename}' does not match the expected pattern for altered fingerprints.")


class FingerprintPairDataset(Dataset):
    def __init__(self, real_df, altered_df, num_pairs: int = 20_000, transform=None):
        self.num_pairs = num_pairs
        self.transform = transform

        self.real_by_identity = (
            real_df.groupby(["subject", "finger"])["path"]
            .apply(list).to_dict()
        )
        self.altered_by_identity = (
            altered_df.groupby(["subject", "finger"])["path"]
            .apply(list).to_dict()
        )
        self.identities = list(self.real_by_identity.keys())
        self.identities_with_altered = [
            k for k in self.identities if k in self.altered_by_identity
        ]
        # Only identities with ≥2 real images can form a non-degenerate genuine pair.
        self.identities_with_pairs = [
            k for k, v in self.real_by_identity.items() if len(v) >= 2
        ]

    def __len__(self):
        return self.num_pairs

    def _load(self, path: str):
        img = Image.open(path).convert("L")
        if self.transform:
            img = self.transform(img)
        return img

    def __getitem__(self, _idx):
        r = random.random()

        if r < 0.5:
            # Genuine: same subject, same finger — distinct L and R hand images
            key = random.choice(self.identities_with_pairs)
            paths = self.real_by_identity[key]
            p1, p2 = random.sample(paths, 2)
            label = 0

        elif r < 0.75:
            # Impostor A: real vs its own altered version (forgery detection)
            key = random.choice(self.identities_with_altered)
            p1 = random.choice(self.real_by_identity[key])
            p2 = random.choice(self.altered_by_identity[key])
            label = 1

        else:
            # Impostor B: real vs a completely different subject's real (identity boundary)
            k1, k2 = random.sample(self.identities, 2)
            p1 = random.choice(self.real_by_identity[k1])
            p2 = random.choice(self.real_by_identity[k2])
            label = 1

        return (
            self._load(p1),
            self._load(p2),
            torch.tensor(label, dtype=torch.float32),
        )
