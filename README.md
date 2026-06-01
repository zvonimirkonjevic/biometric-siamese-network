# Biometric Siamese Network

Fingerprint verification using a Siamese Network trained on the [SOCOFing dataset](https://www.kaggle.com/datasets/ruizgara/socofing). The model learns a similarity metric over fingerprint image pairs to distinguish genuine matches from altered or impostor samples — the core task behind biometric authentication systems.

---

## Problem

Fingerprint verification is a **one-shot similarity problem**: given two fingerprint images, determine whether they belong to the same finger. Classical methods rely on hand-crafted minutiae features. This project replaces that pipeline with a learned deep similarity function that generalises across subjects, fingers, and synthetic alterations.

## Approach

A **Siamese Network** processes two fingerprint images through a shared CNN encoder, then computes a distance between the resulting embeddings. Training uses contrastive loss over genuine/impostor pairs:

```
Encoder(image_A) ──┐
                   ├──► Distance ──► Similarity score
Encoder(image_B) ──┘
```

Pairs are constructed as:
- **Genuine:** two images of the same finger (same subject, same finger position)
- **Impostor:** same finger against an altered version (obliteration, central rotation, or Z-cut)

The altered subset introduces three difficulty tiers (Easy / Medium / Hard), enabling evaluation of robustness to progressive image degradation.

## Dataset — SOCOFing

**Sokoto Coventry Fingerprint Dataset** ([paper](https://arxiv.org/abs/1807.10609)) — Shehu et al., Coventry University.

| Property | Value |
|---|---|
| Subjects | 600 African individuals (18+) |
| Genuine images | 6,000 (10 fingers × 600 subjects) |
| Altered images | ~49,000 across Easy / Medium / Hard |
| Labels | gender, hand (L/R), finger position |
| Alteration types | Obliteration, Central Rotation, Z-cut |
| License | CC BY-NC-SA 4.0 |

File naming convention: `{ID}__{Gender}_{Hand}_{Finger}_finger.BMP`  
Example: `100__M_Left_index_finger.BMP`

## Project Structure

```
src/
├── data/          # Dataset loading, pair sampling, augmentation
├── model/         # Siamese network and encoder architecture
├── train.py       # Training loop with contrastive loss
└── evaluate.py    # ROC/AUC, EER, and threshold analysis
notebooks/
└── exploration.ipynb   # Dataset EDA and result visualisation
```

## Setup

Requires Python 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/zvonimirkonjevic/biometric-siamese-network.git
cd biometric-siamese-network
uv sync
```

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/ruizgara/socofing) and place it under `data/`:

```
data/
├── Real/
└── Altered/
    ├── Altered-Easy/
    ├── Altered-Medium/
    └── Altered-Hard/
```

## Key Design Decisions

**Shared weights** — both branches of the Siamese network share identical weights, enforcing that the same representation is learned regardless of which image is query vs. reference.

**Contrastive loss** — directly optimises embedding distance: genuine pairs are pulled together, impostor pairs pushed apart with a margin.

**Difficulty curriculum** — training progresses from Easy to Hard alterations, preventing the model from overfitting to simple forgery patterns before seeing harder ones.

**Evaluation metric** — Equal Error Rate (EER) and ROC-AUC are standard in biometric verification; accuracy alone is misleading on imbalanced pair sets.

## References

- Shehu, Y. I., Ruiz-Garcia, A., Palade, V., & James, A. (2018). [Sokoto Coventry Fingerprint Dataset](https://arxiv.org/abs/1807.10609). *arXiv:1807.10609*.
- Koch, G., Zemel, R., & Salakhutdinov, R. (2015). [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf). *ICML Deep Learning Workshop*.
- Chopra, S., Hadsell, R., & LeCun, Y. (2005). [Learning a Similarity Metric Discriminatively, with Application to Face Verification](http://yann.lecun.com/exdb/publis/pdf/chopra-05.pdf). *CVPR*.
