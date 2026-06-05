import sys
from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))
from src.model import SiameseNetwork

IMG_SIZE = 96

_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


CHECKPOINT = Path(__file__).parent / "artifacts" / "siamese_net.pt"


@st.cache_resource
def load_model() -> SiameseNetwork:
    model = SiameseNetwork()
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def embed(model: SiameseNetwork, img: Image.Image) -> torch.Tensor:
    x = _transform(img.convert("L")).unsqueeze(0)
    with torch.inference_mode():
        return model(x)


st.set_page_config(page_title="Fingerprint Verifier", layout="wide")
st.title("Fingerprint Verification")
st.caption("Upload two fingerprint images to check whether they belong to the same finger.")

with st.sidebar:
    st.header("Model")
    threshold = st.slider(
        "Match threshold (L2 distance)",
        min_value=0.1, max_value=2.0, value=0.5, step=0.01,
        help="Pairs with distance below this are classified as genuine.",
    )

model = None
if CHECKPOINT.exists():
    try:
        model = load_model()
        st.sidebar.success("Model loaded.")
    except Exception as e:
        st.sidebar.error(f"Could not load model: {e}")
else:
    st.sidebar.warning(f"No checkpoint found. Run `src/train.py` first.")

col1, col2 = st.columns(2)

img1 = img2 = None

with col1:
    st.subheader("Fingerprint A")
    file1 = st.file_uploader("Upload image", type=["bmp", "png", "jpg", "jpeg"], key="fp1")
    if file1:
        img1 = Image.open(file1)
        st.image(img1, use_container_width=True)

with col2:
    st.subheader("Fingerprint B")
    file2 = st.file_uploader("Upload image", type=["bmp", "png", "jpg", "jpeg"], key="fp2")
    if file2:
        img2 = Image.open(file2)
        st.image(img2, use_container_width=True)

st.divider()

ready = img1 is not None and img2 is not None and model is not None
if st.button("Compare", type="primary", disabled=not ready):
    emb1 = embed(model, img1)
    emb2 = embed(model, img2)
    dist = F.pairwise_distance(emb1, emb2).item()
    match = dist < threshold

    st.metric(
        label="L2 distance",
        value=f"{dist:.4f}",
        delta=f"{threshold - dist:+.4f} margin",
        delta_color="normal",
    )

    if match:
        st.success(f"MATCH — distance {dist:.4f} is below threshold {threshold:.2f}")
    else:
        st.error(f"NO MATCH — distance {dist:.4f} exceeds threshold {threshold:.2f}")
