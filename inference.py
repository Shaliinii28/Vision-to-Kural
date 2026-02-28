# app/inference.py
"""
Core inference engine for Vision-to-Kural.

This module is imported by app.py. It handles:
  1. Loading all models and indexes at startup (once)
  2. The image → top-K Kural retrieval pipeline at request time

Nothing in this file involves Sarvam-2B at runtime — Sarvam's
work is already baked into the FAISS index. Only CLIP runs live.

Startup time: ~15-25 seconds (CLIP load + index load)
Per-request time: ~300-500ms on CPU
"""

import json
import time
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import faiss
from PIL import Image

# Import CLIP
try:
    import clip as openai_clip
except ImportError:
    raise ImportError("openai-clip not installed.\nRun: pip install openai-clip")

from model import ProjectionHead

log = logging.getLogger("inference")

# ─────────────────────────────────────────────
# File paths (relative to app/ directory)
# ─────────────────────────────────────────────
APP_DIR  = Path(__file__).parent
INDEX_PATH    = APP_DIR / "kural_index.faiss"
METADATA_PATH = APP_DIR / "kural_metadata.json"
WEIGHTS_PATH  = APP_DIR / "projection.pt"

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
CLIP_MODEL_NAME = "ViT-L/14"
IMG_IN_DIM      = 768    # CLIP ViT-L/14 output dim
OUT_DIM         = 512    # shared projection dim
HIDDEN_DIM      = 1024

PAL_META = {
    "Virtue": {"emoji": "🌿", "color": "#065F46", "bg": "#ECFDF5", "tamil": "அறத்துப்பால்"},
    "Wealth": {"emoji": "💰", "color": "#78350F", "bg": "#FFFBEB", "tamil": "பொருட்பால்"},
    "Love":   {"emoji": "❤️",  "color": "#7F1D1D", "bg": "#FFF1F2", "tamil": "காமத்துப்பால்"},
}

# ─────────────────────────────────────────────
# Global model state (loaded once)
# ─────────────────────────────────────────────
_state: dict = {}


def is_loaded() -> bool:
    return bool(_state)


def load_models(device: str = "cpu"):
    """
    Load CLIP, projection head, FAISS index, and metadata into
    module-level _state dict. Call once at app startup.
    """
    if is_loaded():
        log.info("Models already loaded, skipping.")
        return

    start = time.time()
    log.info("Loading models…")

    # ── Validate files ──
    for path, label in [
        (INDEX_PATH,    "kural_index.faiss"),
        (METADATA_PATH, "kural_metadata.json"),
        (WEIGHTS_PATH,  "projection.pt"),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                f"Run the offline pipeline (scripts/01-04) first."
            )

    # ── CLIP ──
    log.info(f"Loading CLIP {CLIP_MODEL_NAME}…")
    clip_model, clip_preprocess = openai_clip.load(CLIP_MODEL_NAME, device=device)
    clip_model.eval()
    _state["clip"]       = clip_model
    _state["preprocess"] = clip_preprocess
    _state["device"]     = device
    log.info("CLIP loaded ✓")

    # ── Projection head (image side only) ──
    log.info("Loading projection head…")
    ckpt = torch.load(WEIGHTS_PATH, map_location="cpu")
    cfg  = ckpt.get("config", {})

    img_proj = ProjectionHead(
        in_dim     = cfg.get("img_in_dim",  IMG_IN_DIM),
        out_dim    = cfg.get("out_dim",     OUT_DIM),
        hidden_dim = cfg.get("hidden_dim",  HIDDEN_DIM),
    )
    img_proj.load_state_dict(ckpt["img_proj"])
    img_proj.eval()
    img_proj = img_proj.to(device)
    _state["img_proj"] = img_proj
    log.info("Projection head loaded ✓")

    # ── FAISS index ──
    log.info("Loading FAISS index…")
    index = faiss.read_index(str(INDEX_PATH))
    _state["index"] = index
    log.info(f"FAISS index loaded: {index.ntotal} vectors ✓")

    # ── Kural metadata ──
    log.info("Loading Kural metadata…")
    with open(METADATA_PATH, encoding="utf-8") as f:
        kurals = json.load(f)
    _state["kurals"] = kurals
    log.info(f"Kural metadata loaded: {len(kurals)} entries ✓")

    elapsed = time.time() - start
    log.info(f"All models ready in {elapsed:.1f}s")


# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────
@torch.no_grad()
def encode_image(pil_image: Image.Image) -> np.ndarray:
    """
    Encode a PIL image → 512-dim projected & normalised vector.
    Returns: numpy array (1, 512) float32
    """
    device     = _state["device"]
    preprocess = _state["preprocess"]
    clip_model = _state["clip"]
    img_proj   = _state["img_proj"]

    tensor = preprocess(pil_image).unsqueeze(0).to(device)
    clip_feat = clip_model.encode_image(tensor).float()  # (1, 768)
    projected = img_proj(clip_feat)                       # (1, 512)
    arr = projected.cpu().numpy().astype(np.float32)
    faiss.normalize_L2(arr)
    return arr  # (1, 512)


def retrieve_kurals(
    pil_image: Image.Image,
    top_k: int = 5,
    pal_filter: Optional[str] = None,
) -> list[dict]:
    """
    Main retrieval function.

    Args:
        pil_image:  PIL Image from the user upload.
        top_k:      Number of Kurals to return (after filtering).
        pal_filter: Optional filter — "Virtue", "Wealth", or "Love".
                    If None, returns from all three Pals.

    Returns:
        List of Kural dicts with an added 'score' field.
        Sorted by descending cosine similarity.
    """
    if not is_loaded():
        raise RuntimeError("Call load_models() before retrieve_kurals()")

    index  = _state["index"]
    kurals = _state["kurals"]

    # Encode image
    query = encode_image(pil_image)   # (1, 512)

    # Fetch more candidates if filtering
    fetch_k = top_k * 4 if pal_filter and pal_filter != "All" else top_k + 2
    scores, indices = index.search(query, k=min(fetch_k, index.ntotal))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(kurals):
            continue
        kural = kurals[idx]

        # Apply Pal filter
        if pal_filter and pal_filter not in ("All", "all"):
            if kural.get("pal", "") != pal_filter:
                continue

        entry = {
            **kural,
            "score":    float(score),
            "pal_meta": PAL_META.get(kural.get("pal", ""), {
                "emoji": "📜", "color": "#1A56A8",
                "bg": "#EFF6FF", "tamil": "",
            }),
        }
        results.append(entry)

        if len(results) == top_k:
            break

    return results
