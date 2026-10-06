"""Frozen rubert-tiny2 encoder loaded strictly from a local directory."""

from __future__ import annotations

from pathlib import Path

from transformers import AutoModel, AutoTokenizer

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "assets" / "model"


class Backbone:
    """Tokenizer plus an encoder whose weights never receive gradients."""

    def __init__(self, model_dir: str | Path | None = None) -> None:
        directory = Path(model_dir) if model_dir is not None else DEFAULT_MODEL_DIR
        if not (directory / "model.safetensors").is_file():
            raise FileNotFoundError(
                f"model weights not found in {directory}; run ./download_model.sh first"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
        self.model = AutoModel.from_pretrained(str(directory), local_files_only=True)
        self.model.requires_grad_(False)
        self.model.eval()
