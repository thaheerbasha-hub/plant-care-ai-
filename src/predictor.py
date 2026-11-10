from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


MODEL_REPO = "rodynaemad/plant-disease-dinov2-small"
MODEL_REVISION = "06ebf5e308e100fd000d738c42b059ebb5f58a16"
SPACE_REVISION = "b26e704bd72c2fe05632ea6f7be680a2eca0d8f5"
MODEL_URL = (
    f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/"
    "onnx/model_with_features_quantized.onnx"
)
SPACE_ROOT = "https://huggingface.co/spaces/rodynaemad/plant-disease-scanner/resolve/"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH = MODEL_DIR / "plant-disease-dinov2-small-int8.onnx"
META_PATH = MODEL_DIR / "model_meta.json"
CENTROIDS_PATH = MODEL_DIR / "centroids.f32"


@dataclass(frozen=True)
class Prediction:
    label: str
    score: float


@dataclass(frozen=True)
class ClassificationResult:
    candidates: list[Prediction]
    similarity: float
    in_scope: bool


def _download(url: str, destination: Path, label: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".download")
    request = urllib.request.Request(url, headers={"User-Agent": "PlantCareAI-local-prototype/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as output:
            total = 0
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                total += len(chunk)
                if total > 150 * 1024 * 1024:
                    raise ValueError(f"The downloaded {label} was unexpectedly large.")
        if total == 0:
            raise ValueError(f"The downloaded {label} was empty.")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _ensure_model_files() -> None:
    if not MODEL_PATH.exists():
        _download(MODEL_URL, MODEL_PATH, "classifier")
    if not META_PATH.exists():
        _download(f"{SPACE_ROOT}{SPACE_REVISION}/meta.json", META_PATH, "model metadata")
    if not CENTROIDS_PATH.exists():
        _download(f"{SPACE_ROOT}{SPACE_REVISION}/centroids.f32", CENTROIDS_PATH, "out-of-scope centroids")


class OnnxPlantDiseasePredictor:
    def __init__(self, session: Any, labels: list[str], centroids: np.ndarray, threshold: float) -> None:
        self.session = session
        self.labels = labels
        self.centroids = centroids
        self.threshold = threshold
        self.input_name = session.get_inputs()[0].name
        self.output_names = [output.name for output in session.get_outputs()]

    def predict(self, image: Image.Image, top_k: int = 3) -> ClassificationResult:
        # The published model card specifies 224x224 RGB input with ImageNet normalization.
        image = image.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tensor = ((pixels - mean) / std).transpose(2, 0, 1)[None, ...].astype(np.float32)

        outputs = self.session.run(None, {self.input_name: tensor})
        output_by_name = dict(zip(self.output_names, outputs))
        logits = next(
            (value for name, value in output_by_name.items() if "logit" in name.lower()),
            outputs[0],
        )
        features = next(
            (value for name, value in output_by_name.items() if "feature" in name.lower()),
            None,
        )
        if features is None:
            raise RuntimeError("The classifier did not return its feature vector for the out-of-scope check.")
        logits = np.asarray(logits, dtype=np.float64).reshape(-1)
        logits -= np.max(logits)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()
        indices = np.argsort(probabilities)[::-1][: min(top_k, len(self.labels))]
        feature = np.asarray(features, dtype=np.float32).reshape(-1)
        feature_norm = float(np.linalg.norm(feature))
        if feature_norm == 0:
            similarity = -1.0
        else:
            similarity = float(np.max(self.centroids @ (feature / feature_norm)))
        candidates = [
            Prediction(self.labels[int(index)], float(probabilities[int(index)]))
            for index in indices
        ]
        return ClassificationResult(candidates, similarity, similarity >= self.threshold)


@lru_cache(maxsize=1)
def load_predictor() -> tuple[OnnxPlantDiseasePredictor | None, str]:
    try:
        _ensure_model_files()
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        return None, (
            "The local model could not be downloaded. Check your internet connection and try again. "
            f"Details: {exc}"
        )

    try:
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        labels = meta["labels"]
        threshold = float(meta["threshold"])
        dimension = int(meta["dim"])
        if len(labels) != 38 or dimension != 768:
            return None, "The downloaded model class list was not in the expected format."
        centroids = np.fromfile(CENTROIDS_PATH, dtype="<f4")
        if centroids.size != len(labels) * dimension:
            return None, "The out-of-scope metadata did not match the classifier."
        centroids = centroids.reshape(len(labels), dimension)
    except (OSError, json.JSONDecodeError, KeyError, AttributeError):
        return None, "The downloaded model class list could not be read."

    try:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.log_severity_level = 3
        session = ort.InferenceSession(
            str(MODEL_PATH),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        return OnnxPlantDiseasePredictor(session, labels, centroids, threshold), ""
    except Exception as exc:
        return None, f"The local classifier could not start: {exc}"


def format_label(label: str) -> str:
    crop, _, condition = label.partition("___")
    crop = crop.replace("_", " ")
    condition = condition.replace("_", " ")
    if not condition:
        return crop
    if condition.lower() == "healthy":
        return f"{crop} - Healthy"
    return f"{crop} - {condition}"
