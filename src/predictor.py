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


# Houseplant identification model (ViT-based, 47 classes)
HOUSEPLANT_MODEL_URL = (
    "https://huggingface.co/onnx-community/house-plant-image-detection-ONNX/resolve/main/onnx/model_int8.onnx"
)
HOUSEPLANT_LABELS_URL = (
    "https://huggingface.co/dima806/house-plant-image-detection/raw/main/config.json"
)
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
HOUSEPLANT_MODEL_PATH = MODEL_DIR / "house-plant-int8.onnx"
HOUSEPLANT_LABELS_PATH = MODEL_DIR / "house-plant-labels.json"
PLANT_CARE_DB_PATH = MODEL_DIR / "plant_care_database.json"


@dataclass(frozen=True)
class Prediction:
    label: str
    score: float


@dataclass(frozen=True)
class PlantCareInfo:
    scientific_name: str
    watering_schedule: dict
    care_instructions: dict
    characteristics: dict
    price: float | None
    category: str


@dataclass(frozen=True)
class ClassificationResult:
    candidates: list[Prediction]
    plant_care_info: PlantCareInfo | None = None


def _download(url: str, destination: Path, label: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".download")
    request = urllib.request.Request(url, headers={"User-Agent": "PlantCareAI-local-prototype/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            total = 0
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                total += len(chunk)
                if total > 200 * 1024 * 1024:
                    raise ValueError(f"The downloaded {label} was unexpectedly large.")
        if total == 0:
            raise ValueError(f"The downloaded {label} was empty.")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _ensure_model_files() -> None:
    if not HOUSEPLANT_MODEL_PATH.exists():
        _download(HOUSEPLANT_MODEL_URL, HOUSEPLANT_MODEL_PATH, "houseplant classifier")
    if not HOUSEPLANT_LABELS_PATH.exists():
        _download(HOUSEPLANT_LABELS_URL, HOUSEPLANT_LABELS_PATH, "model labels")


class OnnxHouseplantPredictor:
    def __init__(self, session: Any, labels: list[str], plant_care_db: dict) -> None:
        self.session = session
        self.labels = labels
        self.plant_care_db = plant_care_db
        self.input_name = session.get_inputs()[0].name
        self.output_names = [output.name for output in session.get_outputs()]

    def predict(self, image: Image.Image, top_k: int = 3) -> ClassificationResult:
        # ViT preprocessing: resize to 224x224, normalize with mean=0.5, std=0.5
        image = image.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        tensor = ((pixels - mean) / std).transpose(2, 0, 1)[None, ...].astype(np.float32)

        outputs = self.session.run(None, {self.input_name: tensor})
        # Output is logits directly
        logits = outputs[0].reshape(-1)

        # Softmax
        logits = logits - np.max(logits)
        probabilities = np.exp(logits)
        probabilities = probabilities / probabilities.sum()

        indices = np.argsort(probabilities)[::-1][: min(top_k, len(self.labels))]
        candidates = [
            Prediction(self.labels[int(index)], float(probabilities[int(index)]))
            for index in indices
        ]

        # Get care info for top prediction
        plant_care_info = None
        if candidates:
            top_idx = str(candidates[0].label)  # This is the label index as string
            # Find the index in labels list
            try:
                label_idx = self.labels.index(candidates[0].label)
                care_data = self.plant_care_db.get(str(label_idx))
                if care_data and care_data.get("scientific_name"):
                    plant_care_info = PlantCareInfo(
                        scientific_name=care_data["scientific_name"],
                        watering_schedule=care_data["watering_schedule"],
                        care_instructions=care_data["care_instructions"],
                        characteristics=care_data["characteristics"],
                        price=care_data["price"],
                        category=care_data["category"],
                    )
            except (ValueError, KeyError):
                pass

        return ClassificationResult(candidates=candidates, plant_care_info=plant_care_info)


@lru_cache(maxsize=1)
def load_predictor() -> tuple[OnnxHouseplantPredictor | None, str]:
    try:
        _ensure_model_files()
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        return None, (
            "The local model could not be downloaded. Check your internet connection and try again. "
            f"Details: {exc}"
        )

    try:
        # Load labels from config.json (keys are strings "0", "1", etc. directly)
        config = json.loads(HOUSEPLANT_LABELS_PATH.read_text(encoding="utf-8"))
        # The file is already a dict with string keys "0", "1", etc.
        labels = [config[str(i)] for i in range(len(config))]

        # Load plant care database
        plant_care_db = {}
        if PLANT_CARE_DB_PATH.exists():
            plant_care_db = json.loads(PLANT_CARE_DB_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, KeyError, AttributeError) as exc:
        return None, f"Could not load model labels or care database: {exc}"

    try:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.log_severity_level = 3
        session = ort.InferenceSession(
            str(HOUSEPLANT_MODEL_PATH),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        return OnnxHouseplantPredictor(session, labels, plant_care_db), ""
    except Exception as exc:
        return None, f"The local classifier could not start: {exc}"


def format_label(label: str) -> str:
    """Format plant label for display - already in readable format."""
    return label