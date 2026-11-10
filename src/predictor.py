from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = ROOT / "models" / "plantcare_effnet_b0.pth"
LABELS_PATH = ROOT / "models" / "class_labels.json"


@dataclass(frozen=True)
class Prediction:
    label: str
    score: float


class EfficientNetPredictor:
    def __init__(self, model: Any, labels: list[str], transform: Any, device: Any) -> None:
        self.model = model
        self.labels = labels
        self.transform = transform
        self.device = device

    def predict(self, image: Image.Image, top_k: int = 3) -> list[Prediction]:
        import torch

        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(tensor)[0], dim=0)
            scores, indices = torch.topk(probabilities, k=min(top_k, len(self.labels)))
        return [
            Prediction(self.labels[int(index)], float(score))
            for score, index in zip(scores.cpu(), indices.cpu())
        ]


def load_predictor() -> tuple[EfficientNetPredictor | None, str]:
    if not WEIGHTS_PATH.exists() or not LABELS_PATH.exists():
        return None, (
            "Disease predictions are not available yet. Add trained model weights and a matching "
            "class-label file to the models folder. The site will not invent a disease result."
        )

    try:
        import torch
        from torchvision import models, transforms
    except ImportError:
        return None, "Optional model packages are missing. Install the packages listed in requirements-model.txt."

    try:
        labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
        if not isinstance(labels, list) or not labels or not all(isinstance(label, str) for label in labels):
            return None, "The class-label file must contain a JSON list of class names."

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(labels))
        checkpoint = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        model.load_state_dict(checkpoint)
        model.to(device).eval()

        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        return EfficientNetPredictor(model, labels, transform, device), ""
    except Exception:
        return None, "The model files could not be loaded. Check that the weights match the class-label list."
