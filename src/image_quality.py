from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ImageQuality:
    width: int
    height: int
    brightness: float
    contrast: float
    blur_score: float
    clipped_fraction: float
    summary: str


@dataclass(frozen=True)
class PreparationRoute:
    name: str
    message: str
    applied: bool = False
    should_retake: bool = False


def assess_image(image: Image.Image) -> ImageQuality:
    """Calculate simple photo checks for the prototype, not a diagnosis."""
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    clipped_fraction = float(np.mean((gray < 12) | (gray > 243)))

    issues: list[str] = []
    if blur_score < 45:
        issues.append("the photo may be blurry")
    if brightness < 55:
        issues.append("the photo looks quite dark")
    elif brightness > 205:
        issues.append("the photo looks very bright")
    if contrast < 28:
        issues.append("the contrast is low")
    if clipped_fraction > 0.25:
        issues.append("many pixels are very dark or very bright")

    if issues:
        summary = "The photo may need attention: " + ", ".join(issues) + "."
    else:
        summary = "The basic photo checks look usable. This does not confirm that the image is suitable for diagnosis."

    return ImageQuality(
        width=image.width,
        height=image.height,
        brightness=brightness,
        contrast=contrast,
        blur_score=blur_score,
        clipped_fraction=clipped_fraction,
        summary=summary,
    )


def prepare_image(image: Image.Image, quality: ImageQuality) -> tuple[Image.Image, PreparationRoute]:
    """Apply one mild contrast adjustment when the heuristic flags low contrast."""
    if quality.blur_score < 25 or quality.brightness < 28 or quality.brightness > 228:
        return image, PreparationRoute(
            name="retake",
            message="The photo is too blurry or poorly exposed for a useful screening attempt. Please retake it in steady, even light.",
            should_retake=True,
        )

    if quality.contrast < 28:
        rgb = np.asarray(image.convert("RGB"))
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        luminance, channel_a, channel_b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        adjusted = cv2.merge((clahe.apply(luminance), channel_a, channel_b))
        adjusted_rgb = cv2.cvtColor(adjusted, cv2.COLOR_LAB2RGB)
        return Image.fromarray(adjusted_rgb), PreparationRoute(
            name="mild_contrast",
            message="The contrast is low, so the prototype applied a gentle luminance adjustment. Check the adjusted image before continuing.",
            applied=True,
        )

    return image, PreparationRoute(
        name="original",
        message="No image adjustment was needed. The original photo will be kept as uploaded.",
    )
