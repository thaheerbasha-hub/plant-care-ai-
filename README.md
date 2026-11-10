# PlantCare AI

A website-only Python prototype for checking leaf-photo quality and preparing for plant-disease screening.

## What works in this prototype

- Upload JPG, PNG, or WebP leaf photos up to 10 MB.
- Review image size, brightness, contrast, blur, and exposure warnings.
- Keep a usable photo unchanged, apply a mild contrast adjustment when the contrast check is low, or ask for a retake when the photo is very blurry or badly exposed.
- Show clear project and privacy information.
- Load an optional EfficientNet-B0 classifier when compatible trained weights and a class-label file are provided.

The image-quality checks are simple project heuristics, not validated agronomic measures. No trained model or disease-classification result is included. The page will say so instead of displaying a made-up prediction.

## Run it locally

1. Install Python 3.10 or newer.
2. Create and activate a virtual environment.
3. Install the website packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the site:

   ```bash
   streamlit run app.py
   ```

To enable the optional model loader, install `requirements-model.txt` and add both files below:

- `models/plantcare_effnet_b0.pth` - an EfficientNet-B0 state dictionary trained for this project.
- `models/class_labels.json` - a JSON list of class names in the exact order used during training.

The model files are ignored by Git because trained weights can be large and their provenance and usage terms need to be checked before sharing. A compatible model must be trained and validated before the classifier should be used for screening.

## Is TypeScript useful here?

Not for the first version. Streamlit builds the interface in Python, which keeps this intermediate project focused on Python and image processing. TypeScript would be useful later if the project moves to a custom React frontend with richer interactions. That would add a second app and a separate API, so this prototype keeps the stack small.

## Research status and limits

This repository contains a working photo-quality website prototype and an optional model-loading interface. It does not contain PlantVillage or PlantDoc images, trained model weights, validated thresholds, or experimental results. The proposed quality routing and disease-screening approach needs evaluation before any reliability claim is made.

PlantCare AI is a learning prototype. It is not a confirmed plant diagnosis or treatment recommendation. For decisions that affect a crop, consult a local agricultural expert.
