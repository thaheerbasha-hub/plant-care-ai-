# PlantCare AI

A local, website-only Python prototype for leaf-photo quality checks and plant-disease screening.

## What the site does

- Accepts JPG, PNG, or WebP photos up to 10 MB.
- Checks image size, brightness, contrast, blur, and exposure.
- Keeps usable photos unchanged, applies a mild contrast adjustment for low-contrast images, and recommends a retake for very blurry or badly exposed photos.
- Runs a 38-class image classifier locally on the computer after its quantized ONNX model downloads on first use (about 24 MB).
- Checks whether the image resembles the single-leaf images the model supports and withholds disease guesses when it does not.
- Shows the top model matches, image-quality notes, and a clear research-prototype disclaimer.
- Does not save uploaded images.

The model is trained and evaluated on PlantVillage-style images. Its model card reports held-out PlantVillage results but warns that field performance can be substantially lower. The out-of-scope similarity threshold is provided by the model author and has not been independently checked here. The score shown by this website is a raw model score, not calibrated confidence. Confirm any result with an agricultural expert; do not use this site to decide treatment.

## Run locally

1. Install Python 3.10 or newer.
2. Open PowerShell in this folder and create a virtual environment. If the Python launcher is installed, use:

   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If `py -3.12` is not recognized, use the path to your Python 3.10+ executable. On this computer:

   ```powershell
   & 'C:\Users\thahe\AppData\Local\Programs\Python\Python312\python.exe' -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

4. Start the site:

   ```powershell
   streamlit run app.py
   ```

5. Open the local address shown by Streamlit, usually `http://localhost:8501`.

The computer needs internet access the first time a photo is submitted so the model and its metadata can be downloaded. After that, the files are cached in `models/` and inference runs locally. The model cache is ignored by Git.

## Is TypeScript useful here?

Not for this version. Streamlit builds the whole interface in Python, which keeps the project focused on Python and image processing. TypeScript would make sense later if the project moved to a custom React frontend with a separate API.

## Research and model attribution

This prototype uses the [DINOv2 Plant Disease model](https://huggingface.co/rodynaemad/plant-disease-dinov2-small), shared under CC BY-SA 4.0. Its own model card describes the dataset, reported evaluation, intended use, and limitations. The image-quality thresholds in this project are simple heuristics and have not been validated. This repository does not contain PlantVillage or PlantDoc images, does not claim its own accuracy, and does not include invented experimental results.
