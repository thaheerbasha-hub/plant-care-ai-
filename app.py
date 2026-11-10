from __future__ import annotations

from io import BytesIO

import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.image_quality import assess_image, prepare_image
from src.predictor import format_label, load_predictor, PlantCareInfo


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
SUPPORTED_TYPES = ["jpg", "jpeg", "png", "webp"]


st.set_page_config(
    page_title="PlantCare AI | Plant Identification & Care",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
      color-scheme: light;
      --forest:#173c32; --leaf:#2e7057; --mint:#e9f2e9;
      --cream:#f7f8f3; --ink:#18251f; --muted:#52645a;
    }
    html, body, .stApp {
      background:var(--cream) !important;
      color:var(--ink) !important;
      font-family:'Segoe UI',Arial,sans-serif !important;
      font-size:16px;
    }
    [data-testid="stAppViewContainer"], [data-testid="stMain"] { color:var(--ink) !important; }
    [data-testid="stMarkdownContainer"] { color:var(--ink) !important; }
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stWidgetLabel"] *, label { color:var(--ink) !important; line-height:1.6; }
    h1,h2,h3,h4 { font-family:'Segoe UI',Arial,sans-serif !important; color:var(--forest) !important; font-weight:700 !important; }
    [data-testid="stSidebar"] { background:#eaf1e9 !important; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *,
    [data-testid="stSidebar"] [data-testid="stRadio"] label,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * { color:#18382b !important; }
    [data-testid="stSidebar"] [data-testid="stRadio"] input { accent-color:var(--leaf); }
    [data-testid="stHeader"], [data-testid="stToolbar"] { background:var(--cream) !important; }
    [data-testid="stCaptionContainer"], .stCaption { color:var(--muted) !important; }
    [data-testid="stFileUploaderDropzone"] {
      background:#fff !important; border:1px dashed #718879 !important;
    }
    [data-testid="stFileUploaderDropzone"] * { color:#20342a !important; }
    [data-testid="stFileUploaderDropzone"] button {
      background:#e7f0e9 !important; color:#174e39 !important;
      border:1px solid #769382 !important; font-weight:700 !important;
    }
    .hero { background:linear-gradient(115deg,#173c32,#2d7057); border-radius:24px; padding:32px 36px; color:#fff; margin:8px 0 24px; }
    .hero h1 { color:#fff !important; font-size:2.2rem; margin:0 0 8px; }
    .hero p { color:#f0f6f1 !important; font-size:1rem; margin:0; max-width:760px; }
    .eyebrow { text-transform:uppercase; letter-spacing:.13em; font-size:.72rem; font-weight:700; color:#d0e8d4; margin-bottom:12px; }
    .metric-card { background:white; border:1px solid #e5e9df; border-radius:16px; padding:16px 18px; min-height:90px; }
    .metric-label { color:#64736b; font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; }
    .metric-value { color:var(--forest); font-family:'Manrope',sans-serif; font-size:1.2rem; font-weight:800; margin-top:6px; }
    .soft-card { background:white; border:1px solid #e5e9df; border-radius:18px; padding:20px 22px; }
    .status-pill { display:inline-block; background:#e2efe4; color:#235640; padding:6px 10px; border-radius:999px; font-weight:700; font-size:.78rem; }
    .small-note { color:#52645a !important; font-size:.9rem; }
    div.stButton > button { border-radius:12px; border:0; background:#235640; color:white !important; font-weight:700; }
    div.stButton > button:hover { background:#173c32; color:white !important; }
    .care-section { background:white; border:1px solid #e5e9df; border-radius:16px; padding:20px; margin:16px 0; }
    .care-title { color:var(--forest); font-size:1.1rem; font-weight:700; margin-bottom:12px; }
    .care-item { margin:8px 0; }
    .care-label { color:#64736b; font-size:.85rem; font-weight:600; }
    .care-value { color:var(--ink); margin-top:2px; }
    .characteristic-pill { display:inline-block; background:#e9f2e9; color:#173c32; padding:4px 10px; border-radius:999px; font-size:.8rem; font-weight:600; margin:4px 4px 4px 0; }
    .characteristic-pill.warning { background:#fef3e2; color:#b45309; }
    </style>
    """,
    unsafe_allow_html=True,
)


def metric_card(label: str, value: str) -> str:
    return (
        '<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        "</div>"
    )


def render_care_info(care: PlantCareInfo) -> None:
    """Render plant care information in a nice format."""
    st.markdown("#### 🌱 Plant Care Guide")

    # Scientific name
    st.markdown(
        f'<div class="care-section"><div class="care-title">Scientific Name</div>'
        f'<div class="care-item"><span class="care-value">{care.scientific_name}</span></div></div>',
        unsafe_allow_html=True,
    )

    # Watering schedule
    ws = care.watering_schedule
    if ws:
        st.markdown(
            f'<div class="care-section"><div class="care-title">💧 Watering Schedule</div>'
            f'<div class="care-item"><span class="care-label">Frequency:</span> '
            f'<span class="care-value">{ws.get("frequency", "Not specified")}</span></div>'
            f'<div class="care-item"><span class="care-label">Amount:</span> '
            f'<span class="care-value">{ws.get("amount", "Not specified")}</span></div>'
            f'<div class="care-item"><span class="care-label">Notes:</span> '
            f'<span class="care-value">{ws.get("notes", "Not specified")}</span></div></div>',
            unsafe_allow_html=True,
        )

    # Care instructions
    ci = care.care_instructions
    if ci:
        st.markdown(
            f'<div class="care-section"><div class="care-title">☀️ Care Instructions</div>'
            f'<div class="care-item"><span class="care-label">Light:</span> '
            f'<span class="care-value">{ci.get("light", "Not specified")}</span></div>'
            f'<div class="care-item"><span class="care-label">Temperature:</span> '
            f'<span class="care-value">{ci.get("temperature", "Not specified")}</span></div>'
            f'<div class="care-item"><span class="care-label">Humidity:</span> '
            f'<span class="care-value">{ci.get("humidity", "Not specified")}</span></div>'
            f'<div class="care-item"><span class="care-label">Fertilizer:</span> '
            f'<span class="care-value">{ci.get("fertilizer", "Not specified")}</span></div>'
            f'<div class="care-item"><span class="care-label">Pruning:</span> '
            f'<span class="care-value">{ci.get("pruning", "Not specified")}</span></div></div>',
            unsafe_allow_html=True,
        )

    # Characteristics
    ch = care.characteristics
    if ch:
        pills = []
        if ch.get("toxicToPets"):
            pills.append('<span class="characteristic-pill warning">⚠️ Toxic to Pets</span>')
        else:
            pills.append('<span class="characteristic-pill">✅ Pet Safe</span>')

        if ch.get("airPurifying"):
            pills.append('<span class="characteristic-pill">🌬️ Air Purifying</span>')

        if ch.get("flowering"):
            pills.append('<span class="characteristic-pill">🌸 Flowering</span>')

        difficulty = ch.get("difficulty", "Not specified")
        pills.append(f'<span class="characteristic-pill">📊 Difficulty: {difficulty}</span>')

        st.markdown(
            f'<div class="care-section"><div class="care-title">🌿 Characteristics</div>'
            f'<div class="care-item">{"".join(pills)}</div></div>',
            unsafe_allow_html=True,
        )


def show_upload_page() -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">PlantCare AI · Research prototype</div>
          <h1>Identify your houseplant & get care guidance</h1>
          <p>Upload a photo of a houseplant for species identification and detailed care instructions. Results are preliminary — confirm with a local expert for critical decisions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Start with a plant photo")
    uploaded = st.file_uploader(
        "Choose a clear photo of a houseplant",
        type=SUPPORTED_TYPES,
        help="JPG, PNG, or WebP. The prototype accepts files up to 10 MB.",
    )
    st.caption("For best results, photograph a single plant in good lighting, filling most of the frame.")
    if not uploaded:
        left, right = st.columns([1.1, 0.9], gap="large")
        with left:
            st.markdown(
                '<div class="soft-card"><strong>What happens next?</strong><br>'
                '<span class="small-note">The site checks image quality (blur, brightness, contrast). '
                'A ViT-based model identifies the plant from 47 common houseplant species. '
                'Detailed care instructions are provided from a curated database.</span></div>',
                unsafe_allow_html=True,
            )
        with right:
            st.markdown(
                '<div class="soft-card"><span class="status-pill">Website prototype</span>'
                '<h3 style="margin-top:16px">Photo quality first</h3>'
                '<p class="small-note">A classifier can be wrong when a photo is blurry, poorly lit, or outside its training data. PlantCare AI shows those limits instead of hiding them.</p>'
                '</div>',
                unsafe_allow_html=True,
            )
        st.markdown("#### How this prototype works")
        st.markdown(
            "1. Check image size, brightness, contrast, and blur\n"
            "2. Identify plant species from 47 common houseplants\n"
            "3. Display detailed care instructions (watering, light, humidity, etc.)\n"
            "4. Treat every model score as uncalibrated until validated"
        )
        return

    file_bytes = uploaded.getvalue()
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        st.error("This photo is over 10 MB. Please choose a smaller image.")
        return
    try:
        original = Image.open(BytesIO(file_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        st.error("We couldn't open that file as an image. Please try a JPG, PNG, or WebP photo.")
        return
    if min(original.size) < 64:
        st.error("This image is too small to inspect. Please choose a larger photo.")
        return

    quality = assess_image(original)
    prepared, route = prepare_image(original, quality)
    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        st.image(original, caption="Your uploaded photo", use_container_width=True)

    with right:
        st.subheader("Photo check")
        st.markdown(
            '<span class="status-pill">Prototype quality checks · not validated</span>',
            unsafe_allow_html=True,
        )
        st.write(quality.summary)
        metrics = [
            ("Image size", f"{quality.width} × {quality.height}"),
            ("Brightness", f"{quality.brightness:.0f} / 255"),
            ("Contrast spread", f"{quality.contrast:.0f}"),
            ("Blur estimate", f"{quality.blur_score:.0f}"),
        ]
        for start in (0, 2):
            metric_columns = st.columns(2)
            for column, (label, value) in zip(metric_columns, metrics[start : start + 2]):
                column.markdown(metric_card(label, value), unsafe_allow_html=True)
        st.markdown("#### Suggested image handling")
        st.info(route.message)
        if route.should_retake:
            st.warning("A new photo is recommended for more reliable identification.")
        if route.applied:
            with st.expander("View the gently adjusted image"):
                st.image(prepared, caption="Prototype contrast adjustment", use_container_width=True)

        st.markdown("#### Plant Identification")
        with st.spinner("Loading the plant classifier. First use downloads the model file (~87 MB)."):
            predictor, predictor_message = load_predictor()
        if predictor is None:
            st.warning(predictor_message)
        elif route.should_retake:
            st.info("The classifier is ready, but this photo should be retaken for best results.")
        else:
            try:
                result = predictor.predict(prepared)
                if result.candidates:
                    top = result.candidates[0]
                    st.success(f"Identified: **{format_label(top.label)}** · model score {top.score:.1%}")
                    st.caption("Raw model score, not calibrated confidence. A match is not a guaranteed identification.")
                    if len(result.candidates) > 1:
                        st.write("Other possibilities:")
                        for item in result.candidates[1:]:
                            st.write(f"- {format_label(item.label)}: {item.score:.1%}")

                    # Display care information if available
                    if result.plant_care_info:
                        render_care_info(result.plant_care_info)
                    else:
                        st.info("No detailed care information available for this species.")
                else:
                    st.error("Could not identify the plant. Try a clearer photo.")
            except Exception as exc:
                st.error("The model could not analyze this image.")
                st.caption(f"Technical detail: {exc}")

    st.divider()
    st.caption(
        "PlantCare AI is a research prototype. It does not provide guaranteed identification or professional care advice. "
        "For plant health decisions, consult a local horticultural expert."
    )
    st.caption("Classifier: [House Plant Image Detection (ViT)](https://huggingface.co/dima806/house-plant-image-detection) · Apache-2.0. Plant care data from PlantPal database (48 species). Field performance is unknown.")


def show_about_page() -> None:
    st.markdown("<div class='hero'><div class='eyebrow'>Project overview</div><h1>How PlantCare AI works</h1><p>A website-only Python project for houseplant identification with honest uncertainty.</p></div>", unsafe_allow_html=True)
    st.markdown("### The idea")
    st.write("PlantCare AI identifies common houseplants from photos and provides detailed care instructions. Before classification, the prototype checks whether the image is bright and sharp enough to use. If the photo is poor, it asks for a retake instead of presenting a confident-looking guess.")
    st.markdown("### Technical approach")
    st.write(
        "- **Image quality checks**: blur (Laplacian variance), brightness, contrast, clipping\n"
        "- **Model**: Vision Transformer (ViT-B/16) fine-tuned on houseplant species, exported to ONNX INT8\n"
        "- **Classes**: 47 common houseplant species\n"
        "- **Care database**: Curated care data for each species (watering, light, humidity, toxicity, etc.)\n"
        "- **Local inference**: Model downloads once (~87 MB), then runs entirely on your machine"
    )
    st.markdown("### Current status")
    st.info("Image quality checks and 47-class plant identification are connected. The model downloads on first use. Displayed scores are not calibrated, and this project has not independently validated accuracy or field performance.")
    st.markdown("### Privacy")
    st.write("Images are processed for the current page and are not saved by this prototype. Avoid uploading photos that contain personal information.")
    st.markdown("### Data sources")
    st.write(
        "- **Model**: [dima806/house-plant-image-detection](https://huggingface.co/dima806/house-plant-image-detection) (ViT fine-tuned on houseplants)\n"
        "- **ONNX export**: [onnx-community/house-plant-image-detection-ONNX](https://huggingface.co/onnx-community/house-plant-image-detection-ONNX)\n"
        "- **Care data**: PlantPal database (48 species with detailed care instructions)"
    )


with st.sidebar:
    st.markdown("## 🌿 PlantCare AI")
    page = st.radio("Navigate", ["Identify plant", "About the project"], label_visibility="collapsed")
    st.divider()
    st.caption("Website-only · Python prototype")
    st.caption("47 houseplant species · Local ONNX inference")

if page == "Identify plant":
    show_upload_page()
else:
    show_about_page()