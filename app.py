from __future__ import annotations

from io import BytesIO

import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.image_quality import assess_image, prepare_image
from src.predictor import format_label, load_predictor


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
SUPPORTED_TYPES = ["jpg", "jpeg", "png", "webp"]

st.set_page_config(
    page_title="PlantCare AI | Leaf photo screening",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root { --forest:#173c32; --leaf:#2e7057; --mint:#e9f2e9; --cream:#f7f7f1; --ink:#1b2924; }
    .stApp { background:var(--cream); color:var(--ink); font-family:'DM Sans',sans-serif; }
    h1,h2,h3 { font-family:'Manrope',sans-serif !important; color:var(--forest); }
    [data-testid="stSidebar"] { background:#eef3eb; }
    .hero { background:linear-gradient(115deg,#173c32,#2d7057); border-radius:24px; padding:32px 36px; color:#fff; margin:8px 0 24px; }
    .hero h1 { color:#fff !important; font-size:2.2rem; margin:0 0 8px; }
    .hero p { color:#e3eee5; font-size:1rem; margin:0; max-width:760px; }
    .eyebrow { text-transform:uppercase; letter-spacing:.13em; font-size:.72rem; font-weight:700; color:#b9d9be; margin-bottom:12px; }
    .metric-card { background:white; border:1px solid #e5e9df; border-radius:16px; padding:16px 18px; min-height:90px; }
    .metric-label { color:#64736b; font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; }
    .metric-value { color:var(--forest); font-family:'Manrope',sans-serif; font-size:1.2rem; font-weight:800; margin-top:6px; }
    .soft-card { background:white; border:1px solid #e5e9df; border-radius:18px; padding:20px 22px; }
    .status-pill { display:inline-block; background:#e2efe4; color:#235640; padding:6px 10px; border-radius:999px; font-weight:700; font-size:.78rem; }
    .small-note { color:#637169; font-size:.86rem; }
    div.stButton > button { border-radius:12px; border:0; background:#235640; color:white; font-weight:700; }
    div.stButton > button:hover { background:#173c32; color:white; }
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


def show_upload_page() -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">PlantCare AI · Research prototype</div>
          <h1>A clearer first look at leaf health.</h1>
          <p>Upload a leaf photo to check image quality and explore the proposed screening workflow. Disease predictions stay disabled until a trained model is added.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Start with a leaf photo")
    uploaded = st.file_uploader(
        "Choose a clear photo of one leaf",
        type=SUPPORTED_TYPES,
        help="JPG, PNG, or WebP. The prototype accepts files up to 10 MB.",
    )
    st.caption("For the most useful quality check, keep the leaf in focus and fill a good part of the frame.")
    if not uploaded:
        left, right = st.columns([1.1, 0.9], gap="large")
        with left:
            st.markdown(
                '<div class="soft-card"><strong>What happens next?</strong><br>'
                '<span class="small-note">The site checks blur, brightness, contrast, and image size. It may suggest a retake or apply a mild contrast adjustment. A disease model can be connected later.</span></div>',
                unsafe_allow_html=True,
            )
        with right:
            st.markdown(
                '<div class="soft-card"><span class="status-pill">Website prototype</span>'
                '<h3 style="margin-top:16px">Photo quality first</h3>'
                '<p class="small-note">A classifier can be wrong when a photo is blurry, poorly lit, or outside its training data. PlantCare AI is designed to show those limits instead of hiding them.</p>'
                '</div>',
                unsafe_allow_html=True,
            )
        st.markdown("#### Planned steps")
        st.markdown("1. Check the uploaded image\n2. Suggest a retake or a gentle adjustment\n3. Run a trained classifier when model weights are available\n4. Add calibrated confidence and an explanation map as later project steps")
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
            st.warning("A new photo is recommended before any disease screening.")
        if route.applied:
            with st.expander("View the gently adjusted image"):
                st.image(prepared, caption="Prototype contrast adjustment", use_container_width=True)

        st.markdown("#### Disease screening")
        with st.spinner("Loading the plant classifier. First use downloads the model file (~24 MB)."):
            predictor, predictor_message = load_predictor()
        if predictor is None:
            st.warning(predictor_message)
        elif route.should_retake:
            st.info("The classifier is ready, but this photo needs to be retaken first.")
        else:
            try:
                result = predictor.predict(prepared)
                if not result.in_scope:
                    st.error("This photo does not look enough like the single-leaf images supported by the model. The disease guesses would be unreliable.")
                    st.caption(f"Out-of-scope similarity: {result.similarity:.3f} (minimum {predictor.threshold:.3f}). Try one clear leaf on a plain background.")
                elif result.candidates:
                    top = result.candidates[0]
                    st.success(f"Possible match: **{format_label(top.label)}** · model score {top.score:.1%}")
                    st.caption(f"Raw model score, not calibrated confidence. Image similarity check: {result.similarity:.3f}. A match is not a confirmed diagnosis.")
                    if len(result.candidates) > 1:
                        st.write("Other model candidates")
                        for item in result.candidates[1:]:
                            st.write(f"- {format_label(item.label)}: {item.score:.1%}")
            except Exception as exc:  # keep the interface usable if optional weights are incompatible
                st.error("The model could not analyze this image. Check the model file and label list.")
                st.caption(f"Technical detail: {exc}")

    st.divider()
    st.caption(
        "PlantCare AI is a student research prototype. It does not provide a confirmed diagnosis or treatment advice. "
        "For plant health decisions, consult a local agricultural expert."
    )
    st.caption("Classifier: [DINOv2 Plant Disease model](https://huggingface.co/rodynaemad/plant-disease-dinov2-small) · CC BY-SA 4.0. Its author reports a centroid similarity check for out-of-scope images. Field performance is unknown.")


def show_about_page() -> None:
    st.markdown("<div class='hero'><div class='eyebrow'>Project overview</div><h1>How PlantCare AI is planned to work</h1><p>A website-only Python project about image quality, careful classification, and honest uncertainty.</p></div>", unsafe_allow_html=True)
    st.markdown("### The idea")
    st.write("PlantCare AI is designed to examine a photo of a leaf and screen for a supported disease class. Before classification, the prototype checks whether the image is bright and sharp enough to use. If the photo is poor, it can ask for a retake instead of presenting a confident-looking guess.")
    st.markdown("### Research plan")
    st.write("The proposed study uses PlantVillage for model training and PlantDoc for a separate test on more varied images. Only classes that match clearly across the two datasets will be compared. The study will measure classification quality, confidence calibration, how often the system abstains, and response time.")
    st.markdown("### Current status")
    st.info("The website can inspect uploaded photo quality. A disease classifier is optional and needs trained model weights. No disease-detection accuracy or research result is claimed by this prototype.")
    st.markdown("### Privacy")
    st.write("Images are processed for the current page and are not saved by this prototype. Avoid uploading photos that contain personal information.")


with st.sidebar:
    st.markdown("## 🌿 PlantCare AI")
    page = st.radio("Navigate", ["Photo check", "About the project"], label_visibility="collapsed")
    st.divider()
    st.caption("Website-only · Python prototype")

if page == "Photo check":
    show_upload_page()
else:
    show_about_page()
