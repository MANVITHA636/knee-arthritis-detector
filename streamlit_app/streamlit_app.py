import streamlit as st
import torch
torch.set_num_threads(1)
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import io
from datetime import datetime
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from transformers import CLIPModel, CLIPProcessor

st.set_page_config(page_title="ArthroScan AI", page_icon="🦵", layout="centered")

import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pth")
IMAGE_SIZE = 224
SEVERITY_LABELS = {
    "0": "Grade 0 - Normal", "1": "Grade 1 - Doubtful", "2": "Grade 2 - Mild",
    "3": "Grade 3 - Moderate", "4": "Grade 4 - Severe",
}
CONFIDENCE_THRESHOLD = 65.0
MARGIN_THRESHOLD = 20.0

# ---- OOD gate settings ----
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
KNEE_PROMPTS = [
    "a thermal image of a human knee joint",
    "an x-ray image of a human knee joint",
]
NON_KNEE_PROMPTS = [
    "a photo of a human face",
    "a screenshot of a computer or phone screen",
    "a photo of a random object, animal, or scene unrelated to a knee",
]
CLIP_PROMPTS = KNEE_PROMPTS + NON_KNEE_PROMPTS
KNEE_SCORE_THRESHOLD = 0.55  # combined probability mass on knee prompts required to pass

st.markdown("""
<style>
.header-bar { display: flex; align-items: center; gap: 12px; padding: 14px 18px;
    background: #F4F8FC; border-radius: 12px; border: 1px solid #D6E6F5; margin-bottom: 18px; }
.header-title { margin: 0; font-weight: 600; font-size: 17px; color: #0C447C; }
.header-sub { margin: 0; font-size: 13px; color: #5F6B76; }
.step-bar { display: flex; align-items: center; gap: 8px; margin-bottom: 18px; }
.step-circle-active { width: 24px; height: 24px; border-radius: 50%; background: #185FA5;
    color: white; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 600; }
.step-circle-done { width: 24px; height: 24px; border-radius: 50%; background: #2E8B57;
    color: white; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 600; }
.step-circle-inactive { width: 24px; height: 24px; border-radius: 50%; background: white;
    border: 1px solid #B5D4F4; color: #5F6B76; display: flex; align-items: center; justify-content: center; font-size: 12px; }
.step-label-active { font-size: 13px; font-weight: 600; color: #0C447C; }
.step-label-inactive { font-size: 13px; color: #5F6B76; }
.step-line { flex: 1; height: 1px; background: #D6E6F5; }
.disclaimer-banner { background: #E6F1FB; border-radius: 8px; padding: 10px 14px; margin-bottom: 18px; }
.disclaimer-text { margin: 0; font-size: 12.5px; color: #0C447C; }
.error-card { background: #FDECEA; border: 1px solid #E5484D; border-radius: 12px; padding: 18px; margin-top: 12px; }
.error-title { margin: 0 0 6px 0; font-size: 15px; font-weight: 700; color: #B3261E; }
.error-text { margin: 0; font-size: 13px; color: #7A1E19; }
.report-card { background: white; border: 1px solid #D6E6F5; border-radius: 12px; padding: 20px; margin-top: 12px; }
.report-heading { margin: 0 0 14px 0; font-size: 17px; font-weight: 700; color: #0C447C;
    border-bottom: 1px solid #D6E6F5; padding-bottom: 10px; }
.report-row { display: flex; padding: 4px 0; font-size: 13.5px; }
.report-key { width: 140px; color: #5F6B76; }
.report-val { color: #1C2733; font-weight: 500; }
.report-grade { font-size: 19px; font-weight: 700; color: #185FA5; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-bar">
    <span style="font-size: 22px;">🦵</span>
    <div>
        <p class="header-title">ArthroScan AI</p>
        <p class="header-sub">Knee osteoarthritis severity screening</p>
    </div>
</div>
""", unsafe_allow_html=True)

if "step" not in st.session_state:
    st.session_state.step = 1
if "name" not in st.session_state:
    st.session_state.name = ""
if "age" not in st.session_state:
    st.session_state.age = None
if "sex" not in st.session_state:
    st.session_state.sex = None
if "symptoms" not in st.session_state:
    st.session_state.symptoms = ""
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

def render_steps(active):
    labels = ["Info", "Upload", "Result"]
    html = '<div class="step-bar">'
    for i, lab in enumerate(labels):
        n = i + 1
        if n < active:
            html += f'<div style="display:flex;align-items:center;gap:6px;"><div class="step-circle-done">✓</div><span class="step-label-active">{lab}</span></div>'
        elif n == active:
            html += f'<div style="display:flex;align-items:center;gap:6px;"><div class="step-circle-active">{n}</div><span class="step-label-active">{lab}</span></div>'
        else:
            html += f'<div style="display:flex;align-items:center;gap:6px;"><div class="step-circle-inactive">{n}</div><span class="step-label-inactive">{lab}</span></div>'
        if i < len(labels) - 1:
            html += '<div class="step-line"></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

render_steps(st.session_state.step)

st.markdown("""
<div class="disclaimer-banner">
    <p class="disclaimer-text">Research proof-of-concept only. Trained specifically on knee joint
    X-ray/thermal-style images. Not a substitute for professional medical diagnosis.</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    device = torch.device("cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    class_names = checkpoint["class_names"]
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device); model.eval()
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False
    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
    return model, cam, class_names, device

@st.cache_resource
def load_clip():
    device = torch.device("cpu")
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    clip_model.eval()
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    return clip_model, clip_processor, device

def is_knee_image(pil_img):
    """Zero-shot OOD gate: returns (passed, knee_score, best_label)."""
    clip_model, clip_processor, clip_device = load_clip()
    inputs = clip_processor(text=CLIP_PROMPTS, images=pil_img, return_tensors="pt", padding=True)
    inputs = {k: v.to(clip_device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]
    knee_score = probs[:len(KNEE_PROMPTS)].sum().item()
    best_label = CLIP_PROMPTS[torch.argmax(probs).item()]
    return knee_score >= KNEE_SCORE_THRESHOLD, knee_score, best_label

model, cam, class_names, device = load_model()
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ================= STEP 1: INFO =================
if st.session_state.step == 1:
    st.subheader("Knee Arthritis - Patient Info")
    st.session_state.name = st.text_input("Name", value=st.session_state.name, placeholder="Enter your name")
    st.session_state.age = st.number_input(
        "Age", min_value=1, max_value=120, value=st.session_state.age, placeholder="Enter your age"
    )
    st.session_state.sex = st.radio(
        "Sex", ["Male", "Female", "Other"], horizontal=True, index=None,
    ) if st.session_state.sex is None else st.radio(
        "Sex", ["Male", "Female", "Other"], horizontal=True, index=["Male", "Female", "Other"].index(st.session_state.sex)
    )
    st.write("")
    if st.button("Continue →", type="primary", use_container_width=True):
        if st.session_state.name.strip() == "":
            st.warning("Please enter your name to continue.")
        elif st.session_state.age is None:
            st.warning("Please enter your age to continue.")
        elif st.session_state.sex is None:
            st.warning("Please select your sex to continue.")
        else:
            st.session_state.step = 2
            st.rerun()

# ================= STEP 2: UPLOAD =================
elif st.session_state.step == 2:
    st.subheader("Knee Arthritis - Symptoms & Image")
    st.session_state.symptoms = st.text_area(
        "What are your symptoms?",
        value=st.session_state.symptoms,
        placeholder="Type your symptoms (e.g. knee pain, stiffness, swelling)...",
    )
    uploaded_file = st.file_uploader("Upload your knee image", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded_file is not None:
        st.session_state.uploaded_image = uploaded_file.getvalue()

    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Back", use_container_width=True):
            st.session_state.step = 1
            st.rerun()
    with col_next:
        if st.button("Continue →", type="primary", use_container_width=True):
            if st.session_state.uploaded_image is None:
                st.warning("Please upload a knee image to continue.")
            else:
                st.session_state.step = 3
                st.rerun()

# ================= STEP 3: RESULT =================
elif st.session_state.step == 3:
    st.subheader("Result")

    pil_img = Image.open(io.BytesIO(st.session_state.uploaded_image)).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    rgb_img = np.array(pil_img).astype(np.float32) / 255.0
    input_tensor = transform(pil_img).unsqueeze(0)

    with st.spinner("Checking image type..."):
        ood_passed, knee_score, best_label = is_knee_image(pil_img)

    col1, col2 = st.columns(2)
    with col1:
        st.image(pil_img, caption="Uploaded image", use_container_width=True)

    report_text = None

    if not ood_passed:
        with col2:
            st.markdown("""
            <div class="error-card">
                <p class="error-title">❌ Error</p>
                <p class="error-text">This is not a knee thermal image. Please upload the knee thermal image only.</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        with st.spinner("Analyzing image..."):
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                pred_idx = torch.argmax(probs).item()
                confidence = probs[pred_idx].item() * 100

        sorted_probs = torch.sort(probs, descending=True).values
        margin = (sorted_probs[0] - sorted_probs[1]).item() * 100
        is_valid = confidence >= CONFIDENCE_THRESHOLD and margin >= MARGIN_THRESHOLD

        if not is_valid:
            with col2:
                st.markdown("""
                <div class="error-card">
                    <p class="error-title">❌ Error</p>
                    <p class="error-text">The model isn't confident about this image. Please upload a clearer knee thermal image.</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            label = SEVERITY_LABELS.get(class_names[pred_idx], class_names[pred_idx])
            visualization = None
            try:
                grayscale_cam = cam(input_tensor=input_tensor)[0]
                visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            except Exception:
                visualization = None

            with col2:
                if visualization is not None:
                    st.image(visualization, caption="Grad-CAM heatmap", use_container_width=True)
                else:
                    st.info("Heatmap temporarily unavailable, but the prediction below is still valid.")

            # ---------- FORMATTED PATIENT REPORT CARD ----------
            st.markdown(f"""
            <div class="report-card">
                <p class="report-heading">🦵 ArthroScan AI - Screening Report</p>
                <div class="report-row"><span class="report-key">Name</span><span class="report-val">{st.session_state.name}</span></div>
                <div class="report-row"><span class="report-key">Age</span><span class="report-val">{st.session_state.age}</span></div>
                <div class="report-row"><span class="report-key">Sex</span><span class="report-val">{st.session_state.sex}</span></div>
                <div class="report-row"><span class="report-key">Symptoms</span><span class="report-val">{st.session_state.symptoms or 'None provided'}</span></div>
                <div class="report-row" style="margin-top:10px;"><span class="report-key">Predicted Grade</span><span class="report-grade">{label}</span></div>
                <div class="report-row"><span class="report-key">Confidence</span><span class="report-val">{confidence:.1f}%</span></div>
            </div>
            """, unsafe_allow_html=True)

            st.write("")
            st.write("**All class probabilities**")
            for i in range(len(class_names)):
                cls_label = SEVERITY_LABELS.get(class_names[i], class_names[i])
                st.progress(float(probs[i]), text=f"{cls_label}: {probs[i].item()*100:.1f}%")

            report_text = (
                f"ArthroScan AI - Knee Osteoarthritis Screening Report\n"
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                f"{'-'*50}\n"
                f"Name: {st.session_state.name}\n"
                f"Age: {st.session_state.age}\n"
                f"Sex: {st.session_state.sex}\n"
                f"Symptoms: {st.session_state.symptoms or 'None provided'}\n"
                f"{'-'*50}\n"
                f"Predicted Grade: {label}\n"
                f"Confidence: {confidence:.1f}%\n\n"
                f"All class probabilities:\n"
            )
            for i in range(len(class_names)):
                cls_label = SEVERITY_LABELS.get(class_names[i], class_names[i])
                report_text += f"  {cls_label}: {probs[i].item()*100:.1f}%\n"
            report_text += (
                f"\n{'-'*50}\n"
                f"This is a research proof-of-concept only and is not a substitute for "
                f"professional medical diagnosis.\n"
            )

    st.write("")
    col_back2, col_report = st.columns(2)
    with col_back2:
        if st.button("← Start Over", use_container_width=True):
            st.session_state.step = 1
            st.session_state.uploaded_image = None
            st.session_state.age = None
            st.session_state.sex = None
            st.rerun()
    with col_report:
        if report_text:
            st.download_button(
                "⬇ Download Report", data=report_text,
                file_name=f"arthroscan_report_{st.session_state.name.replace(' ', '_')}.txt",
                mime="text/plain", use_container_width=True,
            )
