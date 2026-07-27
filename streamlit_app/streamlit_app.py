import streamlit as st
import torch
torch.set_num_threads(1)
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

st.set_page_config(page_title="ArthroScan AI", page_icon="🦵", layout="centered")

MODEL_PATH = "model.pth"
IMAGE_SIZE = 224
SEVERITY_LABELS = {
    "0": "Grade 0 - Normal", "1": "Grade 1 - Doubtful", "2": "Grade 2 - Mild",
    "3": "Grade 3 - Moderate", "4": "Grade 4 - Severe",
}

# Two-part safeguard thresholds (heuristic, not a trained OOD detector):
CONFIDENCE_THRESHOLD = 65.0   # top prediction must be at least this confident
MARGIN_THRESHOLD = 20.0       # gap between top-1 and top-2 must be at least this

st.markdown("""
<style>
.header-bar { display: flex; align-items: center; gap: 12px; padding: 14px 18px;
    background: #F4F8FC; border-radius: 12px; border: 1px solid #D6E6F5; margin-bottom: 18px; }
.header-title { margin: 0; font-weight: 600; font-size: 17px; color: #0C447C; }
.header-sub { margin: 0; font-size: 13px; color: #5F6B76; }
.disclaimer-banner { background: #E6F1FB; border-radius: 8px; padding: 10px 14px; margin-bottom: 18px; }
.disclaimer-text { margin: 0; font-size: 12.5px; color: #0C447C; }
.result-card { background: white; border: 1px solid #D6E6F5; border-radius: 12px; padding: 18px; margin-top: 12px; }
.result-label { margin: 0 0 8px 0; font-size: 13px; color: #5F6B76; }
.result-grade { font-size: 21px; font-weight: 600; color: #185FA5; }
.result-conf { font-size: 13px; color: #5F6B76; margin-left: 8px; }
.error-card { background: #FDECEA; border: 1px solid #E5484D; border-radius: 12px; padding: 18px; margin-top: 12px; }
.error-title { margin: 0 0 6px 0; font-size: 15px; font-weight: 700; color: #B3261E; }
.error-text { margin: 0; font-size: 13px; color: #7A1E19; }
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
    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
    return model, cam, class_names, device

model, cam, class_names, device = load_model()
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

uploaded_file = st.file_uploader("Upload a knee image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    rgb_img = np.array(pil_img).astype(np.float32) / 255.0
    input_tensor = transform(pil_img).unsqueeze(0)

    with st.spinner("Analyzing image..."):
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            confidence = probs[pred_idx].item() * 100

    sorted_probs = torch.sort(probs, descending=True).values
    margin = (sorted_probs[0] - sorted_probs[1]).item() * 100
    is_valid = confidence >= CONFIDENCE_THRESHOLD and margin >= MARGIN_THRESHOLD

    col1, col2 = st.columns(2)
    with col1:
        st.image(pil_img, caption="Uploaded image", use_container_width=True)

    if not is_valid:
        with col2:
            st.markdown("""
            <div class="error-card">
                <p class="error-title">❌ Error: Not a recognizable knee image</p>
                <p class="error-text">The uploaded image does not match the knee X-ray/thermal patterns
                this model was trained on, so no reliable prediction can be made. Please upload a clear
                knee joint image (X-ray or thermal-style) and try again.</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        with torch.no_grad():
            grayscale_cam = cam(input_tensor=input_tensor)[0]
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        with col2:
            st.image(visualization, caption="Grad-CAM heatmap", use_container_width=True)

        label = SEVERITY_LABELS.get(class_names[pred_idx], class_names[pred_idx])
        st.markdown(f"""
        <div class="result-card">
            <p class="result-label">Prediction result</p>
            <span class="result-grade">{label}</span><span class="result-conf">{confidence:.1f}% confidence</span>
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        st.write("**All class probabilities**")
        for i in range(len(class_names)):
            cls_label = SEVERITY_LABELS.get(class_names[i], class_names[i])
            st.progress(float(probs[i]), text=f"{cls_label}: {probs[i].item()*100:.1f}%")
else:
    st.write("Upload an image to get started.")
