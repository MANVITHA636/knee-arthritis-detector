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

# ---------- CUSTOM CSS: blue/white medical theme ----------
st.markdown("""
<style>
.header-bar {
    display: flex; align-items: center; gap: 12px;
    padding: 14px 18px; background: #F4F8FC; border-radius: 12px;
    border: 1px solid #D6E6F5; margin-bottom: 18px;
}
.header-title { margin: 0; font-weight: 600; font-size: 17px; color: #0C447C; }
.header-sub { margin: 0; font-size: 13px; color: #5F6B76; }
.step-bar { display: flex; align-items: center; gap: 8px; margin-bottom: 18px; }
.step-circle-active {
    width: 24px; height: 24px; border-radius: 50%; background: #185FA5;
    color: white; display: flex; align-items: center; justify-content: center;
    font-size: 12px; font-weight: 600;
}
.step-circle-inactive {
    width: 24px; height: 24px; border-radius: 50%; background: white;
    border: 1px solid #B5D4F4; color: #5F6B76; display: flex;
    align-items: center; justify-content: center; font-size: 12px;
}
.step-label-active { font-size: 13px; font-weight: 600; color: #0C447C; }
.step-label-inactive { font-size: 13px; color: #5F6B76; }
.step-line { flex: 1; height: 1px; background: #D6E6F5; }
.disclaimer-banner {
    background: #E6F1FB; border-radius: 8px; padding: 10px 14px; margin-bottom: 18px;
}
.disclaimer-text { margin: 0; font-size: 12.5px; color: #0C447C; }
.result-card {
    background: white; border: 1px solid #D6E6F5; border-radius: 12px;
    padding: 18px; margin-top: 12px;
}
.result-label { margin: 0 0 8px 0; font-size: 13px; color: #5F6B76; }
.result-grade { font-size: 21px; font-weight: 600; color: #185FA5; }
.result-conf { font-size: 13px; color: #5F6B76; margin-left: 8px; }
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("""
<div class="header-bar">
    <span style="font-size: 22px;">🦵</span>
    <div>
        <p class="header-title">ArthroScan AI</p>
        <p class="header-sub">Knee osteoarthritis severity screening</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------- STEP INDICATOR (dynamic based on whether an image is uploaded) ----------
def render_steps(active_step):
    steps = ["Upload", "Analyze", "Result"]
    html = '<div class="step-bar">'
    for i, s in enumerate(steps):
        step_num = i + 1
        if step_num <= active_step:
            html += f'<div style="display:flex;align-items:center;gap:6px;"><div class="step-circle-active">{step_num}</div><span class="step-label-active">{s}</span></div>'
        else:
            html += f'<div style="display:flex;align-items:center;gap:6px;"><div class="step-circle-inactive">{step_num}</div><span class="step-label-inactive">{s}</span></div>'
        if i < len(steps) - 1:
            html += '<div class="step-line"></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

# ---------- DISCLAIMER ----------
st.markdown("""
<div class="disclaimer-banner">
    <p class="disclaimer-text">Research proof-of-concept only. Not a substitute for professional medical diagnosis.</p>
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

if uploaded_file is None:
    render_steps(1)
    st.write("")
else:
    render_steps(3)
    pil_img = Image.open(uploaded_file).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    rgb_img = np.array(pil_img).astype(np.float32) / 255.0
    input_tensor = transform(pil_img).unsqueeze(0)

    with st.spinner("Analyzing image..."):
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            confidence = probs[pred_idx].item() * 100
        label = SEVERITY_LABELS.get(class_names[pred_idx], class_names[pred_idx])
        grayscale_cam = cam(input_tensor=input_tensor)[0]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    col1, col2 = st.columns(2)
    with col1:
        st.image(pil_img, caption="Uploaded image", use_container_width=True)
    with col2:
        st.image(visualization, caption="Grad-CAM heatmap", use_container_width=True)

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
