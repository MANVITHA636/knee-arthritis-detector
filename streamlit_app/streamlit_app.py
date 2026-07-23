import streamlit as st
import torch
torch.set_num_threads(1)
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

st.set_page_config(page_title="Knee Arthritis Detector", page_icon="🦵", layout="centered")

MODEL_PATH = "model.pth"
IMAGE_SIZE = 224

SEVERITY_LABELS = {
    "0": "Grade 0 - Normal",
    "1": "Grade 1 - Doubtful",
    "2": "Grade 2 - Mild",
    "3": "Grade 3 - Moderate",
    "4": "Grade 4 - Severe",
}


@st.cache_resource
def load_model():
    device = torch.device("cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    class_names = checkpoint["class_names"]

    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    target_layer = model.layer4[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])

    return model, cam, class_names, device


model, cam, class_names, device = load_model()

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])

st.title("🦵 AI-Based Knee Arthritis Severity Detection")
st.caption(
    "Upload a thermal-style knee image to get a predicted osteoarthritis "
    "severity grade (0=Normal to 4=Severe) along with a heatmap showing "
    "which regions influenced the prediction."
)
st.info(
    "This is a research/academic proof-of-concept and is not a substitute "
    "for professional medical diagnosis.",
    icon="ℹ️",
)

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

        predicted_class = class_names[pred_idx]
        label = SEVERITY_LABELS.get(predicted_class, f"Grade {predicted_class}")

        grayscale_cam = cam(input_tensor=input_tensor)[0]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    col1, col2 = st.columns(2)
    with col1:
        st.image(pil_img, caption="Uploaded Image", use_container_width=True)
    with col2:
        st.image(visualization, caption="Grad-CAM Heatmap", use_container_width=True)

    st.success(f"**Prediction: {label}**")
    st.metric("Confidence", f"{confidence:.1f}%")

    st.subheader("All class probabilities")
    for i in range(len(class_names)):
        cls_label = SEVERITY_LABELS.get(class_names[i], class_names[i])
        st.progress(float(probs[i]), text=f"{cls_label}: {probs[i].item()*100:.1f}%")
else:
    st.write("👆 Upload an image to get started.")
