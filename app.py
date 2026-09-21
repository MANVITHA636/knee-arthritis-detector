import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.cm as cm
from datetime import datetime

MODEL_PATH = "model.pth"
IMAGE_SIZE = 224
SEVERITY_LABELS = {
    "0": "Grade 0 - Normal", "1": "Grade 1 - Doubtful", "2": "Grade 2 - Mild",
    "3": "Grade 3 - Moderate", "4": "Grade 4 - Severe",
}
CONFIDENCE_THRESHOLD = 65.0
MARGIN_THRESHOLD = 20.0
SATURATION_THRESHOLD = 55

device = torch.device("cpu")
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
class_names = checkpoint["class_names"]
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def compute_gradcam(input_tensor, target_layer, class_idx):
    activations, gradients = [], []
    def fh(module, inp, out): activations.append(out)
    def bh(module, gi, go): gradients.append(go[0])
    h1 = target_layer.register_forward_hook(fh)
    h2 = target_layer.register_full_backward_hook(bh)
    model.zero_grad()
    output = model(input_tensor)
    output[0, class_idx].backward()
    h1.remove(); h2.remove()
    act = activations[0].detach()[0]
    grad = gradients[0].detach()[0]
    weights = grad.mean(dim=(1, 2))
    cam = torch.zeros(act.shape[1:], dtype=torch.float32)
    for i, w in enumerate(weights):
        cam += w * act[i]
    cam = F.relu(cam)
    cam = cam / (cam.max() + 1e-8)
    cam_img = Image.fromarray(np.uint8(cam.numpy() * 255)).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    return np.array(cam_img).astype(np.float32) / 255.0

def overlay_heatmap(rgb_img, cam, alpha=0.45):
    heatmap = cm.jet(cam)[:, :, :3]
    overlay = np.clip(heatmap * alpha + rgb_img * (1 - alpha), 0, 1)
    return np.uint8(overlay * 255)

def analyze(name, age, sex, symptoms, image):
    if image is None:
        return "Please upload an image.", None, None

    pil_img = Image.fromarray(image).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    rgb_img = np.array(pil_img).astype(np.float32) / 255.0
    input_tensor = transform(pil_img).unsqueeze(0)

    outputs = model(input_tensor)
    probs = torch.softmax(outputs, dim=1)[0]
    pred_idx = torch.argmax(probs).item()
    confidence = probs[pred_idx].item() * 100
    sorted_probs = torch.sort(probs, descending=True).values
    margin = (sorted_probs[0] - sorted_probs[1]).item() * 100

    hsv_img = pil_img.convert("HSV")
    mean_saturation = np.array(hsv_img)[:, :, 1].mean()

    is_valid = (confidence >= CONFIDENCE_THRESHOLD and margin >= MARGIN_THRESHOLD and mean_saturation >= SATURATION_THRESHOLD)

    if not is_valid:
        return "❌ This file is not valid. Please upload the knee thermal image only.", None, None

    label = SEVERITY_LABELS.get(class_names[pred_idx], class_names[pred_idx])

    try:
        cam = compute_gradcam(input_tensor, model.layer4[-1], pred_idx)
        heatmap = overlay_heatmap(rgb_img, cam)
    except Exception:
        heatmap = None

    report = (
        f"🦵 ArthroScan AI - Screening Report\n"
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"{'-'*40}\n"
        f"Name: {name}\nAge: {age}\nSex: {sex}\nSymptoms: {symptoms or 'None provided'}\n"
        f"{'-'*40}\n"
        f"Predicted Grade: {label}\nConfidence: {confidence:.1f}%\n\n"
        f"All class probabilities:\n"
    )
    for i in range(len(class_names)):
        cls_label = SEVERITY_LABELS.get(class_names[i], class_names[i])
        report += f"  {cls_label}: {probs[i].item()*100:.1f}%\n"
    report += f"\n{'-'*40}\nResearch proof-of-concept only. Not a substitute for professional medical diagnosis.\n"

    return f"**{label}** — Confidence: {confidence:.1f}%", heatmap, report

with gr.Blocks(title="ArthroScan AI") as demo:
    gr.Markdown("## 🦵 ArthroScan AI — Knee Osteoarthritis Severity Screening")
    gr.Markdown("*Research proof-of-concept only. Not a substitute for professional medical diagnosis.*")

    with gr.Row():
        with gr.Column():
            name = gr.Textbox(label="Name")
            age = gr.Number(label="Age", minimum=1, maximum=120)
            sex = gr.Radio(["M", "F", "O"], label="Sex")
            symptoms = gr.Textbox(label="Symptoms", placeholder="e.g. knee pain, stiffness, swelling")
            image = gr.Image(label="Upload knee thermal image", type="numpy")
            submit = gr.Button("Analyze", variant="primary")
        with gr.Column():
            result = gr.Markdown()
            heatmap_out = gr.Image(label="Grad-CAM heatmap")
            report_out = gr.Textbox(label="Full Report", lines=10)

    submit.click(analyze, inputs=[name, age, sex, symptoms, image], outputs=[result, heatmap_out, report_out])

demo.launch()