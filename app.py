import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import gradio as gr
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

MODEL_PATH = "model.pth"
IMAGE_SIZE = 224

device = torch.device("cpu")

checkpoint = torch.load(MODEL_PATH, map_location=device)
class_names = checkpoint["class_names"]

SEVERITY_LABELS = {
    "0": "Grade 0 - Normal",
    "1": "Grade 1 - Doubtful",
    "2": "Grade 2 - Mild",
    "3": "Grade 3 - Moderate",
    "4": "Grade 4 - Severe",
}

model = models.resnet18(weights=None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_names))
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

target_layer = model.layer4[-1]
cam = GradCAM(model=model, target_layers=[target_layer])

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])


def predict(image):
    if image is None:
        return "Please upload an image.", None

    pil_img = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    rgb_img = np.array(pil_img).astype(np.float32) / 255.0
    input_tensor = transform(pil_img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item() * 100

    predicted_class = class_names[pred_idx]
    label = SEVERITY_LABELS.get(predicted_class, f"Grade {predicted_class}")

    grayscale_cam = cam(input_tensor=input_tensor)[0]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    all_probs_text = "\n".join([
        f"{SEVERITY_LABELS.get(class_names[i], class_names[i])}: {probs[i].item()*100:.1f}%"
        for i in range(len(class_names))
    ])

    result_text = (
        f"**Prediction: {label}**\n"
        f"**Confidence: {confidence:.1f}%**\n\n"
        f"All class probabilities:\n{all_probs_text}"
    )

    return result_text, visualization


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Thermal Knee Image"),
    outputs=[
        gr.Markdown(label="Prediction Result"),
        gr.Image(label="Grad-CAM Heatmap (what the model focused on)"),
    ],
    title="AI-Based Knee Arthritis Severity Detection",
    description=(
        "Upload a thermal-style knee image to get a predicted osteoarthritis "
        "severity grade (0=Normal to 4=Severe) along with a heatmap showing "
        "which regions influenced the prediction.\n\n"
        "Note: This is a research/academic proof-of-concept and is not a "
        "substitute for professional medical diagnosis."
    ),
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)