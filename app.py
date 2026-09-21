import gradio as gr
import spaces
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.cm as cm
from datetime import datetime
from fpdf import FPDF
import tempfile

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

# ---------------------------------------------------------------------------
# Grad-CAM  (unchanged)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Brand palette — premium clinical AI direction
# (shared between the in-app HTML and the PDF)
# ---------------------------------------------------------------------------

# PDF uses RGB tuples (FPDF requirement) — kept close to the on-screen palette,
# slightly adapted for print legibility.
CHARCOAL = (20, 23, 28)             # #14171C — top header (dark, not blue)
CHARCOAL_2 = (32, 36, 43)           # #20242B — header gradient end
BRAND_BLUE = (22, 119, 184)         # #1677B8 — reserved for the "clicked" button state only
BRAND_TEAL = (22, 166, 160)         # #16A6A0 — primary accent
DARK = (23, 43, 58)                 # #172B3A
GREY = (100, 116, 139)              # #64748B
LIGHT_TEAL_BG = (233, 247, 245)     # #E9F7F5
BORDER_GREY = (217, 228, 234)       # #D9E4EA

# Hex tokens used throughout the CSS
CHARCOAL_HEX = "#14171C"
CHARCOAL_2_HEX = "#20242B"
BRAND_BLUE_HEX = "#1677B8"          # reserved: only used for the button "clicked" state
BRAND_TEAL_HEX = "#16A6A0"
SOFT_TEAL_HEX = "#E9F7F5"
PAGE_BG_HEX = "#F3F7FA"
DARK_HEX = "#172B3A"
GREY_HEX = "#64748B"
BORDER_GREY_HEX = "#D9E4EA"

GRADE_TO_SEGMENTS = {  # how many of 4 bar segments to fill, by grade
    "0": 1, "1": 2, "2": 2, "3": 3, "4": 4,
}

# Severity accent per grade — kept muted/clinical, never alarmist bright red/green.
GRADE_ACCENT = {
    "0": BRAND_TEAL_HEX,
    "1": "#3FA9A4",
    "2": "#D6A400",
    "3": "#C97A2B",
    "4": "#B4552F",
}

def match_word(confidence):
    if confidence >= 85:
        return "Strong match"
    if confidence >= 70:
        return "Moderate match"
    return "Possible match"

# ---------------------------------------------------------------------------
# In-app result HTML (clinical AI dashboard result card + summary + bars)
# ---------------------------------------------------------------------------

def build_result_html(name, age, sex, symptoms, label, class_key, confidence, probs_list):
    filled = GRADE_TO_SEGMENTS.get(class_key, 2)
    accent = GRADE_ACCENT.get(class_key, BRAND_TEAL_HEX)
    segments_html = "".join(
        f'<div class="seg {"filled" if i < filled else ""}" style="{"background:" + accent + ";" if i < filled else ""}"></div>'
        for i in range(4)
    )

    # sort so the predicted class bar can be highlighted, but keep display order
    max_val = max(v for _, v in probs_list) if probs_list else 0
    bars_html = ""
    for lbl, val in probs_list:
        is_pred = abs(val - confidence) < 1e-6 and lbl == label
        bar_cls = "prob-bar-fill predicted" if is_pred else "prob-bar-fill"
        bars_html += f"""
        <div class="prob-row">
            <div class="prob-row-top">
                <span class="prob-label">{lbl}</span>
                <span class="prob-val">{val:.1f}%</span>
            </div>
            <div class="prob-bar-track">
                <div class="{bar_cls}" style="width:{max(val, 1.5):.1f}%;"></div>
            </div>
        </div>
        """

    return f"""
    <div class="result-wrap">
        <div class="result-card">
            <div class="result-eyebrow">AI SCREENING RESULT</div>
            <div class="result-title" style="color:{accent};">{label}</div>
            <div class="segbar">{segments_html}</div>
            <div class="match-word">{match_word(confidence)} &middot; {confidence:.1f}% confidence</div>
        </div>
        <div class="summary-panel">
            <div class="panel-heading">PATIENT SUMMARY</div>
            <div class="info-row"><span class="info-label">Name</span><span class="info-val">{name}</span></div>
            <div class="info-row"><span class="info-label">Age</span><span class="info-val">{age}</span></div>
            <div class="info-row"><span class="info-label">Gender</span><span class="info-val">{sex}</span></div>
            <div class="info-row"><span class="info-label">Symptoms</span><span class="info-val">{symptoms or "None provided"}</span></div>
        </div>
        <div class="prob-panel">
            <div class="panel-heading">ALL CLASS PROBABILITIES</div>
            {bars_html}
        </div>
    </div>
    """

def step_nav(active):
    steps = ["Patient Info", "Thermal Image", "AI Analysis"]
    items = ""
    for i, s in enumerate(steps, start=1):
        if i < active:
            state = "done"
        elif i == active:
            state = "active"
        else:
            state = "pending"
        marker = "&#10003;" if state == "done" else f"{i:02d}"
        items += f"""
        <div class="nav-item {state}">
            <div class="nav-dot">{marker}</div>
            <div class="nav-label">{s}</div>
        </div>
        """
        if i != len(steps):
            connector_state = "done" if i < active else "pending"
            items += f'<div class="nav-connector {connector_state}"></div>'
    return f'<div class="step-nav">{items}</div>'

def top_header_html():
    return """
    <div class="app-header">
        <div class="app-header-left">
            <div class="app-title">ArthroScan AI</div>
            <div class="app-subtitle">AI-Powered Knee Osteoarthritis Screening</div>
        </div>
        <div class="app-badge">
            <span class="badge-dot"></span>
            AI Screening System
        </div>
    </div>
    """

def info_note_html():
    return """
    <div class="info-note">
        <span class="info-note-icon">&#9432;</span>
        Patient data is used only to generate the screening report.
    </div>
    """

def upload_hint_html():
    return """
    <div class="upload-hint">
        <div class="upload-hint-title">Image requirements</div>
        <div class="upload-hint-body">Use a clear thermal image of the knee joint. Supported formats: JPG, PNG.</div>
    </div>
    """

def gradcam_header_html():
    return """
    <div class="section-heading-row">
        <div>
            <div class="section-title">AI Attention Map</div>
            <div class="section-subtitle">Visualization of image regions influencing the model prediction.</div>
        </div>
        <div class="xai-badge">Explainable AI &middot; Grad-CAM</div>
    </div>
    """

def report_header_html():
    return """
    <div class="report-card-heading">
        <div class="section-title">Screening Report</div>
        <div class="section-subtitle">Your AI screening report is ready to download.</div>
    </div>
    """

def disclaimer_html():
    return """
    <div class="disclaimer">
        <span class="disclaimer-icon">&#9432;</span>
        ArthroScan AI is a screening aid and does not replace professional medical diagnosis.
    </div>
    """

# ---------------------------------------------------------------------------
# PDF report (mirrors the app's clinical layout, includes the heatmap image)
# ---------------------------------------------------------------------------

def make_pdf(name, age, sex, symptoms, label, class_key, confidence, probs_text_pairs, heatmap_img):
    pdf = FPDF()
    pdf.add_page()

    # Header band
    pdf.set_fill_color(*CHARCOAL)
    pdf.rect(0, 0, 210, 26, style="F")
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(10, 7)
    pdf.cell(0, 10, "ArthroScan AI", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(200, 224, 235)
    pdf.set_x(10)
    pdf.cell(0, 6, "Knee Osteoarthritis Screening Report", ln=True)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(170, 200, 215)
    pdf.set_xy(150, 9)
    pdf.cell(50, 5, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="R")

    pdf.set_draw_color(*BORDER_GREY)
    pdf.set_line_width(0.4)
    pdf.line(10, 32, 200, 32)

    # Result card (left) + patient info panel (right), side by side
    top_y = 40
    card_w = 95
    panel_x = 10 + card_w + 6
    accent = GRADE_ACCENT.get(class_key, BRAND_TEAL_HEX).lstrip("#")
    accent_rgb = tuple(int(accent[i:i+2], 16) for i in (0, 2, 4))

    # -- left: predicted grade card --
    pdf.set_fill_color(*LIGHT_TEAL_BG)
    pdf.set_draw_color(*BORDER_GREY)
    pdf.rect(10, top_y, card_w, 46, style="FD")

    pdf.set_xy(15, top_y + 4)
    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(*GREY)
    pdf.cell(0, 4, "AI SCREENING RESULT", ln=True)

    pdf.set_xy(15, top_y + 9)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(*accent_rgb)
    pdf.multi_cell(card_w - 10, 6, label)

    # segmented confidence bar
    filled = GRADE_TO_SEGMENTS.get(class_key, 2)
    bar_y = top_y + 26
    seg_w = (card_w - 10 - 3 * 2) / 4
    seg_x = 15
    for i in range(4):
        if i < filled:
            pdf.set_fill_color(*accent_rgb)
        else:
            pdf.set_fill_color(214, 221, 227)
        pdf.rect(seg_x, bar_y, seg_w, 4, style="F")
        seg_x += seg_w + 2

    pdf.set_xy(15, bar_y + 7)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*GREY)
    pdf.cell(0, 5, f"{match_word(confidence)} - {confidence:.1f}% confidence", ln=True)

    # -- right: patient info panel --
    pdf.set_draw_color(*BORDER_GREY)
    pdf.rect(panel_x, top_y, 190 - panel_x, 46)

    pdf.set_xy(panel_x + 4, top_y + 3)
    pdf.set_font("Helvetica", "B", 7)
    pdf.set_text_color(*GREY)
    pdf.cell(0, 4, "PATIENT SUMMARY", ln=True)

    info_rows = [
        ("Name", str(name)),
        ("Age", str(age)),
        ("Gender", str(sex)),
    ]
    y = top_y + 9
    for label_, val in info_rows:
        pdf.set_xy(panel_x + 4, y)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*GREY)
        pdf.cell(20, 5, label_)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*DARK)
        pdf.cell(0, 5, val)
        y += 6

    pdf.set_xy(panel_x + 4, y + 1)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*GREY)
    pdf.cell(0, 5, "Symptoms", ln=True)
    pdf.set_x(panel_x + 4)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*DARK)
    pdf.multi_cell(190 - panel_x - 8, 5, symptoms or "None provided")

    # Probabilities table (horizontal bars)
    probs_y = top_y + 46 + 10
    pdf.set_xy(10, probs_y)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*DARK)
    pdf.cell(0, 7, "All class probabilities", ln=True)

    row_y = pdf.get_y() + 2
    bar_max_w = 100
    for cls_label, val in probs_text_pairs:
        pdf.set_xy(10, row_y)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*DARK)
        pdf.cell(60, 5, cls_label)

        # bar track
        pdf.set_fill_color(*BORDER_GREY)
        pdf.rect(72, row_y + 1, bar_max_w, 3, style="F")
        fill_w = bar_max_w * min(val, 100) / 100
        pdf.set_fill_color(*BRAND_TEAL)
        pdf.rect(72, row_y + 1, fill_w, 3, style="F")

        pdf.set_xy(72 + bar_max_w + 3, row_y)
        pdf.set_text_color(*GREY)
        pdf.cell(0, 5, f"{val:.1f}%")
        row_y += 7

    # Grad-CAM heatmap image
    img_y = row_y + 5
    if heatmap_img is not None:
        tmp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        Image.fromarray(heatmap_img).save(tmp_img.name)
        pdf.set_xy(10, img_y)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*BRAND_TEAL)
        pdf.cell(0, 7, "AI Attention Map (Grad-CAM)", ln=True)
        pdf.set_draw_color(*BORDER_GREY)
        pdf.rect(10, img_y + 8, 90, 90)
        pdf.image(tmp_img.name, x=11, y=img_y + 9, w=88)

    pdf.set_y(-20)
    pdf.set_draw_color(*BORDER_GREY)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(*GREY)
    pdf.cell(0, 8, "ArthroScan AI is a screening aid only and is not a substitute for professional medical diagnosis.", align="C")

    tmp_pdf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    pdf.output(tmp_pdf.name)
    return tmp_pdf.name

# ---------------------------------------------------------------------------
# Inference (unchanged)
# ---------------------------------------------------------------------------

@spaces.GPU
def analyze(name, age, sex, symptoms, image):
    if image is None:
        return "<p>Please upload an image.</p>", None, None

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
        return "<p>&#10060; This file is not valid. Please upload the knee thermal image only.</p>", None, None

    class_key = class_names[pred_idx]
    label = SEVERITY_LABELS.get(class_key, class_key)

    try:
        cam = compute_gradcam(input_tensor, model.layer4[-1], pred_idx)
        heatmap = overlay_heatmap(rgb_img, cam)
    except Exception:
        heatmap = None

    probs_pairs = []
    for i in range(len(class_names)):
        cls_label = SEVERITY_LABELS.get(class_names[i], class_names[i])
        probs_pairs.append((cls_label, probs[i].item() * 100))

    pdf_path = make_pdf(name, age, sex, symptoms, label, class_key, confidence, probs_pairs, heatmap)

    result_html = build_result_html(name, age, sex, symptoms, label, class_key, confidence, probs_pairs)

    return result_html, heatmap, pdf_path

# ---------------------------------------------------------------------------
# Step navigation callbacks (unchanged logic)
# ---------------------------------------------------------------------------

def go_to_step2(name, age, sex):
    if not name or not name.strip():
        gr.Warning("Please enter your name to continue.")
        return gr.update(), gr.update()
    if age is None:
        gr.Warning("Please enter your age to continue.")
        return gr.update(), gr.update()
    if not sex:
        gr.Warning("Please select your sex to continue.")
        return gr.update(), gr.update()
    return gr.update(visible=False), gr.update(visible=True)

def go_back_to_step1():
    return gr.update(visible=True), gr.update(visible=False)

def run_and_advance(name, age, sex, symptoms, image):
    if image is None:
        gr.Warning("Please upload a knee image to continue.")
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    result_html_val, heatmap_val, pdf_val = analyze(name, age, sex, symptoms, image)
    return (
        gr.update(visible=False),
        gr.update(visible=True),
        result_html_val,
        heatmap_val,
        gr.update(value=pdf_val, visible=True),
    )

def restart():
    return (
        gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
        "", None, None, "", None, "", None, gr.update(value=None, visible=False)
    )

# ---------------------------------------------------------------------------
# CSS — premium clinical AI dashboard direction
# ---------------------------------------------------------------------------

CSS = f"""
:root {{
    --charcoal: {CHARCOAL_HEX};
    --charcoal-2: {CHARCOAL_2_HEX};
    --brand-blue: {BRAND_BLUE_HEX};
    --brand-teal: {BRAND_TEAL_HEX};
    --soft-teal: {SOFT_TEAL_HEX};
    --teal-dark: #0E7C74;
    --page-bg: {PAGE_BG_HEX};
    --text-dark: {DARK_HEX};
    --text-grey: {GREY_HEX};
    --border: {BORDER_GREY_HEX};
    color-scheme: light !important;
}}

/* ---------- force the light clinical theme regardless of the visitor's
   OS/browser dark-mode setting. ----------
   Gradio's dark mode isn't just a background swap: every built-in component
   (labels, inputs, buttons, markdown) reads its color from a shared set of
   Gradio CSS variables (--body-text-color, --background-fill-*, etc.), and
   those variables get reassigned to dark values when `.dark` is present on
   the container. Overriding individual components is whack-a-mole and is
   what caused the invisible/illegible text. Instead we pin the variables
   themselves back to our light palette, on both the plain and `.dark`
   container, so every component (ours and Gradio's own) renders correctly
   no matter which mode the visitor's browser requests. */
html, body {{ color-scheme: light !important; }}

.gradio-container,
.gradio-container.dark,
.dark .gradio-container {{
    --body-text-color: {DARK_HEX} !important;
    --body-text-color-subdued: {GREY_HEX} !important;
    --background-fill-primary: #FFFFFF !important;
    --background-fill-secondary: {PAGE_BG_HEX} !important;
    --block-background-fill: #FFFFFF !important;
    --block-label-text-color: {GREY_HEX} !important;
    --block-title-text-color: {DARK_HEX} !important;
    --input-background-fill: {PAGE_BG_HEX} !important;
    --border-color-primary: {BORDER_GREY_HEX} !important;
    --border-color-accent: {BORDER_GREY_HEX} !important;
    --button-secondary-background-fill: #FFFFFF !important;
    --button-secondary-text-color: {DARK_HEX} !important;
    --neutral-50: {PAGE_BG_HEX} !important;
    --neutral-100: #FFFFFF !important;
    --neutral-950: {DARK_HEX} !important;

    background: var(--page-bg) !important;
    background-image:
        radial-gradient(circle at 15% 0%, rgba(20, 23, 28, 0.045), transparent 45%),
        radial-gradient(circle at 100% 20%, rgba(22, 166, 160, 0.06), transparent 40%) !important;
    color: var(--text-dark) !important;
}}

/* Safety net: every piece of text inside our shell defaults to the dark
   readable color. Elements that intentionally sit on a dark background
   (header title, badges, buttons) set their own color afterward with more
   specific selectors below, so they still win. */
.body-shell, .body-shell * {{
    color: var(--text-dark);
}}
.body-shell label span, .body-shell label {{
    color: var(--text-grey) !important;
    background: transparent !important;
    -webkit-text-fill-color: var(--text-grey) !important;
}}
.body-shell input[type="text"],
.body-shell input[type="number"],
.body-shell textarea {{
    -webkit-text-fill-color: var(--text-dark) !important;
}}

* {{
    font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}}

.app-shell {{
    max-width: 900px !important;
    margin: 28px auto !important;
    padding: 0 !important;
}}

/* ---------- header ---------- */
/* Light, same treatment as the rest of the interface — no separate dark
   bar, so the title is just as visible/consistent as every other section. */
.app-header {{
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-bottom: none;
    border-radius: 12px 12px 0 0;
    padding: 22px 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 10px;
}}
.app-title {{
    color: var(--text-dark);
    font-size: 21px;
    font-weight: 800;
    letter-spacing: -0.01em;
}}
.app-subtitle {{
    color: var(--text-grey);
    font-size: 13px;
    font-weight: 500;
    margin-top: 2px;
}}
.app-badge {{
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: var(--soft-teal);
    border: 1px solid rgba(22, 166, 160, 0.35);
    color: var(--teal-dark);
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.03em;
    padding: 6px 12px;
    border-radius: 999px;
}}
.badge-dot {{
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--brand-teal);
    box-shadow: 0 0 0 3px rgba(22, 166, 160, 0.25);
    display: inline-block;
}}

/* ---------- body shell ---------- */
.body-shell {{
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-top: none;
    border-radius: 0 0 12px 12px;
    box-shadow: 0 8px 24px rgba(11, 31, 51, 0.06);
    overflow: hidden;
}}

/* ---------- horizontal step nav ---------- */
.step-nav {{
    display: flex;
    align-items: center;
    padding: 20px 28px;
    background: #FBFDFE;
    border-bottom: 1px solid var(--border);
}}
.nav-item {{
    display: flex;
    align-items: center;
    gap: 9px;
    flex-shrink: 0;
}}
.nav-dot {{
    width: 26px;
    height: 26px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    font-weight: 700;
    background: #E7ECF1;
    color: #97A3B0;
    border: 1px solid var(--border);
}}
.nav-item.active .nav-dot {{
    background: var(--charcoal);
    color: #FFFFFF;
    border-color: var(--charcoal);
    box-shadow: 0 0 0 4px rgba(20, 23, 28, 0.12);
}}
.nav-item.done .nav-dot {{
    background: var(--brand-teal);
    color: #FFFFFF;
    border-color: var(--brand-teal);
}}
.nav-label {{
    font-size: 12.5px;
    font-weight: 600;
    color: #97A3B0;
    white-space: nowrap;
}}
.nav-item.active .nav-label {{
    color: var(--text-dark);
}}
.nav-item.done .nav-label {{
    color: var(--text-grey);
}}
.nav-connector {{
    flex: 1;
    height: 2px;
    background: var(--border);
    margin: 0 10px;
    min-width: 20px;
}}
.nav-connector.done {{
    background: var(--brand-teal);
}}

.step-body {{
    padding: 26px 28px 30px 28px !important;
}}

.step-heading, .step-heading *,
.step-heading.prose, .step-heading .prose,
.dark .step-heading, .dark .step-heading * {{
    background: transparent !important;
    color: var(--text-dark) !important;
    font-weight: 700 !important;
    font-size: 17px !important;
    margin-bottom: 2px !important;
    box-shadow: none !important;
}}
.body-shell .prose, .body-shell .prose * {{
    background: transparent !important;
}}
.step-subtext {{
    color: var(--text-grey);
    font-size: 13px;
    margin-bottom: 16px;
}}

.gr-group {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}}

.body-shell label {{
    color: var(--text-grey) !important;
    font-weight: 600 !important;
    font-size: 12.5px !important;
    letter-spacing: 0.01em;
}}

.body-shell input[type="text"],
.body-shell input[type="number"],
.body-shell textarea {{
    background: var(--page-bg) !important;
    color: var(--text-dark) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}}
.body-shell input[type="text"]:focus,
.body-shell input[type="number"]:focus,
.body-shell textarea:focus {{
    border-color: var(--brand-teal) !important;
    box-shadow: 0 0 0 3px rgba(22, 166, 160, 0.15) !important;
}}

/* ---------- Gender field ---------- */
/* The "Gender" field label itself stays plain/neutral like every other
   field label; only the individual option pills react to selection.
   Driven purely by :has(input:checked) — never by Gradio's own dynamic
   "selected" class, which was unreliable and previously bled the blue
   fill onto the "Gender" label itself. */
.body-shell [role="radiogroup"] > label {{
    background: var(--page-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-dark) !important;
    transition: background 0.15s ease, border-color 0.15s ease;
}}
.body-shell [role="radiogroup"] > label:has(input:checked) {{
    background: var(--brand-blue) !important;
    border-color: var(--brand-blue) !important;
}}
.body-shell [role="radiogroup"] > label:has(input:checked) span {{
    color: #FFFFFF !important;
}}

/* The round selector: hollow white circle with a black outline at rest,
   solid blue fill only once checked. */
.body-shell [role="radiogroup"] input[type="radio"] {{
    appearance: none !important;
    -webkit-appearance: none !important;
    -moz-appearance: none !important;
    outline: none !important;
    box-sizing: border-box !important;
    width: 15px !important;
    height: 15px !important;
    border-radius: 50% !important;
    border: 2px solid #000000 !important;
    background: #FFFFFF !important;
    box-shadow: none !important;
    accent-color: transparent !important;
    margin: 0 6px 0 0 !important;
    flex-shrink: 0;
}}
.body-shell [role="radiogroup"] input[type="radio"]:checked {{
    background: var(--brand-blue) !important;
    border-color: var(--brand-blue) !important;
}}

/* stack the intake fields one after another, evenly spaced */
.intake-col {{
    gap: 14px !important;
}}

/* info note (patient info page) */
.info-note {{
    display: flex;
    align-items: flex-start;
    gap: 8px;
    background: var(--soft-teal);
    border: 1px solid rgba(22, 166, 160, 0.25);
    color: var(--teal-dark);
    font-size: 12.5px;
    padding: 10px 14px;
    border-radius: 8px;
    margin: 14px 0 4px 0;
}}
.info-note-icon {{
    color: var(--brand-teal);
    font-size: 14px;
    line-height: 1.2;
}}

/* upload dropzone styling (targets Gradio's image upload wrapper) */
.upload-zone .image-frame,
.upload-zone [data-testid="image"] {{
    border: 1.5px dashed rgba(22, 166, 160, 0.5) !important;
    border-radius: 10px !important;
    background: var(--soft-teal) !important;
}}
.upload-hint {{
    background: var(--page-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 14px;
    margin-top: 14px;
}}
.upload-hint-title {{
    font-size: 12.5px;
    font-weight: 700;
    color: var(--text-dark);
    margin-bottom: 3px;
}}
.upload-hint-body {{
    font-size: 12px;
    color: var(--text-grey);
}}

button {{
    border-radius: 999px !important;
    font-weight: 600 !important;
    font-size: 13.5px !important;
    transition: background 0.15s ease, color 0.15s ease, border-color 0.15s ease !important;
}}
/* Primary buttons: white at rest, blue only while pressed/active — never
   filled blue by default. */
#continue-btn, #continue-btn2, #analyze-btn {{
    background: #FFFFFF !important;
    color: var(--text-dark) !important;
    border: 1.5px solid var(--border) !important;
    box-shadow: 0 2px 6px rgba(20, 23, 28, 0.06) !important;
}}
#continue-btn *, #continue-btn2 *, #analyze-btn * {{
    color: var(--text-dark) !important;
}}
#continue-btn:hover, #continue-btn2:hover, #analyze-btn:hover {{
    border-color: var(--brand-blue) !important;
}}
#continue-btn:active, #continue-btn2:active, #analyze-btn:active,
#continue-btn:focus, #continue-btn2:focus, #analyze-btn:focus {{
    background: var(--brand-blue) !important;
    border-color: var(--brand-blue) !important;
}}
#continue-btn:active *, #continue-btn2:active *, #analyze-btn:active *,
#continue-btn:focus *, #continue-btn2:focus *, #analyze-btn:focus * {{
    color: #FFFFFF !important;
}}
#back-btn {{
    background: #FFFFFF !important;
    border: 1.5px solid var(--border) !important;
}}
#back-btn * {{
    color: var(--text-dark) !important;
}}
#back-btn:active, #back-btn:focus {{
    background: var(--brand-blue) !important;
    border-color: var(--brand-blue) !important;
}}
#back-btn:active *, #back-btn:focus * {{
    color: #FFFFFF !important;
}}

/* ---------- result page ---------- */
.section-heading-row {{
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 8px;
    margin: 18px 0 10px 0;
}}
.section-title {{
    font-size: 15px;
    font-weight: 700;
    color: var(--text-dark);
}}
.section-subtitle {{
    font-size: 12.5px;
    color: var(--text-grey);
    margin-top: 2px;
}}
.xai-badge {{
    background: var(--soft-teal);
    color: var(--teal-dark);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.02em;
    padding: 5px 10px;
    border-radius: 999px;
    border: 1px solid rgba(22, 166, 160, 0.3);
    white-space: nowrap;
}}

.result-wrap {{
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
}}
.result-card {{
    flex: 1 1 260px;
    background: linear-gradient(180deg, var(--soft-teal) 0%, #FFFFFF 100%);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px 20px;
}}
.result-eyebrow {{
    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: var(--brand-teal);
    margin-bottom: 8px;
}}
.result-title {{
    font-weight: 800;
    font-size: 19px;
    margin-bottom: 12px;
}}
.segbar {{
    display: flex;
    gap: 4px;
    margin-bottom: 9px;
}}
.seg {{
    flex: 1;
    height: 7px;
    border-radius: 4px;
    background: #DDE6EC;
}}
.match-word {{
    font-size: 12.5px;
    color: var(--text-grey);
    font-weight: 500;
}}
.summary-panel, .prob-panel {{
    flex: 1 1 220px;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 20px;
    background: #FFFFFF;
}}
.prob-panel {{
    flex: 1 1 100%;
}}
.panel-heading {{
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.06em;
    color: var(--text-grey);
    margin-bottom: 10px;
}}
.info-row {{
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid #EEF2F5;
    font-size: 13px;
}}
.info-row:last-child {{
    border-bottom: none;
}}
.info-label {{
    color: var(--text-grey);
    font-weight: 600;
}}
.info-val {{
    color: var(--text-dark);
    text-align: right;
    max-width: 65%;
}}
.prob-row {{
    margin-bottom: 11px;
}}
.prob-row:last-child {{
    margin-bottom: 0;
}}
.prob-row-top {{
    display: flex;
    justify-content: space-between;
    font-size: 12.5px;
    margin-bottom: 4px;
}}
.prob-label {{
    color: var(--text-dark);
    font-weight: 500;
}}
.prob-val {{
    color: var(--text-grey);
    font-weight: 600;
}}
.prob-bar-track {{
    background: #EEF2F5;
    border-radius: 5px;
    height: 7px;
    overflow: hidden;
}}
.prob-bar-fill {{
    height: 100%;
    background: #9FB3C4;
    border-radius: 5px;
}}
.prob-bar-fill.predicted {{
    background: var(--brand-teal);
}}

/* hide the built-in download icon on image components (this Gradio
   version's Image component has no show_download_button constructor arg) */
.upload-zone button[aria-label="Download"],
.gradcam-frame button[aria-label="Download"] {{
    display: none !important;
}}

/* Grad-CAM image framing */
.gradcam-frame img {{
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 14px rgba(11, 31, 51, 0.08) !important;
}}

/* PDF report card */
.report-card-heading {{
    background: var(--soft-teal);
    border: 1px solid rgba(22, 166, 160, 0.25);
    border-radius: 10px 10px 0 0;
    padding: 14px 18px 8px 18px;
}}

/* disclaimer */
.disclaimer {{
    display: flex;
    align-items: flex-start;
    gap: 7px;
    color: var(--text-grey);
    font-size: 11.5px;
    margin-top: 18px;
    padding-top: 14px;
    border-top: 1px solid var(--border);
}}
.disclaimer-icon {{
    font-size: 13px;
}}

/* ---------- responsive ---------- */
@media (max-width: 640px) {{
    .app-shell {{
        margin: 0 !important;
        max-width: 100% !important;
    }}
    .app-header {{
        border-radius: 0;
        padding: 16px 18px;
    }}
    .body-shell {{
        border-radius: 0;
        border-left: none;
        border-right: none;
    }}
    .step-nav {{
        padding: 14px 16px;
        overflow-x: auto;
    }}
    .nav-label {{
        display: none;
    }}
    .step-body {{
        padding: 18px 16px 24px 16px !important;
    }}
    .result-wrap {{
        gap: 10px;
    }}
}}
"""

with gr.Blocks(title="ArthroScan AI", css=CSS, theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate")) as demo:
    with gr.Column(elem_classes="app-shell"):
        gr.HTML(top_header_html())

        with gr.Column(elem_classes="body-shell"):
            with gr.Group(visible=True) as step1:
                gr.HTML(step_nav(1))
                with gr.Column(elem_classes="step-body"):
                    gr.Markdown("Patient Information", elem_classes="step-heading")
                    gr.HTML('<div class="step-subtext">Enter basic patient details before starting the AI screening.</div>')
                    with gr.Column(elem_classes="intake-col"):
                        name = gr.Textbox(label="Name", placeholder="Enter your name")
                        age = gr.Number(label="Age", minimum=1, maximum=120)
                        sex = gr.Radio(["Male", "Female", "Other"], label="Gender")
                        symptoms = gr.Textbox(label="Symptoms", placeholder="e.g. knee pain, stiffness, swelling")
                    gr.HTML(info_note_html())
                    next1 = gr.Button("Continue →", elem_id="continue-btn")

            with gr.Group(visible=False) as step2:
                gr.HTML(step_nav(2))
                with gr.Column(elem_classes="step-body"):
                    gr.Markdown("Upload Knee Thermal Image", elem_classes="step-heading")
                    gr.HTML('<div class="step-subtext">Upload a thermal image of the knee for AI-based osteoarthritis screening.</div>')
                    with gr.Column(elem_classes="upload-zone"):
                        image = gr.Image(label="Drag & drop or browse your image", type="numpy")
                    gr.HTML(upload_hint_html())
                    with gr.Row():
                        back2 = gr.Button("← Back", elem_id="back-btn")
                        next2 = gr.Button("Analyze Image →", elem_id="continue-btn2")

            with gr.Group(visible=False) as step3:
                gr.HTML(step_nav(3))
                with gr.Column(elem_classes="step-body"):
                    result_html = gr.HTML()
                    gr.HTML(gradcam_header_html())
                    with gr.Column(elem_classes="gradcam-frame"):
                        heatmap_out = gr.Image(label=None, show_label=False)
                    gr.HTML(report_header_html())
                    pdf_out = gr.File(label="↓ Download PDF Report", visible=False)
                    gr.HTML(disclaimer_html())
                    back3 = gr.Button("← Start Over", elem_id="back-btn")

        next1.click(go_to_step2, inputs=[name, age, sex], outputs=[step1, step2])
        back2.click(go_back_to_step1, outputs=[step1, step2])
        next2.click(
            run_and_advance,
            inputs=[name, age, sex, symptoms, image],
            outputs=[step2, step3, result_html, heatmap_out, pdf_out],
        )
        back3.click(
            restart,
            outputs=[step1, step2, step3, name, age, sex, symptoms, image, result_html, heatmap_out, pdf_out],
        )

demo.launch()
