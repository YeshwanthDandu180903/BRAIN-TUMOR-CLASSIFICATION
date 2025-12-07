# app.py (updated)
import os
import io
import json
import uuid
import time
import numpy as np
import cv2
from PIL import Image
from datetime import datetime
from flask import Flask, request, render_template, url_for, send_file, flash, redirect, session
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import Model
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from werkzeug.utils import secure_filename

# ------------------ CONFIG ------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "final_effnetb2.h5")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "models", "label_map.json")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
EXAMPLE_FOLDER = os.path.join(BASE_DIR, "static", "disease_examples")

NORMAL_REFERENCE_IMAGE = os.path.join(EXAMPLE_FOLDER, "normal_brain_mri.jpg")

ALLOWED_EXT = {"png", "jpg", "jpeg"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EXAMPLE_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = "change-me-to-a-random-secret-key"  # change for production
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"

# ------------------ LOAD MODEL & LABELS ------------------
model = None
label_map = {}
CLASS_NAMES = {}

try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    model = load_model(MODEL_PATH)
except Exception as e:
    # keep server running—template should show friendly error
    print("Error loading model:", e)
    model = None

try:
    if os.path.exists(LABEL_MAP_PATH):
        with open(LABEL_MAP_PATH, "r") as f:
            label_map = json.load(f)
        # Build mapping index -> readable label
        CLASS_NAMES = {v: k.replace("_", " ") for k, v in label_map.items()}
    else:
        print("Warning: label_map.json not found; using fallback labels.")
        CLASS_NAMES = {0: "Glioma", 1: "Meningioma", 2: "No Tumor", 3: "Pituitary"}
except Exception as e:
    print("Error reading label map:", e)
    CLASS_NAMES = {0: "Glioma", 1: "Meningioma", 2: "No Tumor", 3: "Pituitary"}

# ------------------ DISEASE INFO ------------------
DISEASE_INFO = {
  "Glioma": {
    "description": "A tumor that starts in the support cells of the brain (called glial cells). Its seriousness depends on how aggressive the tumor cells are (their grade) and where the tumor is located.",
    "cause": "Often due to random changes in cells' DNA. Sometimes previous head radiation can increase risk, but often the exact cause is unknown.",
    "treatment": "Usually involves surgery to remove as much of the tumor as possible, followed by radiation therapy or chemotherapy depending on the tumor's grade.",
    "symptoms": ["Headaches", "Seizures", "Vision problems", "Balance problems", "Speech or memory issues"],
    "example_image": "glioma.jpg"
  },
  "Meningioma": {
    "description": "A usually non-cancerous tumor that grows from the meninges (the protective layers covering the brain and spinal cord). It often grows slowly, and its effects depend on the tumor's size and location.",
    "cause": "Most occur by chance. Known risk factors include past head radiation and rare genetic conditions (like neurofibromatosis type 2).",
    "treatment": "If small and not causing issues, the tumor may just be watched closely. Otherwise, it is often removed with surgery. Precise radiation (radiosurgery) may be used if needed.",
    "symptoms": ["Headaches", "Seizures", "Vision or hearing changes", "Weakness or numbness"],
    "example_image": "meningioma.jpg"
  },
  "Pituitary": {
    "description": "Most are benign growths (adenomas) in the pituitary gland, a small hormone-producing gland at the base of the brain. These tumors can cause hormone changes or press on nerves that affect vision.",
    "cause": "Often no known cause. Rarely, inherited endocrine conditions (like MEN1 syndrome) can lead to pituitary tumors.",
    "treatment": "Small, symptom-free tumors may be monitored. If the tumor produces too much hormone, medications can help. Surgery or radiation is recommended for tumors that are large or causing symptoms.",
    "symptoms": ["Headaches", "Peripheral vision loss", "Hormone-related changes (e.g., menstrual irregularities or weight changes)", "Fatigue"],
    "example_image": "pituitary_adenoma.jpg"
  },
  "No Tumor": {
    "description": "No tumor was found on the MRI scan. The brain tissue appears healthy and normal.",
    "cause": "Not applicable (normal brain).",
    "treatment": "No treatment needed; follow normal health guidance.",
    "symptoms": ["None"],
    "example_image": "no_tumor.jpg"
  }
}


# ------------------ UTILITIES ------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def first_no_tumor_image():
    if os.path.exists(NORMAL_REFERENCE_IMAGE):
        return NORMAL_REFERENCE_IMAGE
    fallback = os.path.join(EXAMPLE_FOLDER, "no_tumor.jpg")
    return fallback if os.path.exists(fallback) else None

def preprocess_image_for_model(path, img_size=240):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Could not read image (cv2 returned None).")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    arr = img.astype(np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, 0)


# ------------------ PDF REPORT ------------------
def _draw_wrapped(c, text, x, y, max_width, font_name="Helvetica", font_size=11, leading=14):
    """Draw text with simple word wrapping; returns updated y."""
    c.setFont(font_name, font_size)
    words = text.split()
    line = ""
    lines = []
    for w in words:
        candidate = f"{line} {w}".strip()
        if c.stringWidth(candidate, font_name, font_size) <= max_width:
            line = candidate
        else:
            if line:
                lines.append(line)
            line = w
    if line:
        lines.append(line)
    for ln in lines:
        c.drawString(x, y, ln)
        y -= leading
    return y


def generate_pdf_report(prediction_dict, patient_overlay_bgr, normal_img_path, save_path):
    c = canvas.Canvas(save_path, pagesize=A4)
    width, height = A4
    left_margin = 40
    y = height - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(left_margin, y, "Medical AI Report - Brain Tumor Classification")
    y -= 30
    c.setFont("Helvetica", 10)
    c.drawString(left_margin, y, f"Report ID: {str(uuid.uuid4())}")
    c.drawString(left_margin + 300, y, f"Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    y -= 25
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_margin, y, "Prediction")
    y -= 18
    c.setFont("Helvetica", 11)
    c.drawString(left_margin, y, f"Label: {prediction_dict['label']}    Confidence: {prediction_dict['confidence']}")
    y -= 16
    desc = f"Description: {prediction_dict.get('description','-')}"
    cause = f"Cause: {prediction_dict.get('cause','-')}"
    y = _draw_wrapped(c, desc, left_margin, y, width - 2*left_margin, font_size=11, leading=14)
    y -= 2
    y = _draw_wrapped(c, cause, left_margin, y, width - 2*left_margin, font_size=11, leading=14)
    y -= 8
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_margin, y, "Symptoms")
    y -= 16
    c.setFont("Helvetica", 10)
    for s in prediction_dict.get("symptoms", [])[:8]:
        c.drawString(left_margin+12, y, f"• {s}")
        y -= 14
        if y < 110:
            c.showPage()
            y = height - 50
    y -= 10
    # Side-by-side: patient MRI vs normal reference
    try:
        slot_w = (width - 2*left_margin - 20) / 2  # two columns with gap
        slot_h = 260

        # Patient image
        overlay_rgb = cv2.cvtColor(patient_overlay_bgr, cv2.COLOR_BGR2RGB)
        patient_img = Image.fromarray(overlay_rgb)
        p_img = patient_img.copy()
        p_img.thumbnail((slot_w, slot_h))
        p_io = io.BytesIO()
        p_img.save(p_io, format='PNG')
        p_io.seek(0)
        p_reader = ImageReader(p_io)

        # Normal reference
        ref_reader = None
        if normal_img_path and os.path.exists(normal_img_path):
            ref_img = Image.open(normal_img_path).convert("RGB")
            r_img = ref_img.copy()
            r_img.thumbnail((slot_w, slot_h))
            r_io = io.BytesIO()
            r_img.save(r_io, format='PNG')
            r_io.seek(0)
            ref_reader = ImageReader(r_io)

        c.setFont("Helvetica-Bold", 12)
        c.drawString(left_margin, y, "Patient MRI")
        if ref_reader:
            c.drawString(left_margin + slot_w + 20, y, "Reference: Normal brain (no tumor)")
        y -= 12

        p_y = y - p_img.height
        c.drawImage(p_reader, left_margin, p_y, width=p_img.width, height=p_img.height)

        if ref_reader:
            r_y = y - r_img.height
            c.drawImage(ref_reader, left_margin + slot_w + 20, r_y, width=r_img.width, height=r_img.height)
            y = min(p_y, r_y) - 20
        else:
            y = p_y - 20

    except Exception:
        pass

    c.showPage()
    c.save()

# ------------------ PREDICTION ------------------
def predict_file(filepath):
    if model is None:
        raise RuntimeError("Model is not loaded on the server.")
    model_input = preprocess_image_for_model(filepath, img_size=240)
    preds = model.predict(model_input)
    idx = int(np.argmax(preds))
    label = CLASS_NAMES.get(idx, f"Class_{idx}")
    confidence = float(preds[0][idx]) * 100
    info = DISEASE_INFO.get(label, {})
    orig_bgr = cv2.imread(filepath)

    overlay = orig_bgr.copy()
    return {
        "label": label,
        "confidence": f"{confidence:.2f}%",
        "description": info.get("description",""),
        "cause": info.get("cause",""),
        "treatment": info.get("treatment",""),
        "symptoms": info.get("symptoms",[]),
        "example_image": info.get("example_image"),
        "overlay_bgr": overlay
        }

    # ------------------ ROUTES ------------------
@app.route("/", methods=["GET", "POST"])
def index():
    show_flag = request.args.get("show")

    # If user loads the page fresh (no show flag), start clean with intro only
    if request.method == "GET" and show_flag is None:
        for key in ["result", "image_url", "pdf_url", "batch_table"]:
            session.pop(key, None)

    # load from session if present (only kept when show flag is set after POST)
    result = session.get("result", None)
    image_url = session.get("image_url", None)
    pdf_url = session.get("pdf_url", None)
    batch_table = session.get("batch_table", None)
    role = request.args.get("role", session.get("role", "doctor"))

    if request.method == "POST":
        # clear previous session results
        session.pop("result", None)
        session.pop("image_url", None)
        session.pop("pdf_url", None)
        session.pop("batch_table", None)

        role = request.form.get("role", role)
        session["role"] = role

        # Batch upload?
        if "files" in request.files and request.files.getlist("files"):
            files = request.files.getlist("files")
            rows = []
            for f in files:
                if f and allowed_file(f.filename):
                    name = secure_filename(f.filename)
                    path = os.path.join(UPLOAD_FOLDER, f"{int(time.time())}_{name}")
                    f.save(path)
                    try:
                        res = predict_file(path)
                        rows.append({
                            "filename": name,
                            "label": res["label"],
                            "confidence": res["confidence"]
                        })
                    except Exception as e:
                        rows.append({"filename": name, "error": str(e)})
            session["batch_table"] = rows
            return redirect(url_for("index", show="1"))

        # Single file upload
        elif "file" in request.files:
            file = request.files.get("file")
            if not file or file.filename == "":
                flash("Please upload an image", "danger")
                return redirect(request.url)
            if not allowed_file(file.filename):
                flash("Invalid file", "danger")
                return redirect(request.url)
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, f"{int(time.time())}_{filename}")
            file.save(filepath)
            image_url_local = url_for("static", filename="uploads/" + os.path.basename(filepath))

            # predict
            try:
                res = predict_file(filepath)
            except Exception as e:
                flash(f"Prediction failed: {e}", "danger")
                return redirect(request.url)

            want_pdf = request.form.get("generate_pdf", "off") == "on"
            pdf_url_local = None
            if want_pdf:
                pdf_fname = f"report_{int(time.time())}_{os.path.splitext(filename)[0]}.pdf"
                pdf_path = os.path.join(UPLOAD_FOLDER, pdf_fname)
                normal_ref = first_no_tumor_image()
                if normal_ref is None:
                    normal_ref = os.path.join(EXAMPLE_FOLDER, "no_tumor.jpg")
                generate_pdf_report({
                    "label": res["label"],
                    "confidence": res["confidence"],
                    "description": res["description"],
                    "cause": res["cause"],
                    "symptoms": res["symptoms"]
                }, res["overlay_bgr"], normal_ref, pdf_path)
                pdf_url_local = url_for("static", filename="uploads/" + pdf_fname)

            example_url_local = None
            if res.get("example_image"):
                example_url_local = url_for("static", filename="disease_examples/" + res["example_image"])

            # store result in session, redirect -> GET (so refresh keeps the result)
            session["result"] = {
                "label": res["label"],
                "confidence": res["confidence"],
                "description": res["description"],
                "cause": res["cause"],
                "treatment": res["treatment"],
                "symptoms": res["symptoms"],
                "example_url": example_url_local
            }
            session["image_url"] = image_url_local
            session["pdf_url"] = pdf_url_local
            return redirect(url_for("index", show="1"))

    # GET: read session values for rendering. If show=1, display once then clear.
    result = session.get("result", None)
    image_url = session.get("image_url", None)
    pdf_url = session.get("pdf_url", None)
    batch_table = session.get("batch_table", None)
    role = session.get("role", role)

    render_args = dict(
        result=result,
        image_url=image_url,
        pdf_url=pdf_url,
        batch_table=batch_table,
        role=role,
        model_loaded=(model is not None)
    )

    # If this is the "show" page, clear stored results so refresh returns to intro
    if request.method == "GET" and request.args.get("show") == "1":
        for key in ["result", "image_url", "pdf_url", "batch_table"]:
            session.pop(key, None)

    return render_template("index.html", **render_args)

# ------------------ RUN APP ------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
