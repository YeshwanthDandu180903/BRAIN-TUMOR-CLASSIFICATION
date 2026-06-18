# 🧠 Brain Tumor Classification & Triage System

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-lightgrey?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Test Accuracy](https://img.shields.io/badge/Test%20Accuracy-98%25-brightgreen)](#results--evaluation)

An end-to-end medical AI imaging project that leverages a fine-tuned **EfficientNet-B2** model to classify brain MRI scans. The application features an interactive **Flask-based web interface** with drag-and-drop capability, real-time predictions, light/dark mode toggling, and automated **ReportLab PDF medical report generation** complete with a healthy reference image.

---

## 🔍 Key Features

- **High-Accuracy Deep Learning Classifier**: Fine-tuned EfficientNet-B2 achieving **98% classification accuracy** on test data.
- **Multi-Class Detection**: Detects and categorizes MRIs into four distinct states:
  - 🔴 **Glioma**
  - 🟡 **Meningioma**
  - 🔵 **Pituitary**
  - 🟢 **No Tumor** (Healthy reference scan)
- **Interactive Flask Web Application**:
  - **Drag-and-Drop** MRI upload interface with immediate preview.
  - Interactive **Light/Dark Mode** theme toggle.
  - Dynamic prediction loading spinners.
- **Automated PDF Report Generation**: 
  - Instantly generates clinical reports featuring prediction labels, confidence scores, and disease symptoms/descriptions.
  - Side-by-side comparison of the patient's MRI scan with a **normal brain reference MRI**.
- **Jupyter Notebook**: Clean training script with stage-by-stage learning rate schedules and model evaluation plots.

---

## 🖥️ User Interface & Demos

### 1. Main Dashboard (Light & Dark Theme)
The application provides a seamless, modern layout with a clean responsive grid and interactive drop zones.

|  Home page | Dashboard |
| --- | --- |
| ![Light Mode](results/img1.png) | ![Dark Mode](results/img2.png) |

### 2. Clinical PDF Reports
Generate and download diagnostic-ready medical reports automatically detailing classification results, clinical symptoms, and a reference comparison.

<div align="center">
  <img src="results/img33.png" alt="PDF Report View" width="70%">
</div>

### 3. Application Walkthrough
Watch the system analyze scan files in real-time.

<div align="center">
  <img src="results/video.gif" alt="System Demo" width="90%">
</div>

---

## 📁 Project Structure

```text
BRAIN-TUMOR-CLASSIFICATION/
├── app/
│   ├── models/
│   │   ├── final_effnetb2.h5       # Fine-tuned EfficientNet-B2 weights (gitignored)
│   │   └── label_map.json          # Index-to-label mapping dictionary
│   ├── static/
│   │   ├── disease_examples/       # Healthy reference image & disease samples
│   │   └── uploads/                # Temporary user-uploaded scans & generated PDFs
│   ├── templates/
│   │   └── index.html              # Frontend bootstrap layout with dark mode script
│   └── flask_app.py                # Main Flask application and PDF generator logic
├── notebooks/
│   └── brain_tumor_classification_using_CNN.ipynb  # End-to-end train & evaluation notebook
├── results/
│   ├── img1.png                # Light-mode screenshot
│   ├── img2.png                # Dark-mode screenshot
│   ├── img33.png               # PDF report preview
│   └── video.gif               # Animated walkthrough
├── .gitignore                  # Git exclusions (data, large model weights, etc.)
├── LICENSE                     # MIT license details
├── requirements.txt            # Project dependencies list
└── README.md                   # Project documentation
```

---

## ⚙️ Installation & Setup

Follow these steps to run the project locally on your system:

### 1. Clone the Repository
```bash
git clone https://github.com/YeshwanthDandu180903/BRAIN-TUMOR-CLASSIFICATION.git
cd BRAIN-TUMOR-CLASSIFICATION
```

### 2. Setup Virtual Environment
We recommend using **Conda** or **venv** with Python 3.10:

**Using Conda:**
```bash
conda create -n brain_tumor python=3.10 -y
conda activate brain_tumor
```

**Using venv:**
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Usage Guide

### Running the Web Application
1. Start the Flask application by running:
   ```bash
   cd app
   python flask_app.py
   ```
2. Open your web browser and go to:
   ```text
   http://127.0.0.1:5000/
   ```
3. Drag and drop a brain MRI scan image (supported formats: `.png`, `.jpg`, `.jpeg`), check **Generate PDF medical report**, and click **Predict** to view results.

### Training the Model
If you want to retrain the model on your dataset:
1. Open the Jupyter Notebook located at:
   ```text
   notebooks/brain_tumor_classification_using_CNN.ipynb
   ```
2. Download or map the dataset (expected shape: folders matching `Training` and `Testing` containing sub-directories for each class).
3. Run through the notebook cells to ingest data, execute transfer learning (EfficientNet-B2 base + custom dense layers), fine-tune the model, and export the `.h5` model file.
4. Save the generated `final_effnetb2.h5` file inside `app/models/`.

---

## 📊 Results & Evaluation

The model was trained using transfer learning on `EfficientNet-B2` with ImageNet initialization, utilizing two-stage fine-tuning:
1. **Stage 1 (Feature Extraction)**: 10 epochs training head classifiers only with base layers frozen.
2. **Stage 2 (Fine-tuning)**: 40 epochs with all layers unfrozen and a low learning rate.

### Performance metrics on held-out test split:
- **Test Accuracy**: **98%**
- **Robustness**: High confidence classification across tumor categories (Glioma, Meningioma, Pituitary) and clean brain scans (No Tumor).

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contact & Contribution

For questions, issues, or contributions, please contact:
- **Author**: Dandu Yeshwanth
- **GitHub**: [@YeshwanthDandu180903](https://github.com/YeshwanthDandu180903)
- **Project Link**: [Brain Tumor Classification](https://github.com/YeshwanthDandu180903/BRAIN-TUMOR-CLASSIFICATION)

*Disclaimer: This application is a prototype for educational and research purposes and should not be used as a replacement for professional medical diagnosis.*
