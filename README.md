# Brain Tumor Classification (Glioma | Meningioma | Pituitary | No Tumor)

Deep-learning pipeline plus Flask UI for MRI-based tumor triage. EfficientNet-B2 is fine-tuned on the BT MRI dataset, provides browser predictions with PDF summaries.

## Features
- **Model**: EfficientNet-B2 (ImageNet init) → two-stage fine-tune (10 frozen, 40 unfrozen epochs).
- **Dataset**: ~7k MRIs (Training/Testing folders). Images live on disk/S3; MongoDB stores metadata/paths only.
- **Inference App**: Drag/drop MRI, confidence, disease info, patient vs. normal MRI comparison in downloadable PDF.
- **Reports**: ReportLab PDF with prediction, description, causes, symptoms, and healthy-reference image.
- **MLOps (in-progress)**: `constants`, ingestion/validation/transformation/trainer stubs, AWS S3 registry plan, Docker/GitHub Actions roadmap.

## Structure
```
BRAIN-TUMOR-CLASSIFICATION/
├── app/ (Flask app, templates, static, models/)
├── config/ (schema/model configs)
├── notebooks/ (Colab training)
├── src/ (pipeline modules: ingestion, validation, transformation, trainer, etc.)
├── artifact/ (pipeline outputs, gitignored)
├── models/ (final_effnetb2.h5, label_map.json)
├── requirements.txt
├── template.py
└── README.md
```

## Environment
```
conda create -n brain_tumor python=3.10 -y
conda activate brain_tumor
pip install -r requirements.txt
pip list
```

## Training Notebook
1. Open `notebooks/brain_tumor_classification_using_CNN.ipynb` (Colab).
2. Mount dataset to `/content/data/BT_MRI_Dataset/BT_MRI_Dataset`.
3. Run generators → EfficientNetB2 build → stage-1 + stage-2 training → evaluation.
4. Save `final_effnetb2.h5` to `app/models/`.

## Flask Inference
```
cd app
set FLASK_APP=flask_app.py
python flask_app.py
# visit http://127.0.0.1:5000
`