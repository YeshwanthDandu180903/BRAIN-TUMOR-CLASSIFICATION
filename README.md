# BRAIN-TUMOR-CLASSIFICATION
Implementing the Brain Tumor Classification using CNN architecture in deep learning


# Brain Tumor Detection — Streamlit UI

This repository contains a Streamlit UI to run predictions with the provided `model.h5` (brain tumor detection).

How to run

1. Create a Python environment (recommended) and install dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Start the Streamlit app from the project root:

```powershell
streamlit run app.py
```

Usage notes & assumptions

- The app looks for `model.h5` in the repository root. Place your trained Keras model there.
- The app will try to infer the model's input shape. If your model expects a different preprocessing (mean/std normalization, different size, or grayscale input), edit `preprocess_image` in `app.py` accordingly.
- The app supports both a binary sigmoid output (single neuron) and a 2-class softmax. If your model has a different label ordering, update the labels mapping in `predict_image`.

Troubleshooting

- If loading the model fails with a TensorFlow error, ensure your Python environment has a TensorFlow build compatible with your OS and CPU/GPU. On Windows a CPU-only install is usually `pip install tensorflow` (GPU-enabled installs require extra steps).
- If Streamlit starts but the UI shows no sample images, add sample images under folders like `tumorous_and_nontumorous`, `brain_tumor_dataset`, `augmented_data` or simple `yes`/`no` dirs — the app will attempt to locate images automatically.

Next steps (optional)

- Add a small test script that loads `model.h5` and runs a prediction on a known sample to validate the end-to-end flow.
- Expand the UI with Grad-CAM visualization to explain predictions.
