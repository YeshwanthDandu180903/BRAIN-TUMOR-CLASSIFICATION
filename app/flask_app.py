import os
import numpy as np
import cv2
from flask import Flask, request, render_template, url_for
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.h5')
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Model
model = None
try:
    print(f"Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure you have downloaded the model file and placed it in the 'models' directory.")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    # Read image
    img = cv2.imread(image_path)
    # Resize to match model input (240x240)
    img = cv2.resize(img, (240, 240))
    # Normalize
    img = img / 255.0
    # Expand dims to match batch shape (1, 240, 240, 3)
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_url = None
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        
        file = request.files['file']
        
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            image_url = url_for('static', filename='uploads/' + filename)
            
            if model:
                try:
                    # Preprocess and Predict
                    processed_img = preprocess_image(filepath)
                    pred_prob = model.predict(processed_img)
                    
                    # Assuming binary classification: 0 = No Tumor, 1 = Tumor
                    # Adjust based on your specific model output (softmax vs sigmoid)
                    if pred_prob.shape[1] == 2:
                        # Softmax output [prob_no, prob_yes]
                        class_idx = np.argmax(pred_prob)
                        confidence = pred_prob[0][class_idx] * 100
                        result = "Tumor Detected (Yes)" if class_idx == 1 else "No Tumor (No)"
                    else:
                        # Sigmoid output [prob]
                        confidence = pred_prob[0][0]
                        if confidence > 0.5:
                            result = "Tumor Detected (Yes)"
                            confidence = confidence * 100
                        else:
                            result = "No Tumor (No)"
                            confidence = (1 - confidence) * 100
                            
                    prediction = {
                        'result': result,
                        'confidence': f"{confidence:.2f}%"
                    }
                except Exception as e:
                    return render_template('index.html', error=f"Prediction Error: {e}", image_url=image_url)
            else:
                return render_template('index.html', error="Model not loaded.", image_url=image_url)

    return render_template('index.html', prediction=prediction, image_url=image_url)

if __name__ == '__main__':
    app.run(debug=True)
