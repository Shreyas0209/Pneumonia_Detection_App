import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from utils.preprocessing import preprocess

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_URL = "https://drive.google.com/uc?export=download&id=1ZdOldz7646NP8bLbdQ5gdcVbIVmmuVZ9"
MODEL_PATH = "best_model2.keras"

def download_model():
    import os
    import requests
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        response = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Model downloaded successfully.")

download_model()

model = load_model(MODEL_PATH)

def assign_tier(prob):
    if prob <= 0.40:
        return "green", "Safe: Pneumonia very unlikely."
    elif prob <= 0.80:
        return "yellow", "Caution: Possible pneumonia detected."
    else:
        return "red", "Immediate Attention: High risk of pneumonia detected!"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            x = preprocess(filepath)
            prob = float(model.predict(x)[0][0])
            label = "PNEUMONIA" if prob > 0.5 else "NORMAL"
            tier, message = assign_tier(prob)
            confidence = prob if label == "PNEUMONIA" else 1 - prob
            return render_template('index.html',
                                   filename=filename,
                                   label=label,
                                   confidence=f"{confidence*100:.2f}%",
                                   tier=tier,
                                   message=message)
    return render_template('index.html')

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename=f'uploads/{filename}'))

if __name__ == '__main__':
    app.run(debug=True)

