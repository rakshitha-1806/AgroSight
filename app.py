from flask import Flask, render_template, request, redirect, jsonify
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
# --- NEW IMPORTS for Gemini ---
from google import genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# -----------------------------

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# -----------------------------
# Initialize Gemini Client
# -----------------------------
try:
    # The client automatically picks up GEMINI_API_KEY from os.environ
    gemini_client = genai.Client()
    print("Gemini client initialized successfully!")
except Exception as e:
    print(f"Could not initialize Gemini client. Error: {e}")
    gemini_client = None

# -----------------------------
# Load trained model
# -----------------------------
MODEL_PATH = 'plant_model.h5'

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Could not load model. Error: {e}")
    model = None

# -----------------------------
# Disease classes and prescriptions
# -----------------------------
class_names = [
    'Pepper_bell_Bacterial_spot',
    'Pepper_bell_healthy',
    'Potato_Early_blight',
    'Potato_healthy',
    'Potato_Late_blight',
    'Tomato_Target_Spot',
    'Tomato__Tomato_mosaic_virus',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_healthy',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite'
]

prescriptions = {
    'Pepper_bell_Bacterial_spot': ("Remove infected leaves and avoid overhead watering.", "Neem oil spray"),
    'Pepper_bell_healthy': ("No action needed, keep monitoring.", "N/A"),
    'Potato_Early_blight': ("Remove infected foliage, practice crop rotation.", "Copper-based biofungicide"),
    'Potato_healthy': ("No action needed.", "N/A"),
    'Potato_Late_blight': ("Remove and destroy infected plants immediately.", "Bordeaux mixture"),
    'Tomato_Target_Spot': ("Prune infected leaves, maintain good air circulation.", "Neem oil"),
    'Tomato__Tomato_mosaic_virus': ("Remove infected plants, sanitize tools.", "No chemical cure, supportive care only"),
    'Tomato__Tomato_YellowLeaf__Curl_Virus': ("Control whiteflies, remove infected plants.", "Introduce natural predators"),
    'Tomato_Bacterial_spot': ("Remove affected leaves, avoid splashing water.", "Copper-based spray"),
    'Tomato_Early_blight': ("Remove lower leaves and infected areas.", "Neem oil or Bacillus subtilis"),
    'Tomato_healthy': ("No action needed.", "N/A"),
    'Tomato_Late_blight': ("Remove infected leaves, destroy infected debris.", "Copper fungicide"),
    'Tomato_Leaf_Mold': ("Increase air circulation, remove infected leaves.", "Sulfur-based spray"),
    'Tomato_Septoria_leaf_spot': ("Remove fallen leaves and infected parts.", "Neem oil"),
    'Tomato_Spider_mites_Two_spotted_spider_mite': ("Spray water to dislodge mites, introduce predators.", "Insecticidal soap or neem oil")
}

# -----------------------------
# Predict function
# -----------------------------
def predict_disease(img_path):
    if model is None:
        return "Model not loaded", "N/A", "N/A"

    img = image.load_img(img_path, target_size=(128,128))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    class_idx = np.argmax(preds, axis=1)[0]

    if class_idx >= len(class_names):
        return "Unknown Disease", "N/A", "N/A"

    disease = class_names[class_idx]
    prescription, pesticide = prescriptions.get(disease, ("No prescription available.", "N/A"))
    return disease, prescription, pesticide

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET","POST"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return redirect(request.url)
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    print("File saved at:", filepath)

    disease, prescription, pesticide = predict_disease(filepath)
    print("Prediction:", disease)

    return render_template("result.html",
                            prediction=disease,
                            prescription=prescription,
                            pesticide=pesticide,
                            filename=filename)

# -----------------------------
# NEW Route for Gemini Chatbot
# -----------------------------
@app.route("/ask_expert", methods=["POST"])
def ask_expert():
    if gemini_client is None:
        return jsonify({"response": "Chatbot is unavailable. API key or service error."}), 503

    # Get the user's question from the AJAX request
    user_query = request.json.get("question")
    if not user_query:
        return jsonify({"response": "Please enter a question."}), 400

    # System instruction to define the AI's persona
    system_instruction = (
        "You are an expert AI Agricultural Consultant and Plant/Farming Expert. "
        "Answer all questions related to plants, gardening, farming, soil health, "
        "and pest management. Keep your tone informative, concise, and helpful."
    )

    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[user_query],
            config={
                "system_instruction": system_instruction,
                "temperature": 0.5 
            }
        )
        return jsonify({"response": response.text})
    except Exception as e:
        print(f"Gemini API error: {e}")
        return jsonify({"response": "Sorry, I'm having trouble connecting to the AI expert right now. Please try again."}), 500


if __name__ == "__main__":
    # Ensure debug=True only during development
    app.run(debug=True)