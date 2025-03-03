import requests

url = "http://127.0.0.1:5000/chat"
data = {"message": "I have a fever and cough.", "age": 30, "gender": "Male", "blood_pressure": "Normal", "cholesterol": "High"}

response = requests.post(url, json=data)
print(response.json())  # Should return AI response
import nest_asyncio
import threading
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pyttsx3
import spacy
from transformers import pipeline
from werkzeug.serving import run_simple

# Apply the fix for Jupyter
nest_asyncio.apply()

# Initialize Flask app
app = Flask(__name__)

# Load models
nlp = spacy.load("en_core_web_sm")
qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")
engine = pyttsx3.init()

# Load ML models
model = joblib.load("disease.pkl")
encoder = joblib.load("label_encoder.pkl")

# Function: Convert text to speech
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Function: Predict disease
def predict_disease(symptoms, age, gender, blood_pressure, cholesterol):
    gender = 1 if gender.lower() == "male" else 0
    bp_map = {"Low": 0, "Normal": 1, "High": 2}
    chol_map = {"Low": 0, "Normal": 1, "High": 2}

    blood_pressure = bp_map.get(blood_pressure, 1)
    cholesterol = chol_map.get(cholesterol, 1)

    symptoms_vector = [1 if s in symptoms else 0 for s in ["Fever", "Cough", "Fatigue", "Difficulty Breathing"]]
    input_data = np.array([symptoms_vector + [age, gender, blood_pressure, cholesterol]]).reshape(1, -1)

    disease_index = model.predict(input_data)[0]
    predicted_disease = encoder.inverse_transform([disease_index])[0]

    return predicted_disease

# Default homepage
@app.route("/", methods=["GET"])
def home():
    return "Flask API is running! Use POST on /chat to interact."

# Flask API Route
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        user_input = data.get("message", "").strip().lower()

        if user_input in ["exit", "quit", "bye"]:
            response = "Take care! Stay healthy!"
        else:
            doc = nlp(user_input)
            symptoms = [token.text for token in doc.ents if token.label_ == "SYMPTOM"]

            if symptoms:
                patient_age = data.get("age", 30)
                patient_gender = data.get("gender", "Male")
                patient_bp = data.get("blood_pressure", "Normal")
                patient_cholesterol = data.get("cholesterol", "Normal")

                predicted_disease = predict_disease(symptoms, patient_age, patient_gender, patient_bp, patient_cholesterol)
                response = f"Based on the symptoms, you might have {predicted_disease}. Please consult a doctor."
            else:
                response = qa_model(question=user_input, context="Common medical knowledge.")["answer"]

        speak(response)
        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run Flask without blocking Jupyter Notebook
def run_flask():
    run_simple("localhost", 5000, app, use_reloader=False)

flask_thread = threading.Thread(target=run_flask)
flask_thread.start()
