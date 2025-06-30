from flask import Flask, render_template, request, jsonify
import base64
import requests
import speech_recognition as sr
import numpy as np
import cv2
from openvino.runtime import Core
import tempfile
import subprocess
import os

app = Flask(__name__)

# ------------------------------
# Gemini API Setup
# ------------------------------
API_KEY = "AIzaSyB2Nl2NXTO2IQQ96Owvo8snt6YKXr58q0k"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

def build_text_part(text):
    return {"text": text}

def build_image_part_from_bytes(img_bytes):
    b64_data = base64.b64encode(img_bytes).decode("utf-8")
    return {"inlineData": {"mimeType": "image/jpeg", "data": b64_data}}

def get_gemini_response(parts):
    body = {"contents": [{"parts": parts}]}
    headers = {"Content-Type": "application/json"}
    resp = requests.post(GEMINI_URL, headers=headers, json=body)
    if resp.ok:
        try:
            return resp.json()['candidates'][0]['content']['parts'][0]['text']
        except:
            return "🤖 Gemini: Couldn't parse the response."
    return f"❌ Error {resp.status_code}: {resp.text}"

# ------------------------------
# OpenVINO Setup
# ------------------------------
ie = Core()

face_model = ie.read_model(r"
face-detection-retail-0004.xml")
face_exec = ie.compile_model(face_model, "CPU")
face_out = face_exec.output(0)

emotion_model = ie.read_model(r"C:\Users\s.v.koteswarao\intel\emotions-recognition-retail-0003\FP16\emotions-recognition-retail-0003.xml")
emotion_exec = ie.compile_model(emotion_model, "CPU")
emotion_out = emotion_exec.output(0)
emotions = ['neutral', 'happy', 'sad', 'surprise', 'anger']

head_model = ie.read_model(r"C:\Users\s.v.koteswarao\intel\head-pose-estimation-adas-0001\FP16\head-pose-estimation-adas-0001.xml")
head_exec = ie.compile_model(head_model, "CPU")
head_out_yaw = head_exec.output("angle_y_fc")
head_out_pitch = head_exec.output("angle_p_fc")
head_out_roll = head_exec.output("angle_r_fc")

# ------------------------------
# Speech Recognizer
# ------------------------------
recognizer = sr.Recognizer()

# ------------------------------
# Routes
# ------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['image']
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    h, w = img.shape[:2]
    resized = cv2.resize(img, (300, 300))
    blob = resized.transpose(2, 0, 1)[np.newaxis, :]

    detections = face_exec([blob])[face_out][0][0]

    disengaged = False
    detected_emotion = "neutral"
    yaw, pitch, roll = 0.0, 0.0, 0.0

    for det in detections:
        conf = det[2]
        if conf > 0.5:
            xmin = int(det[3] * w)
            ymin = int(det[4] * h)
            xmax = int(det[5] * w)
            ymax = int(det[6] * h)

            face_img = img[ymin:ymax, xmin:xmax]
            if face_img.size == 0:
                continue

            em_face = cv2.resize(face_img, (64, 64))
            em_face = em_face.transpose(2, 0, 1)[np.newaxis, :]
            emotion_result = emotion_exec([em_face])[emotion_out]
            detected_emotion = emotions[np.argmax(emotion_result)]

            hp_face = cv2.resize(face_img, (60, 60))
            hp_face = hp_face.transpose(2, 0, 1)[np.newaxis, :]
            yaw = head_exec([hp_face])[head_out_yaw][0][0]
            pitch = head_exec([hp_face])[head_out_pitch][0][0]
            roll = head_exec([hp_face])[head_out_roll][0][0]

            if abs(yaw) > 30 or detected_emotion in ["sad", "anger"]:
                disengaged = True

            break

    return jsonify({"disengaged": disengaged, "emotion": detected_emotion})

@app.route('/ask', methods=['POST'])
def ask_text():
    question = request.form.get('question', '')
    if not question.strip():
        return jsonify({"answer": "Please enter a valid question."})
    parts = [build_text_part(question)]
    answer = get_gemini_response(parts)
    return jsonify({"answer": answer})

@app.route('/ask_voice', methods=['POST'])
def ask_voice():
    audio_file = request.files.get('audio')
    if not audio_file:
        return jsonify({"error": "No audio file uploaded."})

    try:
        # Save uploaded file to temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
            audio_file.save(temp_wav.name)

        # Use SpeechRecognition
        with sr.AudioFile(temp_wav.name) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)

        parts = [build_text_part(text)]
        answer = get_gemini_response(parts)

        return jsonify({"transcription": text, "answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/ask_image', methods=['POST'])
def ask_image():
    prompt = request.form.get('prompt', '')
    image_file = request.files['image']
    if not prompt.strip() or not image_file:
        return jsonify({"answer": "Please upload an image and enter a prompt."})
    img_bytes = image_file.read()
    image_part = build_image_part_from_bytes(img_bytes)
    parts = [build_text_part(prompt), image_part]
    answer = get_gemini_response(parts)
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(debug=True)
