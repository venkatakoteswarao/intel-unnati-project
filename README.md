# AI-Powered Interactive Learning Assistant for Classrooms

## 🚀 Overview

**AI-Powered Interactive Learning Assistant** is a multimodal AI system designed to enhance classroom learning experiences by enabling real-time interaction between students and an AI assistant. It processes **text**, **voice**, and **visual queries** and provides **contextual answers**, **visual aids**, and **student engagement monitoring** via facial expressions and head pose detection.

---

## 🎯 Objective

- Dynamically answer student queries in real-time using:
  - 📝 Text input
  - 🎤 Voice input (speech-to-text)
  - 🖼️ Visual input (images, diagrams)
- Analyze student **engagement and attention** via webcam.
- Provide **contextual explanations** with **charts and diagrams**.

---

## 🛠 Features

| Feature                          | Description |
|----------------------------------|-------------|
| ✅ Text & Voice-based QA         | Ask questions via text or mic. Uses DistilBERT for answering. |
| ✅ Visual Question Answering     | Upload screenshots/diagrams and ask related questions. |
| ✅ Engagement Detection          | Detect emotion (happy/sad/angry), yaw/pitch/roll of head using webcam. |
| ✅ Disengagement Alerts          | Triggers warning if student looks away or shows disengaged emotion. |
| ✅ Flask                         | Interactive web-based interface combining all modules. |

---

## 🧠 Models Used

| Task                           | Model Name                                                   | Framework      |
|--------------------------------|---------------------------------------------------------------|----------------|
| Text QA                        | `distilbert-base-uncased-distilled-squad`                    | Hugging Face   |
| Visual QA                      | `Salesforce/blip-vqa-base` *(can be replaced with ONNX)*     | Transformers   |
| Emotion Recognition            | `emotions-recognition-retail-0003`                           | OpenVINO       |
| Head Pose Estimation           | `head-pose-estimation-adas-0001`                             | OpenVINO       |
| Face Detection                 | `face-detection-retail-0004`                                 | OpenVINO       |

---

## 🧰 Tech Stack

- Python 3.8+
- [Hugging Face Transformers](https://huggingface.co/)
- [OpenVINO Toolkit](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html)
- [OpenCV](https://opencv.org/)
- PyTorch / TensorFlow (optional for QA)
- SpeechRecognition + PyAudio (voice input)
- Tesseract OCR (for visual text)

---


