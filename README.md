# Twin-Aware Face Recognition Access Control System using ESP32
---

# Overview

Traditional face recognition systems perform well under normal conditions but often struggle to distinguish **highly similar individuals**, such as twins or siblings. This project addresses that limitation by introducing a **hybrid recognition pipeline** that combines deep facial embeddings with **Linear Discriminant Analysis (LDA)** to improve recognition accuracy in ambiguous cases.

The system consists of three integrated layers:

- **Computer Vision Layer** for face detection and recognition.
- **Machine Learning Layer** for intelligent ambiguity resolution.
- **Embedded Hardware Layer** for real-time physical access control using an ESP32.

Recognition is performed on a computer using Python, while authorization results are transmitted over Wi-Fi to an **ESP32**, which controls the hardware peripherals responsible for granting or denying access.

This project demonstrates a complete **AI-enabled Embedded Access Control System**, integrating software intelligence with real-world hardware.

---

# Features

- AI-based Face Recognition
- Twin-aware Recognition using Linear Discriminant Analysis (LDA)
- Real-Time Webcam Recognition
- 128-Dimensional Face Embeddings
- Automatic Similarity Detection
- Multi-Face Recognition
- Known Person Priority Logic
- HTTP-based Communication
- Wi-Fi Enabled Embedded System
- ESP32-based Hardware Control
- Modular Software Architecture

---

# System Architecture
<img width="975" height="912" alt="image" src="https://github.com/user-attachments/assets/ae09f362-58e9-4d18-be26-b809a63ced88" />

---
# Circuit Diagram 
<img width="940" height="513" alt="image" src="https://github.com/user-attachments/assets/4cb935e9-f905-43dd-b0c5-f74a630f55d5" />

---
# Software Pipeline

## 1️. Dataset Preparation

A structured facial image dataset is used for training.

Each face image is converted into a **128-dimensional embedding vector** using the **face_recognition** library built on **dlib**.

---

## 2️. Similarity Detection

The system computes the average Euclidean distance between every pair of registered users.

Pairs whose embeddings fall below a predefined threshold are automatically marked as **visually similar identities**.

---

## 3️. Twin Recognition using LDA

Instead of applying an expensive classifier to every recognition result, an independent **Linear Discriminant Analysis (LDA)** classifier is trained **only** for visually similar identity pairs.

This selective approach improves recognition accuracy while maintaining computational efficiency.

---

## 4️.Real-Time Recognition

The recognition engine performs the following operations:

- Capture image from webcam
- Detect faces
- Generate face embeddings
- Compare embeddings with stored database
- Resolve ambiguous matches using LDA
- Determine authorization status

---

## 5️.Communication Layer

Once recognition is completed, the system creates a JSON payload containing the recognized person's name and authorization status.

Example:

```json
{
    "name": "Harini",
    "authorized": true
}
```

The payload is transmitted to the ESP32 over the local Wi-Fi network using an **HTTP POST** request.

---

# Embedded Hardware

The ESP32 serves as the embedded controller responsible for executing the physical access response.

### Responsibilities

- Connect to Wi-Fi
- Host an HTTP Web Server
- Receive recognition results
- Parse JSON requests
- Control GPIO peripherals
- Execute access control logic

### Hardware Components

| Component | Purpose |
|-----------|---------|
| ESP32 DevKit | Embedded Controller |
| Webcam | Image Acquisition |
| Green LED | Access Granted Indicator |
| Red LED | Access Denied Indicator |
| Active Buzzer | Unauthorized Alert |
| DC Motor | Door Unlock Simulation |
| Motor Driver | Controls DC Motor |

---

# Technologies Used

| Category | Technology |
|----------|------------|
| Programming Language | Python |
| Embedded Programming | Arduino IDE |
| Computer Vision | OpenCV |
| Face Recognition | face_recognition (dlib) |
| Machine Learning | Scikit-Learn |
| Numerical Computing | NumPy |
| Networking | HTTP / REST API |
| Model Serialization | Pickle |
| Embedded Platform | ESP32 |

---

## Install Dependencies

```bash
pip install -r requirements.txt
```
---

## Train the Recognition Model

```bash
python training.py
```

This generates the serialized model:

```
face_model.pkl
```
---

## Run the Recognition System

```bash
python server.py
```

The application will:

- Open the webcam
- Detect faces
- Recognize registered users
- Resolve ambiguous identities using LDA
- Send authorization results to the ESP32

---

## Hardware Prototype

<img width="954" height="432" alt="image" src="https://github.com/user-attachments/assets/a2728db9-8df5-48ea-b4d2-45fcef9efbd3" />

---

# Project Highlights

- AI-powered Embedded System
- Twin-aware Face Recognition
- Embedded IoT Access Control
- Computer Vision with OpenCV
- Machine Learning using LDA
- Wi-Fi Communication using HTTP
- ESP32 Web Server
- Real-Time Hardware Response
- Modular Software Design
- End-to-End AI + Embedded Integration

---

# Contributors
- S. Deepika Sri -[https://github.com/SDeepikaSri]
- G. Akshaya-[]

---

# License

This project is licensed under the MIT License.

---

# Author

**Harini S**

---
