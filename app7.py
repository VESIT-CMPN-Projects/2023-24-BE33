from typing import List
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import streamlit
from tensorflow.keras.models import model_from_json

# Initialize FastAPI app
app = FastAPI()

# List of emotions
EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the emotion detection model
def FacialExpressionModel(json_file, weights_file):
    with open(json_file, "r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

model = FacialExpressionModel("model_a1K.json", "model_weights1K.h5")

# Function to detect emotion in a frame
def detect_emotion(frame, model):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    emotion_counts = [0] * len(EMOTIONS_LIST)

    for (x, y, w, h) in faces:
        face_roi = gray_frame[y:y + h, x:x + w]
        roi = cv2.resize(face_roi, (48, 48))
        emotion_pred = EMOTIONS_LIST[np.argmax(model.predict(roi[np.newaxis, :, :, np.newaxis]))]

        emotion_counts[EMOTIONS_LIST.index(emotion_pred)] += 1

    # Calculate the average emotion
    overall_emotion = EMOTIONS_LIST[np.argmax(emotion_counts)]

    return frame, overall_emotion

@app.websocket("/video_feed")
async def video_feed(websocket: WebSocket):
    await websocket.accept()

    cap = cv2.VideoCapture(0)  # Use 0 for the default camera
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, overall_emotion = detect_emotion(frame, model)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ret, jpeg = cv2.imencode('.jpg', frame)
        await websocket.send_bytes(jpeg.tobytes())

    cap.release()
    await websocket.close()

@app.get("/overall_emotion")
def get_overall_emotion():
    return {"emotion": overall_emotion}