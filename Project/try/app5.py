from flask import Flask, render_template, Response
import cv2
import tkinter as tk
from tkinter import filedialog
from tkinter import *
import threading
from sklearn import metrics
from tensorflow.keras.models import model_from_json
from PIL import Image, ImageTk
import numpy as np
import os

app = Flask(__name__)

# Add a control variable for stopping the video stream
stop_detection = False

# Keep track of detected emotions for the video
video_emotions = []

def stop_detection_callback():
    global stop_detection
    stop_detection = True

def reset_stop_detection():
    global stop_detection
    stop_detection = False

def FacialExpressionModel(json_file, weights_file):
    with open(json_file, "r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def detect_emotion(frame, model):
    if stop_detection:
        return frame  # Do not process the frame if stop button is pressed

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_roi = gray_frame[y:y + h, x:x + w]
        roi = cv2.resize(face_roi, (48, 48))
        emotion_pred = EMOTIONS_LIST[np.argmax(model.predict(roi[np.newaxis, :, :, np.newaxis]))]

        cv2.putText(frame, emotion_pred, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame

def detect_emotion_and_update(frame, model):
    global video_emotions

    if stop_detection:
        return frame  # Do not process the frame if stop button is pressed

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    frame_emotions = []  # Emotions detected in the current frame

    for (x, y, w, h) in faces:
        face_roi = gray_frame[y:y + h, x:x + w]
        roi = cv2.resize(face_roi, (48, 48))
        emotion_pred = EMOTIONS_LIST[np.argmax(model.predict(roi[np.newaxis, :, :, np.newaxis]))]

        frame_emotions.append(emotion_pred)

        cv2.putText(frame, emotion_pred, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Append the detected emotions for this frame to the overall video_emotions list
    video_emotions.extend(frame_emotions)

    return frame

def gen_frames(source):
    global video_emotions
    cap = cv2.VideoCapture(source)  # Use 0 for the default camera or file path for uploaded video
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_emotion_and_update(frame, model)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = ImageTk.PhotoImage(image=frame)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))[1].tobytes() + b'\r\n')

        if stop_detection:
            break  # Exit the loop if stop_detection is True

    cap.release()

    # Calculate the overall emotion of the video
    overall_emotion = max(set(video_emotions), key=video_emotions.count)
    print("Overall Emotion of the Video:", overall_emotion)
    video_emotions = []  # Reset the list for the next video

def check_camera_access():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return False
    cap.release()
    return True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(0), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    reset_stop_detection()  # Reset stop_detection before starting the new video stream
    file_path = filedialog.askopenfilename(filetypes=[("Video files", ".mp4;.avi;*.mkv")])
    if file_path:
        return Response(gen_frames(file_path), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "No file selected"

@app.route('/stop_detection', methods=['POST'])
def stop_detection_route():
    stop_detection_callback()
    return "Detection stopped"

@app.route('/reset_detection', methods=['POST'])
def reset_detection_route():
    reset_stop_detection()
    return "Detection reset"

@app.route('/start_real_time_detection', methods=['POST'])
def start_real_time_detection():
    if check_camera_access():
        return Response(gen_frames(0), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Camera access denied"

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    face_cascade = cv2.CascadeClassifier(os.path.join(current_dir, 'haarcascade_frontalface_default.xml'))
    model_json_path = os.path.join(current_dir, 'C:\\BE MPR Projects\\Emotion_Detection_GUI\\model_a1K.json')
    model_weights_path = os.path.join(current_dir, 'C:\\BE MPR Projects\\Emotion_Detection_GUI\\model_weights1K.h5')
    model = FacialExpressionModel(model_json_path, model_weights_path)
    EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
    app.run(debug=True)

def gen_frames(source):
    global video_emotions
    cap = cv2.VideoCapture(source)  # Use 0 for the default camera or file path for uploaded video
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame")
            break

        frame = detect_emotion_and_update(frame, model)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = ImageTk.PhotoImage(image=frame)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))[1].tobytes() + b'\r\n')

        if stop_detection:
            break  # Exit the loop if stop_detection is True

    cap.release()
    print("Video capture released")

    # Calculate the overall emotion of the video
    overall_emotion = max(set(video_emotions), key=video_emotions.count)
    print("Overall Emotion of the Video:", overall_emotion)
    video_emotions = []  # Reset the list for the next video