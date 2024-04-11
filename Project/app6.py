# Import necessary libraries
from flask import Flask, render_template, request, Response
from flask_ngrok import run_with_ngrok
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json

# Initialize Flask app
app = Flask(__name__)
run_with_ngrok(app)  # To expose Flask app on ngrok when running locally

# Add a control variable for stopping the video stream
stop_detection = False

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

# List of emotions
EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Function to detect emotion in a frame
def detect_emotion(frame, model):
    global stop_detection

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

# Video stream function for real-time detection
def video_stream_camera():
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_emotion(frame, model)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

        if stop_detection:
            break  # Exit the loop if stop_detection is True

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(video_stream_camera(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_detection', methods=['POST'])
def stop_detection_callback():
    global stop_detection
    stop_detection = True
    print('Stop detection request received')
    return 'Stop detection'

@app.route('/reset_detection', methods=['POST'])
def reset_detection():
    global stop_detection
    stop_detection = False
    return 'Reset detection'

if __name__ == '__main__':
    app.run()
