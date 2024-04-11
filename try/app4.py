from flask import Flask, render_template, request, Response, jsonify
from flask_ngrok import run_with_ngrok
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json

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
    emotions = []

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_roi = gray_frame[y:y + h, x:x + w]
        roi = cv2.resize(face_roi, (48, 48))
        emotion_pred = EMOTIONS_LIST[np.argmax(model.predict(roi[np.newaxis, :, :, np.newaxis]))]

        cv2.putText(frame, emotion_pred, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        emotions.append(emotion_pred)

    return frame, emotions

# Video stream function for real-time detection
def video_stream_camera():
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_with_emotion, emotions = detect_emotion(frame, model)

        frame_with_emotion = cv2.cvtColor(frame_with_emotion, cv2.COLOR_BGR2RGB)
        _, jpeg = cv2.imencode('.jpg', frame_with_emotion)

        # Send both frame and emotions as JSON
        yield jsonify(frame=jpeg.tobytes(), emotions=emotions)

        if stop_detection:
            break  # Exit the loop if stop_detection is True

    cap.release()

# Route for providing video feed with emotions
@app.route('/video_feed_with_emotion')
def video_feed_with_emotion():
    return Response(video_stream_camera(), mimetype='application/json')

# Other routes and functions remain unchanged

if __name__ == '__main__':
    app.run()
