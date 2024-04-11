from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
from collections import Counter
from tensorflow.keras.models import model_from_json

app = Flask(__name__)

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the facial expression model
def load_facial_expression_model(json_file, weights_file):
    with open(json_file, "r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = load_facial_expression_model("model_a1K.json", "model_weights1K.h5")
EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Route to render the HTML page
@app.route('/')
def home():
    return render_template('index.html')

# Endpoint for image emotion detection
@app.route('/detect_image_emotion', methods=['POST'])
def detect_image_emotion():
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})

        image = cv2.imread(file)
        gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_image,1.3,5)

        emotions=[]
        for (x,y,w,h) in faces:
            fc = gray_image[y:y+h,x:x+w]
            roi = cv2.resize(fc,(48,48))
            pred = EMOTIONS_LIST[np.argmax(model.predict(roi[np.newaxis,:,:,np.newaxis]))]
            emotions.append(pred)

# Endpoint for video emotion detection
@app.route('/detect_video_emotion', methods=['POST'])
def detect_video_emotion():
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})

        nparr = np.fromstring(file.read(), np.uint8)
        video_path = 'uploads/' + file.filename
        with open(video_path, 'wb') as f:
            f.write(nparr)

        cap = cv2.VideoCapture(video_path)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        emotions = []
        max_frames = int(frame_rate * 30)  # Process a maximum of 30 seconds
        frames_processed = 0

        while frames_processed < max_frames:
            ret, frame = cap.read()

            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

            frame_emotions = []
            for (x, y, w, h) in faces:
                roi = cv2.resize(gray_frame[y:y + h, x:x + w], (48, 48))
                pred = EMOTIONS_LIST[np.argmax(model.predict(roi[np.newaxis, :, :, np.newaxis]))]
                frame_emotions.append(pred)

            emotions.extend(frame_emotions)

            frames_processed += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

        if emotions:
                most_common_emotion = Counter(emotions).most_common(1)[0][0]
                return jsonify({'emotion': most_common_emotion})
        else:
            return jsonify({'error': 'No faces detected or no emotion predictions.'})

if __name__ == '__main__':
    app.run(debug=True)
