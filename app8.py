import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import model_from_json
from PIL import Image
from io import BytesIO

# Define the list of emotions
EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Load Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to load the facial expression model
@st.cache(allow_output_mutation=True)
def load_model(json_file, weights_file):
    with open(json_file, "r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)
    model.load_weights(weights_file)
    return model

# Function to detect emotion in the given frame
def detect_emotion(frame, model):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_roi = gray_frame[y:y + h, x:x + w]
        roi = cv2.resize(face_roi, (48, 48))
        emotion_pred = EMOTIONS_LIST[np.argmax(model.predict(roi[np.newaxis, :, :, np.newaxis]))]

        cv2.putText(frame, emotion_pred, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame

# Main function for Streamlit app
def main():
    st.title("Emotion Detection")

    stop_real_time_detection = st.empty()
    start_real_time_detection = st.empty()

    real_time_detection_started = False
    stop_detection = False

    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mkv"])

    if uploaded_file:
        st.write("Uploaded video:")
        st.video(uploaded_file)

        if st.button("Start Detection"):
            st.write("Starting emotion detection on uploaded video...")
            video_bytes = uploaded_file.read()
            cap = cv2.VideoCapture(BytesIO(video_bytes))

            model = load_model("model_a1K.json", "model_weights1K.h5")

            while cap.isOpened() and not stop_detection:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = detect_emotion(frame, model)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(Image.fromarray(frame_rgb), caption='Uploaded Video Emotion Detection', use_column_width=True)

                stop_detection = stop_real_time_detection.button("Stop Detection" + str(hash(uploaded_file)))  # Use a unique key for each stop button

            cap.release()

    if start_real_time_detection.button("Real-Time Detection"):
        st.write("Starting real-time emotion detection...")
        cap = cv2.VideoCapture(0)

        model = load_model("model_a1K.json", "model_weights1K.h5")

        while cap.isOpened() and not stop_detection:
            ret, frame = cap.read()
            if not ret:
                break

            frame = detect_emotion(frame, model)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(Image.fromarray(frame_rgb), caption='Real-time Emotion Detection', use_column_width=True)

            stop_detection = stop_real_time_detection.button("Stop Detection" + str(hash(0)))  # Use a unique key for each stop button

        cap.release()

if __name__ == "__main__":
    main()
