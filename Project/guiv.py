import tkinter as tk
from tkinter import filedialog
from tkinter import *
import threading
from sklearn import metrics
from tensorflow.keras.models import model_from_json
from PIL import Image, ImageTk
import numpy as np
import cv2

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

def video_stream(source):
    global video_emotions
    cap = cv2.VideoCapture(source)  # Use 0 for the default camera or file path for uploaded video
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_emotion_and_update(frame, model)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        frame = ImageTk.PhotoImage(frame)

        sign_image.configure(image=frame)
        sign_image.image = frame
        label1.configure(text='')
        top.update()

        if stop_detection:
            break  # Exit the loop if stop_detection is True

    cap.release()

    # Calculate the overall emotion of the video
    overall_emotion = max(set(video_emotions), key=video_emotions.count)
    print("Overall Emotion of the Video:", overall_emotion)
    video_emotions = []  # Reset the list for the next video

def open_file_dialog():
    reset_stop_detection()  # Reset stop_detection before starting the new video stream
    file_path = filedialog.askopenfilename(filetypes=[("Video files", ".mp4;.avi;*.mkv")])
    if file_path:
        video_thread = threading.Thread(target=lambda: video_stream(file_path))
        video_thread.start()

def video_stream_camera():
    reset_stop_detection()  # Reset stop_detection before starting the new video stream
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_emotion(frame, model)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        frame = ImageTk.PhotoImage(frame)

        sign_image.configure(image=frame)
        sign_image.image = frame
        label1.configure(text='')
        top.update()

        if stop_detection:
            break  # Exit the loop if stop_detection is True

    cap.release()


top = tk.Tk()
top.geometry('800x600')
top.title('Emotion Detector')
top.configure(background='#CDCDCD')

real_time_button = Button(top, text="Real-Time Detection", command=lambda: threading.Thread(target=video_stream_camera).start())
real_time_button.pack()

stop_button = Button(top, text="Stop Detection", command=stop_detection_callback)
stop_button.pack()

upload_button = Button(top, text="Upload Video", command=open_file_dialog)
upload_button.pack()

label1 = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressionModel("model_a1K.json", "model_weights1K.h5")

EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Create a thread for video streaming
video_thread = threading.Thread(target=video_stream)
video_thread.start()

heading = Label(top, text='Emotion Detector', pady=20, font=('arial', 25, 'bold'))
heading.configure(background='#CDCDCD', foreground="#364156")
heading.pack()

sign_image.pack(side='bottom', expand='True')
label1.pack(side='bottom', expand='True')

top.mainloop()