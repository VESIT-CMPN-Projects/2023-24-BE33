import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from collections import Counter

def FacialExpressionModel(json_file, weights_file):
    with open(json_file, "r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def Detect(video_source):
    cap = cv2.VideoCapture(video_source)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Process a maximum of 30 seconds (adjust the frame limit accordingly)
    max_frames = int(frame_rate * 30)
    frames_processed = 0
    
    emotions = []  # Store emotions for each frame
    
    while frames_processed < max_frames:
        ret, frame = cap.read()

        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        frame_emotions = []  # Store emotions for the current frame
        
        for (x, y, w, h) in faces:
            roi = cv2.resize(gray_frame[y:y+h, x:x+w], (48, 48))
            pred = EMOTIONS_LIST[np.argmax(model.predict(roi[np.newaxis, :, :, np.newaxis]))]
            
            frame_emotions.append(pred)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, pred, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA)

        emotions.extend(frame_emotions)  # Add emotions for the current frame to the overall list

        # Convert the frame to ImageTk format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img_tk = ImageTk.PhotoImage(image=img)

        # Update the Tkinter window with the new frame
        video_label.img_tk = img_tk
        video_label.configure(image=img_tk)
        video_label.update_idletasks()

        frames_processed += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

    # Analyze emotions and display the most common emotion
    if emotions:
        most_common_emotion = Counter(emotions).most_common(1)[0][0]
        label1.configure(text=f'Most Common Emotion: {most_common_emotion}')
    else:
        label1.configure(text='No faces detected or no emotion predictions.')

def show_Detect_button(video_source):
    detect_b = Button(top, text="Detect Emotion", command=lambda: Detect(video_source), padx=10, pady=5)
    detect_b.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
    detect_b.place(relx=0.79, rely=0.46)

def upload_video():
    try:
        file_path = filedialog.askopenfilename()
        label1.configure(text='')
        show_Detect_button(file_path)
    except Exception as e:
        print(e)

top = tk.Tk()
top.geometry('800x600')
top.title('Video Sentiment Analysis')
top.configure(background='#CDCDCD')

label1 = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
label1.pack(side='bottom', expand='True')

heading = Label(top, text='Video Sentiment Analysis', pady=20, font=('arial', 25, 'bold'))
heading.configure(background='#CDCDCD', foreground="#364156")
heading.pack()

upload = Button(top, text="Upload Video", command=upload_video, padx=10, pady=5)
upload.configure(background="#364156", foreground='white', font=('arial', 20, 'bold'))
upload.pack(side='bottom', pady=50)

model = FacialExpressionModel("model_a1K.json", "model_weights1K.h5")
EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Create a label to display video frames
video_label = Label(top)
video_label.pack()

top.mainloop()