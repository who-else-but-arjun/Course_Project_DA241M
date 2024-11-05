import cv2
import numpy as np
import json
from keras.models import load_model
from datetime import datetime
import pandas as pd
import streamlit as st
from pipeline import main
model = load_model('transfer_learning_trained_face_cnn_model.h5')
with open('Class.json', 'r') as f:
    class_details = json.load(f)

st.set_page_config(page_title="Real-Time Attendance System", layout="centered")
st.title("📝 Real-Time Attendance Dashboard")
st.write("This system recognizes students in real-time and marks attendance.")

attendance_log = pd.DataFrame(columns=["Name", "Roll No", "Branch", "Timestamp"])
attendance_display = st.empty() 
face_display = st.empty()  
recognized_students = set() 

def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = image / 255.0
    #sfinal = main(image)
    return np.expand_dims(image, axis=0)

def capture_attendance():
    global attendance_log
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to access webcam.")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') \
            .detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                processed_face = preprocess_image(face)
                prediction = model.predict(processed_face, verbose=0)
                class_id = np.argmax(prediction)
                details = class_details.get(str(class_id), {"name": "Unknown", "roll": "Unknown", "branch": "Unknown"})

                name = details["name"]
                roll = details["roll"]
                branch = details["branch"]
                if roll not in recognized_students:
                    recognized_students.add(roll)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    new_entry = pd.DataFrame([[name, roll, branch, timestamp]], columns=attendance_log.columns)
                    attendance_log = pd.concat([attendance_log, new_entry], ignore_index=True)
                    attendance_display.dataframe(attendance_log)
                    with open('attendance_log.txt', 'a') as log_file:
                        log_file.write(f"Recognized {name}, Roll: {roll}, Branch: {branch} at {timestamp}\n")
                    face_display.image(cv2.cvtColor(face, cv2.COLOR_BGR2RGB), caption=name, width=150)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cap.release()
if st.button("Start Attendance Capture"):
    st.write("Starting real-time attendance capture...")
    capture_attendance()
else:
    st.write("Press the button above to start capturing attendance.")
st.markdown(
    """
    <style>
    .stDataFrame {font-size: 18px; font-weight: bold;}
    </style>
    """,
    unsafe_allow_html=True
)
