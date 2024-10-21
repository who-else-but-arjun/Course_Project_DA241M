import cv2
import numpy as np
import json
from keras.models import load_model
from datetime import datetime
import matplotlib.pyplot as plt
from pipeline import main


model = load_model('transfer_learning_trained_face_cnn_model.h5')
cap = cv2.VideoCapture(0)
with open('Class.json', 'r') as f:
    class_details = json.load(f)
    
    
def preprocess_image(image):  
    # Resize the image to the expected model input size
    image = cv2.resize(image, (224, 224))  

    # Convert to RGB if it’s in grayscale or another format
    if len(image.shape) == 2:  # If grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 1:  # If single channel
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Normalize the image to [0, 1]
    image = image / 255.0  

    # Pass the image through the main processing function
    # final = main(image)

    # # Debugging: Check the output of main
    # if final is None:
    #     print("Error: main() returned None.")
    #     return None

    # # Convert PIL image to NumPy array
    # if isinstance(final, np.ndarray):
    #     final = final  # Already a NumPy array
    # else:
    #     final = np.array(final)  # Convert from PIL Image to NumPy array

    # # Ensure the shape is correct: (height, width, channels)
    # if final.ndim == 2:  # If it's grayscale
    #     final = np.expand_dims(final, axis=-1)  # Add channel dimension
    # final = cv2.cvtColor(final, cv2.COLOR_RGB2BGR)  # Convert to BGR format if needed

    # # Resize the final image to the input shape for the model
    # final = cv2.resize(final, (224, 224))  # Resize to model input size

    # # Add batch dimension
    image = np.array(image)
    final = np.expand_dims(image, axis=0)  # Shape becomes (1, 224, 224, 3)
    return final  # Return the processed image


recognized_students = set()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            
            processed_face = preprocess_image(face)

            prediction = model.predict(processed_face, verbose = 0)
            #print(prediction)
            class_id = np.argmax(prediction)

            details = class_details.get(str(class_id), {"name": "Unknown", "roll": "Unknown", "branch": "Unknown"})
            name = details["name"]
            roll = details["roll"]
            branch = details["branch"]

            if roll not in recognized_students:
                recognized_students.add(roll)
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"Recognized {name} at {timestamp}")
                with open('attendence_log.txt', 'a') as log_file:
                    log_file.write(f"Recognized {name}, Roll: {roll}, Branch: {branch} at {timestamp}\n")

                
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
        #if cv2.waitKey(1) & 0xFF == ord('q'):
            #break
cap.release()
