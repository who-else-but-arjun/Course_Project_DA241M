import cv2
import numpy as np
import tensorflow as tf
import json
from matplotlib import pyplot as plt
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.models import load_model 

model = load_model('transfer_learning_trained_face_cnn_model.h5')
facecascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

with open('Class.json', 'r') as f:
    class_info = json.load(f)

test_image = '1.jpg'
imgtest = cv2.imread(test_image, cv2.IMREAD_COLOR)
image_array = np.array(imgtest)
faces = facecascade.detectMultiScale(imgtest, scaleFactor=1.1, minNeighbors=5)

for (x_, y_, w, h) in faces:
            face_detect = cv2.rectangle(imgtest, (x_, y_), (x_+w, y_+h), (255, 0, 255), 2)
            plt.imshow(cv2.cvtColor(face_detect, cv2.COLOR_BGR2RGB))
            plt.show()
            roi = image_array[y_: y_ + h, x_: x_ + w]
            resized_image = cv2.resize(roi, (224,224))

            x = img_to_array(resized_image)
            x = np.expand_dims(x, axis=0)
            #x = tf.keras.applications.vgg16.preprocess_input(x) 
            predicted_prob = model.predict(x, verbose = 0 )
            predicted_class = predicted_prob[0].argmax()

            if str(predicted_class) in class_info:
                class_details = class_info[str(predicted_class)]
                print(f"Predicted face: {class_details['name']}")
                print(f"Roll No: {class_details['roll']}")
                print(f"Branch: {class_details['branch']}")
            else:
                print(f"Class ID {predicted_class} not found in class.json")
