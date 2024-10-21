from keras.models import Sequential
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array, array_to_img
import numpy as np
import cv2
import sys
sys.stdout.reconfigure(encoding='utf-8')


# SRCNN Model Definition
def build_model():
    SRCNN = Sequential()
    
    # First Conv Layer
    SRCNN.add(Conv2D(filters=128, kernel_size=(9, 9), kernel_initializer='glorot_uniform',
                     activation='relu', padding='valid', use_bias=True, input_shape=(None, None, 1)))
    
    # Second Conv Layer
    SRCNN.add(Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer='glorot_uniform',
                     activation='relu', padding='same', use_bias=True))
    
    # Third Conv Layer
    SRCNN.add(Conv2D(filters=1, kernel_size=(5, 5), kernel_initializer='glorot_uniform',
                     activation='linear', padding='valid', use_bias=True))
    
    # Compile Model
    adam = Adam(learning_rate=0.0001)
    SRCNN.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    
    # Load pretrained weights
    SRCNN.load_weights('Weights_for_SRCNN.h5')
    
    return SRCNN

# Preprocess input image
def preprocess_image(image):
    # If the image has an extra batch dimension (1, 224, 224, 3), remove it
    if len(image.shape) == 4 and image.shape[0] == 1:
        image = np.squeeze(image, axis=0)  # Remove the batch dimension
    
    # Ensure the image is in uint8 format before color conversion
    if image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)  # Convert float64 to uint8
    
    # If the image has 3 channels (RGB), convert to grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image  # If already grayscale, no need to convert
    
    # Convert image to array and normalize
    image = img_to_array(gray_image).astype(np.float32) / 255.0
    
    # Ensure the image is (height, width, 1) before adding batch dimension
    if len(image.shape) == 2:  # If it's a 2D grayscale image
        image = np.expand_dims(image, axis=-1)  # Add channel dimension
    
    # Expand to add batch dimension (for model input)
    img = np.expand_dims(image, axis=0)  # Now shape is (1, height, width, 1)
    
    return img



# Post-process output image
def postprocess_image(pred):
    pred = np.squeeze(pred)
    pred = np.clip(pred * 255.0, 0, 255).astype(np.uint8)
    
    return pred

# Enhance image using the SRCNN model
def enhance_images(image, model=build_model()):
    processed_img = preprocess_image(image)
    pred = model.predict(processed_img)
    enhanced_img = postprocess_image(pred)
    
    return enhanced_img

# Main logic
if __name__ == "__main__":
    # Load the SRCNN model
    srcnn_model = build_model()
    
    # Load an image (replace with your image loading logic)
    input_image = cv2.imread('path_to_image')  # Ensure the image path is correct
    
    # Enhance the image
    enhanced_image = enhance_images(input_image, srcnn_model)
    
    # Show the enhanced image
    cv2.imshow('Enhanced Image', enhanced_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
