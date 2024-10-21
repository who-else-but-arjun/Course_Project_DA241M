# Face Recognition Attendance System  

This project implements a face recognition-based attendance system using several machine learning models. The entire pipeline leverages **SRCNN**, **DeblurGANv2**, and **LIME** models for image processing, along with a custom-trained **VGGFace model** for facial recognition. Below is a breakdown of the files, usage, and instructions for setting up the project.

## Project Structure  

### 1. **SRCNN.py**  
- **Purpose:** Defines the SRCNN model used for image super-resolution.  
- **Details:** Loads the pre-trained weights (`generator.h5`) to enhance image quality.  

### 2. **DeblurGANv2.py**  
- **Purpose:** Implements the DeblurGANv2 model to remove blur from images.  
- **Details:** Loads the necessary pre-trained weights for deblurring.  

### 3. **layer_utils.py**  
- **Purpose:** Contains essential utility functions and building blocks.  
- **Details:** Includes implementations for instance normalization and custom layer definitions required for the models.  

### 4. **LIME.py**  
- **Purpose:** Implements the LIME model for light enhancement to improve lighting conditions in images.  

### 5. **pipeline.py**  
- **Purpose:** Cascades the **SRCNN**, **DeblurGANv2**, and **LIME** models to form the complete image processing pipeline.  

### 6. **crop.py**  
- **Purpose:** Uses OpenCV’s Haar Cascade classifier to crop faces from images.  
- **Usage:** Processes raw images from the **Dataset** folder, extracts faces, and saves them to the **Headsets** folder.

### 7. **test.py**  
- **Purpose:** Script to test the functionality of the trained VGGFace model.  

### 8. **VGGface_VGG16.ipynb**  
- **Purpose:** Jupyter notebook used for training the **VGGFace** model and creating the dataset.  
- **Dataset Structure:**  
  - Place images in the format:  
    ```
    Dataset/{student_name}/images  
    ```
  - The `crop.py` extracts faces from these images and stores them in:  
    ```
    Headsets/{student_name}/images  
    ```

### 9. **Attendance.py**  
- **Purpose:** Main script for recognizing faces using the trained **VGGFace model** and logging attendance.  
- **Details:** Uses the webcam to capture real-time images and recognizes students based on the trained VGGFace model.

### 10. **Class.json**  
- **Purpose:** Holds the details of each student in the class as a JSON object, which is referenced during attendance logging.

---

## Prerequisites  

- **Python Version:** 3.11.9  
- **TensorFlow Version:** 2.17  
- Ensure the following folders exist and contain the appropriate files:
  - **Dataset/{student_name}/images** (Raw Images)
  - **Headsets/{student_name}/images** (Extracted Faces)  

---

## Setup Instructions  

1. **Install Dependencies:**  
   Install the required Python libraries using:
   ```bash
   pip install tensorflow opencv-python-headless numpy matplotlib
   ```

2. **Download Weights:**  
   Download the pre-trained weights for all models from the following link:  
   [Google Drive - Weights](https://drive.google.com/drive/u/1/folders/1fHatTQSRryGusJ4VUg5E3R599cmTEtlQ)  
   Place the downloaded weights in the same directory as the code files.

3. **Train the VGGFace Model:**  
   Open `VGGface_VGG16.ipynb` and run the notebook to train the VGGFace model on the dataset.  
   Ensure the dataset follows the structure described above.

4. **Extract Faces for Training:**  
   Run the `crop.py` script to extract faces from the dataset:
   ```bash
   python crop.py
   ```

5. **Run the Attendance System:**  
   Use `Attendance.py` to launch the attendance logging system:
   ```bash
   python Attendance.py
   ```

---

## Usage  

1. **Testing the VGGFace Model:**  
   Use `test.py` to verify the VGGFace model's performance:
   ```bash
   python test.py
   ```

2. **Image Enhancement Pipeline:**  
   Run the `pipeline.py` to process an image through the SRCNN, DeblurGANv2, and LIME models:
   ```bash
   python pipeline.py
   ```

---

## Folder Structure  

```
/project-root  
│  
├── Dataset/  
│   └── {student_name}/images/ (Raw student images)  
├── Headsets/  
│   └── {student_name}/images/ (Cropped faces)  
├── SRCNN.py  
├── DeblurGANv2.py  
├── layer_utils.py  
├── LIME.py  
├── pipeline.py  
├── crop.py  
├── test.py  
├── VGGface_VGG16.ipynb  
├── Attendance.py  
└── Class.json  
```

---

## Notes  
- Ensure all the pre-trained weights are in the correct folder to avoid loading errors.  
- Verify that your webcam is properly connected and recognized by the system before running `Attendance.py`.  
- Adjust the model parameters if needed during training in `VGGface_VGG16.ipynb`.

---
