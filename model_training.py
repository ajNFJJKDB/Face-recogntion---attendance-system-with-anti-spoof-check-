import cv2
import os
import numpy as np

def load_images_and_labels(base_path):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()  
    labels = []
    faces = []
    label_dict = {}
    
    # Each subdirectory in 'images' corresponds to a person
    label_num = 0
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if not os.path.isdir(folder_path):
            continue
        # Process each image in the directory
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            faces.append(gray)
            labels.append(label_num)
        label_dict[label_num] = folder  # Map label number to person ID/name
        label_num += 1
    
    return face_recognizer, np.array(faces), np.array(labels), label_dict

def train_recognizer(base_path):
    recognizer, faces, labels, label_map = load_images_and_labels(base_path)
    recognizer.train(faces, labels)
    return recognizer, label_map

# Path to the folder containing images
base_path = "C:/Users/balaj/OneDrive/Desktop/work/ml_project/Preprocessed_student_images"
recognizer, label_map = train_recognizer(base_path)
print(label_map)

# Now you can save the trained model
recognizer.save('C:/Users/balaj/OneDrive/Desktop/work/ml_project/trained_face_recognizer.yml')

# label_map contains mapping from label numbers to actual IDs/names
print("Training complete, model saved.")
