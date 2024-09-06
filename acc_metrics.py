import cv2
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def load_test_images(base_path):
    images = []
    labels = []
    label_dict = {}

    label_num = 0
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if not os.path.isdir(folder_path):
            continue
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            images.append(image)
            labels.append(label_num)
        label_dict[label_num] = folder
        label_num += 1

    return np.array(images), np.array(labels), label_dict

def evaluate_recognizer(test_images, test_labels, recognizer, label_map):
    predicted_labels = []
    for image in test_images:
        label, confidence = recognizer.predict(image)
        predicted_labels.append(label if confidence < 100 else -1)  # Use -1 for unknown faces

    accuracy = accuracy_score(test_labels, predicted_labels)
    precision = precision_score(test_labels, predicted_labels, average='macro', zero_division=0)
    recall = recall_score(test_labels, predicted_labels, average='macro', zero_division=0)
    f1 = f1_score(test_labels, predicted_labels, average='macro', zero_division=0)

    cm = confusion_matrix(test_labels, predicted_labels)
    return accuracy, precision, recall, f1, cm

# Load trained recognizer and label map
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('C:/Users/balaj/OneDrive/Desktop/work/ml_project/trained_face_recognizer.yml')
label_map = {0: 'student1'}  # Update this dictionary based on actual label mappings

# Load test images and labels
test_base_path = 'C:/Users/balaj/OneDrive/Desktop/work/ml_project/test'
test_images, test_labels, test_label_map = load_test_images(test_base_path)

# Evaluate the recognizer
accuracy, precision, recall, f1, cm = evaluate_recognizer(test_images, test_labels, face_recognizer, label_map)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'Confusion Matrix:\n{cm}')
