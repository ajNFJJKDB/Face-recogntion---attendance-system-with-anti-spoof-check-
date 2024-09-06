import cv2
import numpy as np
import pandas as pd
import os
import datetime

# Function to ensure directory existence
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Directory for storing timestamps
base_directory = 'C:/Users/balaj/OneDrive/Desktop/work/ml_project/timestamp'
today_str = datetime.datetime.now().strftime('%Y-%m-%d')
daily_directory = os.path.join(base_directory, today_str)
ensure_dir(daily_directory)  # Ensure today's directory exists

# Initialize the DataFrame to store timestamps
timestamps_df = pd.DataFrame(columns=['student_id', 'timestamp', 'direction'])

# Load trained recognizer and label map
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('C:/Users/balaj/OneDrive/Desktop/work/ml_project/trained_face_recognizer.yml')
label_map = {0: 'student1', 1: 'student2', 2: 'student3', 3: 'student'}  # Update this dictionary based on actual label mappings

def log_timestamp(student_id, direction):
    global timestamps_df
    new_entry = {'student_id': student_id, 'timestamp': pd.Timestamp('now'), 'direction': direction}
    timestamps_df = timestamps_df.append(new_entry, ignore_index=True)

def load_classifiers():
    classifiers = {
        'frontal': cv2.CascadeClassifier('C:/Users/balaj/OneDrive/Desktop/work/ml_project/haarcascade_frontalface_default.xml'),
        'profile': cv2.CascadeClassifier('C:/Users/balaj/OneDrive/Desktop/work/ml_project/haarcascade_profileface.xml'),
        'fullbody': cv2.CascadeClassifier('C:/Users/balaj/OneDrive/Desktop/work/ml_project/haarcascade_fullbody.xml'),
        'upperbody': cv2.CascadeClassifier('C:/Users/balaj/OneDrive/Desktop/work/ml_project/haarcascade_upperbody.xml'),
        'lowerbody': cv2.CascadeClassifier('C:/Users/balaj/OneDrive/Desktop/work/ml_project/haarcascade_lowerbody.xml')
    }
    return classifiers

def detect_faces(image, classifiers):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = []
    for classifier in classifiers.values():
        detected_faces = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in detected_faces:
            faces.append((x, y, w, h))
    faces = np.array(faces)
    if len(faces) == 0:
        return []
    return non_max_suppression(faces, overlapThresh=0.3)

def non_max_suppression(boxes, overlapThresh):
    if len(boxes) == 0:
        return []
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
    return boxes[pick].astype("int")

# Camera setup
cap = cv2.VideoCapture(0)
classifiers = load_classifiers()

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Detect faces using multiple classifiers
    faces = detect_faces(frame, classifiers)

    # Process each face found
    for (x, y, w, h) in faces:
        # Recognize the face
        face_region = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        label, confidence = face_recognizer.predict(face_region)

        if label in label_map and confidence < 100:  # Confidence threshold for recognized faces
            student_id = label_map[label]
            log_timestamp(student_id, 'IN')
            # Draw a green rectangle for recognized faces
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, student_id, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            # Draw a red rectangle for unknown faces
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('IN Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save timestamps to CSV
timestamps_df.to_csv(os.path.join(daily_directory, 'in_timestamps.csv'), index=False)

