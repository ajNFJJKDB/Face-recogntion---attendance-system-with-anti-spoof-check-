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
label_map = {0: 'student1'}  # Update this dictionary based on actual label mappings

def log_timestamp(student_id, direction):
    global timestamps_df
    new_entry = {'student_id': student_id, 'timestamp': pd.Timestamp('now'), 'direction': direction}
    timestamps_df = timestamps_df.append(new_entry, ignore_index=True)

# Camera setup
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('C:/Users/balaj/OneDrive/Desktop/work/ml_project/haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Convert BGR to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_frame, 1.1, 4)

    # Process each face found
    for (x, y, w, h) in faces:
        # Recognize the face
        face_region = gray_frame[y:y+h, x:x+w]
        label, confidence = face_recognizer.predict(face_region)

        if label in label_map and confidence < 100:  # Confidence threshold for recognized faces
            student_id = label_map[label]
            log_timestamp(student_id, 'OUT')
            # Draw a green rectangle for recognized faces
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, student_id, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            # Draw a red rectangle for unknown faces
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('OUT Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

timestamps_df.to_csv(os.path.join(daily_directory, 'out_timestamps.csv'), index=False)

