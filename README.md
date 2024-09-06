# Face Recognition Based Attendance System

Project report - ML_project.pdf

## Project Description
This project aims to automate the attendance process in educational settings using face recognition technology. By leveraging advanced machine learning techniques and image processing, the system offers a reliable, efficient, and secure method for recording student attendance.

## Key Features
- **Automated Attendance**: Eliminates manual attendance marking, reducing time and minimizing human error.
- **High Accuracy**: Uses machine learning models to prevent proxy attendance and enhance accuracy.
- **Contactless System**: Provides a secure, contactless identification method, ideal for modern classrooms.
- **Real-time Monitoring**: Tracks both entry and exit times, ensuring up-to-date attendance records.
- **CSV Export**: Attendance records are systematically stored and exported in CSV format for administrative use.

## Objectives
1. **Streamline Attendance**: Automate the process to save time and reduce errors.
2. **Enhance Security**: Offer a more secure and reliable attendance system.
3. **Improve Accuracy**: Minimize attendance fraud, such as proxy attendance.

## Components
1. **Monitoring Module**: 
    - Uses face detection and recognition to log students' attendance.
    - Records entry and exit times in real-time.
2. **Management Module**: 
    - Manages collected attendance data and generates CSV reports.

## Process Overview
1. **Data Collection**: 
    - Capture multiple images per student under various lighting conditions to train the model.
2. **Preprocessing**: 
    - Resize, denoise, and apply filters to images for accurate model training.
3. **Model Training**: 
    - Train the model using OpenCV’s LBPH (Local Binary Patterns Histograms) algorithm.
4. **Face Detection & Recognition**:
    - Detect faces using Haar Cascade classifiers and recognize them with the LBPH model.

## Requirements

### Software:
- **Programming Language**: Python
- **Libraries**: 
    - OpenCV
    - NumPy
    - Pandas
- **IDE**: 
    - PyCharm, Jupyter Notebook, or Visual Studio Code
- **Version Control**: Git & GitHub

### Hardware:
- High-resolution camera (infrared capable for low light environments)

## Installation

1. Clone the repository:
    ```
    git clone https://github.com/yourusername/face-recognition-attendance-system.git
    ```

2. Install the required Python packages:
    ```
    pip install -r requirements.txt
    ```

3. Run the data collection script to gather initial student images:
    ```
    python data_collection.py
    ```

4. Train the model:
    ```
    python train_model.py
    ```

5. Start the attendance system:
    ```
    python attendance_system.py
    ```

## Usage
- **Data Collection**: Use this script to capture and preprocess student images for model training.
- **Face Recognition**: The system uses real-time face recognition to track attendance.
- **Attendance Reports**: Attendance data is stored in a CSV file for easy review.

## Future Work
- Explore advanced deep learning models like CNNs, FaceNet, or ResNet for improved accuracy.
- Implement distance-based recognition improvements to increase the system’s reliability at greater distances.
- Integrate transfer learning for enhanced performance on small datasets.

## Results
The system was tested with a sample dataset and achieved an accuracy of 92% under ideal conditions. Recognition performance varies based on distance and lighting, and future improvements will focus on overcoming these limitations.

## Authors
- Balaji G (3122213002015)

References
1. [A Multi-Face Challenging Dataset for Robust Face Recognition](https://ieeexplore.ieee.org/document/8581283)
2. [An Approach for Face Detection and Face Recognition using OpenCV and Face Recognition Libraries in Python](https://ieeexplore.ieee.org/document/10113066)
3. [Real-Time Face Recognition Attendance System Based on Video Processing](https://ieeexplore.ieee.org/document/10393263)

