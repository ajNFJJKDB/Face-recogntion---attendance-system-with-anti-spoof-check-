import cv2
import os

def detect_and_crop_faces(image, face_cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    return gray[y:y+h, x:x+w]

def resize_and_pad(image, size=(256, 256)):
    h, w = image.shape[:2]
    aspect_ratio = w / h
    desired_w, desired_h = size

    if (w / h) > (desired_w / desired_h):
        new_w = desired_w
        new_h = int(new_w / aspect_ratio)
    else:
        new_h = desired_h
        new_w = int(new_h * aspect_ratio)

    resized_image = cv2.resize(image, (new_w, new_h))

    delta_w = desired_w - new_w
    delta_h = desired_h - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_image

def reduce_noise(image, h=5, templateWindowSize=7, searchWindowSize=21):
    return cv2.fastNlMeansDenoising(image, None, h, templateWindowSize, searchWindowSize)

def adaptive_histogram_equalization(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def preprocess_images(input_base_dir, output_base_dir, face_cascade_path, size=(256, 256), noise_reduction_params=(5, 7, 21)):
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    for student_dir in os.listdir(input_base_dir):
        input_dir = os.path.join(input_base_dir, student_dir)
        output_dir = os.path.join(output_base_dir, student_dir)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for filename in os.listdir(input_dir):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(input_dir, filename)
                image = cv2.imread(image_path)

                if image is None:
                    continue

                face = detect_and_crop_faces(image, face_cascade)
                if face is None:
                    continue

                face_resized = resize_and_pad(face, size)
                face_denoised = reduce_noise(face_resized, *noise_reduction_params)
                face_contrast_enhanced = adaptive_histogram_equalization(face_denoised)

                output_filename = f"{os.path.splitext(filename)[0]}_preprocessed{os.path.splitext(filename)[1]}"
                output_path = os.path.join(output_dir, output_filename)
                cv2.imwrite(output_path, face_contrast_enhanced)

input_base_directory = 'C:/Users/balaj/OneDrive/Desktop/work/ml_project/Diversified_student_images'
output_base_directory = 'C:/Users/balaj/OneDrive/Desktop/work/ml_project/Preprocessed_student_images'
face_cascade_path = 'C:/Users/balaj/OneDrive/Desktop/work/ml_project/haarcascade_frontalface_default.xml'
resize_dimensions = (256, 256)
noise_reduction_params = (5, 7, 21)

preprocess_images(input_base_directory, output_base_directory, face_cascade_path, resize_dimensions, noise_reduction_params)
