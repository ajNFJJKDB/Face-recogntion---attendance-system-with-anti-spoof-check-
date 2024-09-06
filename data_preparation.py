import cv2
import numpy as np
import os

def adjust_brightness(image, value):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v = np.clip(v, 0, 255)
    final_hsv = cv2.merge((h, s, v))
    image_brightness_adjusted = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return image_brightness_adjusted

def adjust_contrast(image, alpha):
    image_contrast_adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    return image_contrast_adjusted

def apply_gamma_correction(image, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    image_gamma_corrected = cv2.LUT(image, table)
    return image_gamma_corrected

def adjust_color_temperature(image, temp):
    if temp == 'warm':
        image = cv2.addWeighted(image, 1.5, np.zeros(image.shape, image.dtype), 0, -50)
    elif temp == 'cool':
        image = cv2.addWeighted(image, 0.5, np.zeros(image.shape, image.dtype), 0, 50)
    return image

def process_student_images(input_base_dir, output_base_dir, conditions):
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

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

                for i, (brightness, contrast, gamma, temp) in enumerate(conditions):
                    image_brightness_adjusted = adjust_brightness(image, brightness)
                    image_contrast_adjusted = adjust_contrast(image_brightness_adjusted, contrast)
                    image_gamma_corrected = apply_gamma_correction(image_contrast_adjusted, gamma)
                    image_temp_adjusted = adjust_color_temperature(image_gamma_corrected, temp)

                    output_filename = f"{os.path.splitext(filename)[0]}_condition_{i+1}{os.path.splitext(filename)[1]}"
                    output_path = os.path.join(output_dir, output_filename)
                    cv2.imwrite(output_path, image_temp_adjusted)

input_base_directory = "C:/Users/balaj/OneDrive/Desktop/work/ml_project/student_images"
output_base_directory = "C:/Users/balaj/OneDrive/Desktop/work/ml_project/Diversified_student_images"

# Define the three lighting conditions
conditions = [
    (-50, 0.8, 0.5, 'cool'),   # Low Light Condition
    (0, 1.0, 1.0, 'neutral'),  # Normal Light Condition
    (50, 1.2, 1.5, 'warm')     # Bright Light Condition
]

process_student_images(input_base_directory, output_base_directory, conditions)

