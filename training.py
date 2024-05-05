import cv2
import os
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples = []
    ids = []
    for image_path in image_paths:
        try:
            pil_image = Image.open(image_path).convert('L')
            image_np = np.array(pil_image, 'uint8')
            id = int(os.path.split(image_path)[-1].split(".")[1])
            faces = detector.detectMultiScale(image_np)
            for (x, y, w, h) in faces:
                face_samples.append(image_np[y:y+h, x:x+w])
                ids.append(id)
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
    return face_samples, ids

try:
    faces, ids = get_images_and_labels('TrainingImage')
    recognizer.train(faces, np.array(ids))
    recognizer.save('TrainingImageLabel/trainner.yml')
except Exception as e:
    print(f"Error training recognizer: {str(e)}")