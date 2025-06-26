import os
import cv2
import numpy as np
import joblib

im_size = 64
data = "dataset"

def load_images_from_folder(folder_path, label):
    images = []
    labels = []
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            continue  
        
        img_resized = cv2.resize(img, (im_size, im_size))
        
        img_flattened = img_resized.flatten()
        
        images.append(img_flattened)
        labels.append(label)
    
    return images, labels

cat_images, cat_labels = load_images_from_folder("dataset/cats", 0)
dog_images, dog_labels = load_images_from_folder("dataset/dogs", 1)

X = np.array(cat_images + dog_images)
y = np.array(cat_labels + dog_labels)

joblib.dump(X, "X_data.pkl")
joblib.dump(y, "y_labels.pkl")

print("Preprocessing complete! X and y saved.")


