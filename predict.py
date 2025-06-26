import cv2
import joblib
import numpy as np


im_size=64

img=cv2.imread("dog.jpeg",cv2.IMREAD_GRAYSCALE)

if img is None:
    print(" Image cannot be loaded")
    exit()
else:
    print("Image loaded successfully")

im_resized=cv2.resize(img,(im_size,im_size))

flattened=im_resized.flatten()

model = joblib.load("svm_cat_dog_model.pkl")

prediction = model.predict([flattened])

if prediction==0:
    print("It is a cat")

else:
    print("It is a dog")