import os
import cv2
import numpy as np

DATASET_DIR = "dataset/"
IMG_SIZE = 28  # Resize images to 28x28

X, y = [], []

# Read images from dataset folders
for digit in range(10):  
    digit_path = os.path.join(DATASET_DIR, str(digit))
    
    for img_name in os.listdir(digit_path):
        img_path = os.path.join(digit_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize to 28x28
        X.append(img)
        y.append(digit)

# Convert to NumPy arrays
X = np.array(X) / 255.0  # Normalize pixel values
y = np.array(y)

# Save processed data
np.save("X.npy", X)
np.save("y.npy", y)

print(f"Processed {len(X)} images successfully!")
