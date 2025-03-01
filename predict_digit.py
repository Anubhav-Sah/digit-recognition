import cv2
import numpy as np
import joblib

IMG_SIZE = 28  # Image size (28x28)
model = joblib.load("digit_model.pkl")  # Load trained model

def predict_digit(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize to 28x28
    img = img / 255.0  # Normalize pixel values
    img = img.reshape(1, -1)  # Flatten

    prediction = model.predict(img)
    return prediction[0]

# Test with a new image
image_path = "2.jpg"  # Provide a handwritten digit image
predicted_digit = predict_digit(image_path)
print(f"Predicted Digit: {predicted_digit}")
