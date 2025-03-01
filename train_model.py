import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load train data
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")

# Reshape for Scikit-Learn (Flatten 28x28 images into 1D array)
X_train = X_train.reshape(len(X_train), -1)

# Train an SVM classifier
model = SVC(kernel="linear")
model.fit(X_train, y_train)

# Save trained model
import joblib
joblib.dump(model, "digit_model.pkl")

print("Model trained and saved successfully!")
