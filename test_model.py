import numpy as np
import joblib
from sklearn.metrics import accuracy_score

# Load test data
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# Load trained model
model = joblib.load("digit_model.pkl")

# Reshape test images for Scikit-Learn
X_test = X_test.reshape(len(X_test), -1)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")
