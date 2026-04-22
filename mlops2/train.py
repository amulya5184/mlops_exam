import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model (Hyperparameter: C value)
model = LogisticRegression(C=0.5, max_iter=200)

# Train model
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy}")

# Create models folder
os.makedirs("models", exist_ok=True)

# Save model
joblib.dump(model, "models/model_v2.pkl")

print("Model saved as model_v1.pkl")