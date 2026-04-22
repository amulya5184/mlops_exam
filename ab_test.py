import joblib
import random
from sklearn.datasets import load_iris

# Load models
model_v1 = joblib.load("models/model_v1.pkl")
model_v2 = joblib.load("models/model_v2.pkl")

# Load data
data = load_iris()
X = data.data
y = data.target

# Metrics tracking
v1_correct = 0
v2_correct = 0
v1_total = 0
v2_total = 0

# A/B Testing (70% -> v1, 30% -> v2)
for i in range(len(X)):
    if random.random() < 0.7:
        pred = model_v1.predict([X[i]])
        v1_total += 1
        if pred[0] == y[i]:
            v1_correct += 1
    else:
        pred = model_v2.predict([X[i]])
        v2_total += 1
        if pred[0] == y[i]:
            v2_correct += 1

# Results
print("Model V1 Accuracy:", v1_correct / v1_total)
print("Model V2 Accuracy:", v2_correct / v2_total)