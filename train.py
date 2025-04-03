import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import joblib

# Load data
df = pd.read_csv("data/iris.csv")
X = df.drop("target", axis=1)
y = df["target"]

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict and score
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Save metrics
with open("metrics.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.2f}\n")

# Save plot
plt.bar(["Accuracy"], [accuracy])
plt.ylim(0, 1)
plt.savefig("plots/accuracy.png")

# Save model
joblib.dump(model, "model.pkl")
