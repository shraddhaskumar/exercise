import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("exercise_angles.csv")

# Feature columns (all angles)
feature_cols = [
    "Shoulder_Angle", "Elbow_Angle", "Hip_Angle", "Knee_Angle", "Ankle_Angle",
    "Shoulder_Ground_Angle", "Elbow_Ground_Angle", "Hip_Ground_Angle",
    "Knee_Ground_Angle", "Ankle_Ground_Angle"
]

X = df[feature_cols]

# Target label
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["Label"])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
clf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
clf.fit(X_train, y_train)

# Accuracy check
accuracy = clf.score(X_test, y_test)
print("Model accuracy:", accuracy)

# Save the model and label encoder
joblib.dump(clf, "exercise_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("Model saved as exercise_model.pkl")
print("Encoder saved as label_encoder.pkl")
