"""
AI Study Recommendation App
Author: Ghala Alhajeri

Educational machine-learning project.
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Load dataset
data = pd.read_csv("student_data.csv")

# Features
X = data[
    [
        "study_hours",
        "sleep_hours",
        "attendance",
        "previous_grade",
    ]
]

# Target
y = data["support_level"]


# Split into training and testing data
x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
)


# Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)


# Train model
model.fit(x_train, y_train)


# Test model
predictions = model.predict(x_test)

accuracy = accuracy_score(
    y_test,
    predictions,
)

print("Model Accuracy:", round(accuracy * 100, 2), "%")


# Ask user for information
print("\nAI Study Recommendation App")

study_hours = float(
    input("How many hours do you study per day? ")
)

sleep_hours = float(
    input("How many hours do you sleep per night? ")
)

attendance = float(
    input("What is your attendance percentage? ")
)

previous_grade = float(
    input("What was your previous grade percentage? ")
)


# Prepare input for model
student_input = pd.DataFrame(
    [
        {
            "study_hours": study_hours,
            "sleep_hours": sleep_hours,
            "attendance": attendance,
            "previous_grade": previous_grade,
        }
    ]
)


# Predict support level
prediction = model.predict(student_input)[0]

print(
    "\nPredicted Study Support Level:",
    prediction.upper()
)


# Personalized recommendations
recommendations = []

if study_hours < 2:
    recommendations.append(
        "Increase your daily study time gradually."
    )

if sleep_hours < 7:
    recommendations.append(
        "Try to get at least 7 to 8 hours of sleep."
    )

if attendance < 80:
    recommendations.append(
        "Improve your attendance so you do not miss important lessons."
    )

if previous_grade < 70:
    recommendations.append(
        "Focus on weaker topics and use more practice questions."
    )

if previous_grade >= 70 and study_hours < 4:
    recommendations.append(
        "Use active recall and spaced repetition to study more efficiently."
    )

if study_hours >= 4:
    recommendations.append(
        "Use short breaks to avoid burnout during longer study sessions."
    )


print("\nPersonalized Recommendations:")

if recommendations:
    for recommendation in recommendations:
        print("-", recommendation)
else:
    print(
        "- Your current habits are strong. Continue reviewing consistently."
    )
