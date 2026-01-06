# Supervised ML - Linear Regression (Study Hours vs Exam Score)

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Step 1: Fake dataset
np.random.seed(1)
n_samples = 80

study_hours = np.random.randint(1, 10, n_samples)   # hours per day
# Score = 30 + 7*hours + noise
exam_score = 30 + 7*study_hours + np.random.normal(0, 5, n_samples)

data = pd.DataFrame({
    'Study_Hours': study_hours,
    'Exam_Score': exam_score
})

# Step 2: Features & target
X = data[['Study_Hours']]
y = data['Exam_Score']

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Step 4: Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predict & Evaluate
y_pred = model.predict(X_test)

print("Predicted Scores:", y_pred[:5])
print("Actual Scores   :", y_test.values[:5])
print("MSE:", mean_squared_error(y_test, y_pred))
