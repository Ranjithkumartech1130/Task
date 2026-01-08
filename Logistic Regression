# Step 1: Import libraries
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 2: Create dataset
X = np.array([[1], [2], [3], [4], [5], [6]])   # Hours studied
y = np.array([0, 0, 0, 1, 1, 1])               # Pass(1) / Fail(0)

# Step 3: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Create model
model = LogisticRegression()

# Step 5: Train model
model.fit(X_train, y_train)

# Step 6: Predict
y_pred = model.predict(X_test)

# Step 7: Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Predicted Values:", y_pred)
print("Actual Values   :", y_test)
print("Accuracy        :", accuracy)
