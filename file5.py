# Supervised ML - Linear Regression
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Step 1: Generate fake dataset
np.random.seed(42)
n_samples = 100

size = np.random.randint(500, 3500, n_samples)       # sqft
bedrooms = np.random.randint(1, 6, n_samples)        # 1-5 bedrooms
# Price = 50 + 0.1*size + 10*bedrooms + noise
price = 50 + 0.1*size + 10*bedrooms + np.random.normal(0, 25, n_samples)

# Create DataFrame
data = pd.DataFrame({
    'Size': size,
    'Bedrooms': bedrooms,
    'Price': price
})

# Step 2: Features & target
X = data[['Size', 'Bedrooms']]
y = data['Price']

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predict & Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print("Predicted Prices:", y_pred[:5])
print("Actual Prices   :", y_test.values[:5])
print("Mean Squared Error:", mse)
