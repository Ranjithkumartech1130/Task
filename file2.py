# Unsupervised Learning - KMeans Clustering
# CSV illa run panna mudiyum

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Step 1: Fake dataset generate pannuvom
# 200 customers, random Annual Income & Spending Score

np.random.seed(42)
annual_income = np.random.randint(15, 140, 200)   # k$
spending_score = np.random.randint(1, 100, 200)   # score 1-100

# Create DataFrame
data = pd.DataFrame({
    'Annual Income (k$)': annual_income,
    'Spending Score (1-100)': spending_score
})

# Step 2: Features select pannuvom
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Step 3: KMeans model create pannuvom
kmeans = KMeans(n_clusters=5, random_state=42)

# Step 4: Fit model & predict clusters
data['Cluster'] = kmeans.fit_predict(X)

# Step 5: Visualization
plt.figure(figsize=(8,5))
plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'],
            c=data['Cluster'], cmap='viridis')
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation (Fake Dataset)")
plt.show()
