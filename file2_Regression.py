# Unsupervised Learning - KMeans (Student Marks)

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Step 1: Fake student marks generate
np.random.seed(0)

maths = np.random.randint(35, 100, 120)
science = np.random.randint(30, 100, 120)

data = pd.DataFrame({
    'Maths Marks': maths,
    'Science Marks': science
})

# Step 2: Feature selection
X = data[['Maths Marks', 'Science Marks']]

# Step 3: KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=0)
data['Cluster'] = kmeans.fit_predict(X)

# Step 4: Visualization
plt.figure(figsize=(8,5))
plt.scatter(
    data['Maths Marks'],
    data['Science Marks'],
    c=data['Cluster']
)
plt.xlabel("Maths Marks")
plt.ylabel("Science Marks")
plt.title("Student Performance Clustering")
plt.show()

