# Edge AI - Simple ANN (Sensor data example)

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Fake sensor data (Temperature, Humidity)
X = np.array([
    [30, 60],
    [35, 70],
    [20, 40],
    [25, 50]
])

# Output: 1 = Fan ON, 0 = Fan OFF
y = np.array([1, 1, 0, 0])

# Model
model = Sequential([
    Dense(4, activation='relu', input_shape=(2,)),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(X, y, epochs=50, verbose=0)

# Edge prediction
test = np.array([[33, 65]])
print("Fan ON probability:", model.predict(test))
