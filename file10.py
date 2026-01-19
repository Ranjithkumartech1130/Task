import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Input

# Create sequential data
X = []
y = []

for i in range(10):
    X.append([i, i+1, i+2])
    y.append(i+3)

X = np.array(X)
y = np.array(y)

# Reshape for RNN
X = X.reshape((X.shape[0], X.shape[1], 1))

# Build RNN model (FIXED)
model = Sequential([
    Input(shape=(3, 1)),
    SimpleRNN(10, activation='tanh'),
    Dense(1)
])

# Compile model
model.compile(optimizer='adam', loss='mse')

# Train model
model.fit(X, y, epochs=200, verbose=0)

# Test prediction
test_input = np.array([[10, 11, 12]]).reshape((1, 3, 1))
prediction = model.predict(test_input)

print("Predicted next value:", prediction[0][0])
