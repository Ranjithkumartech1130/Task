# Deep Learning - CNN Binary Classification (Fake RGB Images)

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# Step 1: Generate fake image dataset
num_samples = 800
img_height, img_width = 64, 64
num_classes = 1   # Binary classification

X = np.random.rand(num_samples, img_height, img_width, 3).astype('float32')
y = np.random.randint(0, 2, num_samples)   # 0 or 1

# Step 2: Train-test split
split = int(0.75 * num_samples)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Step 3: CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Step 4: Compile
model.compile(
    optimizer=Adam(),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Step 5: Train
model.fit(
    X_train, y_train,
    epochs=3,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Step 6: Evaluate
loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", acc)


