# Deep Learning - CNN on fake image dataset
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Step 1: Generate fake image dataset
num_samples = 1000
img_height, img_width = 28, 28
num_classes = 10

# Random grayscale images
X = np.random.rand(num_samples, img_height, img_width, 1).astype('float32')
y = np.random.randint(0, num_classes, num_samples)
y = to_categorical(y, num_classes)

# Step 2: Train-test split
split = int(0.8 * num_samples)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Step 3: CNN Model
model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(2,2),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Step 4: Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 5: Train
model.fit(X_train, y_train, epochs=3, batch_size=32, validation_data=(X_test, y_test))

# Step 6: Evaluate
loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", acc)
