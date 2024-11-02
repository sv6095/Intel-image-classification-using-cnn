import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import Rescaling
import matplotlib.pyplot as plt

# directory paths
train_dir = 'iimcd\seg_train\seg_train'
test_dir = 'iimcd\seg_test\seg_test'

# Image dimensions
img_height, img_width = 150, 150

# Data Generators
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(img_height, img_width), batch_size=32, class_mode='categorical', subset='training')
validation_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(img_height, img_width), batch_size=32, class_mode='categorical', subset='validation')

# CNN Model
model = Sequential([
    Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(),
    Conv2D(32, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(train_generator.class_indices), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, validation_data=validation_generator, epochs=15)

# Save the model
model.save('iimc.h5')

# Plot accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
