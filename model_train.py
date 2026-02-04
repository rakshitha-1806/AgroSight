import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

# -----------------------------
# Paths and Parameters
# -----------------------------
DATA_DIR = 'PlantVillage'  # Folder with 15 subfolders
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 10

# -----------------------------
# Data Generators
# -----------------------------
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# -----------------------------
# Model Definition
# -----------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_gen.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# -----------------------------
# Train Model
# -----------------------------
model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

# -----------------------------
# Save Model
# -----------------------------
model.save('plant_model.h5')
print("Model saved as plant_model.h5")
