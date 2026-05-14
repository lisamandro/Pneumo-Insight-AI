from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

# Load model
model = load_model("trained.h5", compile=False)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Test data preprocessing
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    "archive/chest_xray/test",
    target_size=(300,300),   # match your app/model size
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Evaluate model
loss, accuracy = model.evaluate(test_generator)

print("\nAccuracy:", round(accuracy*100,2), "%")

# Predictions
predictions = model.predict(test_generator)

predictions = (predictions > 0.5).astype(int)

print("\nClassification Report:\n")

print(
    classification_report(
        test_generator.classes,
        predictions
    )
)

print("\nConfusion Matrix:\n")

print(
    confusion_matrix(
        test_generator.classes,
        predictions
    )
)