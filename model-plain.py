import io
import csv
import math
import os
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from google.colab import drive
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras import layers, optimizers, regularizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

drive.mount('/content/drive', force_remount=True)

file_path = '/content/drive/MyDrive/Tugas Akhir/Dataset/ip102/ip102_c5_v3.zip'

zip_ref = zipfile.ZipFile(file_path, 'r')
zip_ref.extractall('/content/dataset/ip102/ip102_c5_v3')
zip_ref.close()

test_dir = '/content/dataset/ip102/ip102_c5_v3/test'
train_dir = '/content/dataset/ip102/ip102_c5_v3/train'
val_dir = '/content/dataset/ip102/ip102_c5_v3/val'

EPOCHS = 100
BATCH_SIZE = 32
IMG_WIDTH, IMG_HEIGHT = 224, 224
INPUT_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 3)

augmented_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

normal_datagen = ImageDataGenerator(
    rescale=1./255
)

train_data = augmented_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_data = normal_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_data = normal_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)

LEARNING_RATE = 1e-4
OPTIMIZER = optimizers.RMSprop(learning_rate=LEARNING_RATE)

base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=INPUT_SHAPE)

for layer in base_model.layers:
  layer.trainable = False

# model architecture
model = Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(train_data.num_classes, activation='softmax')
])

model.summary()

model.compile(
    optimizer=OPTIMIZER,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# train model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[early_stopping]
)

# show accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.show()

# show loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()

# test model
test_loss, test_accuracy = model.evaluate(test_data, verbose=2)

print('Test accuracy:', test_accuracy)
print('Test Loss:', test_loss)

# show confussion matrix and classification report
y_pred = np.argmax(model.predict(test_data), axis=1)
y_true = test_data.classes

cm = confusion_matrix(y_true, y_pred)
cr = classification_report(y_true, y_pred, target_names=test_data.class_indices.keys())

print(cm)
print(cr)

# show heatmap
plt.figure(figsize=(8, 8))
plt.imshow(cm, cmap=plt.cm.Blues)
plt.colorbar()
plt.title('Confusion matrix')
plt.xlabel('Predicted label')
plt.ylabel('True label')
tick_marks = np.arange(len(val_data.class_indices))
plt.xticks(tick_marks, val_data.class_indices.keys(), rotation=45)
plt.yticks(tick_marks, val_data.class_indices.keys())
plt.tight_layout()
plt.show()
