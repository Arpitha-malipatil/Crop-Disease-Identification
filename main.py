import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("Libraries loaded successfully")

dataset_path = os.path.join(os.getcwd(), "PlantVillage")

IMAGE_SIZE = 32
BATCH_SIZE = 32
datagen_train = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

datagen_val = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen_train.flow_from_directory(
    dataset_path,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=42
)

val_gen = datagen_val.flow_from_directory(
    dataset_path,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
    seed=42
)

print("Training images:", train_gen.n)
print("Validation images:", val_gen.n)
print("Classes:", train_gen.class_indices)