import os
import gc
import pickle
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten,
                                     Dense, Dropout, BatchNormalization)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("✅ All libraries imported successfully.")
dataset_path = r"D:\VS Code\Python\plant disease idetification\plantvillage dataset\color"

# Global size and batch constants used in all subsequent steps
IMAGE_SIZE = 32      # 32×32 → streams from disk with minimal RAM
BATCH_SIZE = 32

# Training generator: rescale + augmentation
datagen_train = ImageDataGenerator(
    rescale           = 1./255,
    validation_split  = 0.2,       # 80/20 split handled inside the generator
    rotation_range    = 20,
    width_shift_range = 0.1,
    height_shift_range= 0.1,
    horizontal_flip   = True,
    zoom_range        = 0.1,
    fill_mode         = 'nearest'
)

# Validation generator: rescale only (no augmentation on val data)
datagen_val = ImageDataGenerator(
    rescale          = 1./255,
    validation_split = 0.2
)

print("✅ Augmentation generators configured.")
print("   Training  : rescale + rotation + shift + flip + zoom")
print("   Validation: rescale only")