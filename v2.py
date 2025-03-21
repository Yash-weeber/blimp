import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import math
import socket
import struct
import time

# ========================================================================
# Camera and Object Parameters
# ========================================================================
FOCAL_LENGTH_MM = 3.04  # in mm
SENSOR_WIDTH_MM = 3.68  # in mm
FRAME_WIDTH_PIXELS = 640
FRAME_HEIGHT_PIXELS = 480
# Compute focal length in pixel units using sensor dimensions
FOCAL_LENGTH_PIXELS = (FOCAL_LENGTH_MM / SENSOR_WIDTH_MM) * FRAME_WIDTH_PIXELS

# Expected physical diameter of the balloon (cm)
TARGET_BALLOON_DIAMETER_CM = 23

# ========================================================================
# Data Directories and Model Path (adapt these paths as needed)
# ========================================================================
TRAIN_DIR = "E:/blimptrain/"
VALID_DIR = "E:/blimpval/"
MODEL_PATH = "balloon_classifier_model.h5"

# ========================================================================
# Color Tolerance Threshold and Color Mode (HSV or RGB)
# ========================================================================
USE_HSV = True  # Change to False if you prefer RGB thresholding
if USE_HSV:
    # Example HSV thresholds – adjust according to your target balloon color
    lower_color = np.array([0, 100, 100])
    upper_color = np.array([10, 255, 255])
else:
    # Example RGB thresholds
    lower_color = np.array([150, 0, 0])
    upper_color = np.array([255, 80, 80])


# ========================================================================
# Model Training Function: Train a simple CNN using images arranged in
# subfolders (e.g., "shiny" and "not_shiny") for training and validation.
# ========================================================================
def train_model():
    print("Training model from images...")
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_gen = datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(64, 64),
        batch_size=4,
        class_mode="categorical"
    )
    val_gen = datagen.flow_from_directory(
        VALID_DIR,
        target_size=(64, 64),
        batch_size=4,
        class_mode="categorical"
    )

    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(2, activation="softmax")  # 2 neurons for two classes: shiny and not_shiny
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // train_gen.batch_size,
        validation_data=val_gen,
        validation_steps=val_gen.samples // val_gen.batch_size,
        epochs=10  # Adjust epochs as required
    )

    model.save(MODEL_PATH)
    print("Model training complete and saved as", MODEL_PATH)
    return model


# ========================================================================
# Load Existing Model or Train a New Model if Not Found
# ========================================================================
if os.path.exists(MODEL_PATH):
    print("Loading existing classification model...")
    model = load_model(MODEL_PATH)
else:
    model = train_model()


# ========================================================================
# Detection Functions
# ========================================================================
def detect_color_square(image, use_hsv=True, lower_thresh=None, upper_thresh=None):
    """
    Process the input image using a color threshold.
    Find all contours that match the threshold, combine them,
    and then compute a square bounding box that encloses all points.
    """
    if use_hsv:
        processed = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    else:
        processed = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = cv2.inRange(processed, lower_thresh, upper_thresh)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # Combine all found contours and get one bounding rectangle
    all_points = np.concatenate(contours)
    x, y, w, h = cv2.boundingRect(all_points)
    # Ensure the box is square by taking the maximum dimension
    side = max(w, h)
    return (x, y, side, side)


def compute_position(square_bbox):
    """
    Compute the object's distance and offset in real-world units (cm)
    using a pinhole camera model based on the side (perceived diameter) of the detected square.
    """
    x, y, w, h = square_bbox
    if w == 0:
        return None
    # Distance (z) calculation from the perceived size (w in pixels)
    distance = (TARGET_BALLOON_DIAMETER_CM * FOCAL_LENGTH_PIXELS) / w
    # Calculate the object's center in the image
    obj_center_x = x + w / 2
    obj_center_y = y + h / 2
    center_img_x = FRAME_WIDTH_PIXELS / 2
    center_img_y = FRAME_HEIGHT_PIXELS / 2
    x_offset = obj_center_x - center_img_x
    y_offset = obj_center_y - center_img_y
    # Convert pixel offsets to cm using similar triangles
    x_cm = (x_offset * distance) / FOCAL_LENGTH_PIXELS
    y_cm = (y_offset * distance) / FOCAL_LENGTH_PIXELS
