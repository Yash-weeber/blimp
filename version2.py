import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, LeakyReLU
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import pandas as pd
import os
import math
import time
import requests

# ==========================================
# Global Parameters & Settings
# ==========================================
# Camera parameters
FOCAL_LENGTH_MM = 3.04  # in mm
SENSOR_WIDTH_MM = 3.68  # in mm
FRAME_WIDTH_PIXELS = 640
FRAME_HEIGHT_PIXELS = 480
FOCAL_LENGTH_PIXELS = (FOCAL_LENGTH_MM / SENSOR_WIDTH_MM) * FRAME_WIDTH_PIXELS

# Balloon physical diameter (cm) and tolerance for validating size
TARGET_BALLOON_DIAMETER_CM = 23
TOLERANCE = 0.1  # 10% tolerance

# Image input size for training and detection (YOLO typically uses 416x416)
IMAGE_WIDTH, IMAGE_HEIGHT = 416, 416

# Paths for dataset and model
DATASET_DIR = "E:/blimptrain/"# Folder containing images
ANNOTATIONS_FILE = "dataset/annotations.csv"  # CSV file with annotations (filename,x,y,w,h; normalized values)
MODEL_PATH = "balloon_yolo_model.h5"

# ==========================================
# Color Threshold Parameters (modifiable)
# ==========================================
# Example for a red balloon in HSV space:
HSV_LOWER = np.array([0, 100, 100])
HSV_UPPER = np.array([10, 255, 255])
# Example for a red balloon in RGB space:
RGB_LOWER = np.array([100, 0, 0])
RGB_UPPER = np.array([255, 100, 100])
# Choose which color space to use: "HSV" or "RGB"
COLOR_MODE = "HSV"

# ==========================================
# Home Station URL (for sending image and data)
# ==========================================
HOME_STATION_URL = "http://your-home-station-address/api/upload"  # modify as needed


# ==========================================
# Utility Functions for 3D computation & quadrant
# ==========================================
def compute_3d_position(bbox):
    """
    Compute the approximate 3D position (in cm) of the balloon using the pinhole camera model.
    bbox: [x, y, w, h] (in pixels). Here we use the width (w) as the perceived diameter.
    Returns: (x_real, y_real, distance, quadrant, detected_center)
    """
    x, y, w, h = bbox
    if w == 0:
        return None
    # Calculate distance using pinhole model: distance (cm) = (actual_diameter * focal_length_pixels) / perceived_diameter_pixels
    distance = (TARGET_BALLOON_DIAMETER_CM * FOCAL_LENGTH_PIXELS) / w

    # Compute the center of the bounding box
    x_center = x + w / 2
    y_center = y + h / 2

    # Offsets from the image center
    cx = FRAME_WIDTH_PIXELS / 2
    cy = FRAME_HEIGHT_PIXELS / 2
    x_offset = x_center - cx
    y_offset = y_center - cy

    # Convert pixel offset to real-world offset in cm
    x_real = (x_offset * distance) / FOCAL_LENGTH_PIXELS
    y_real = (y_offset * distance) / FOCAL_LENGTH_PIXELS

    # Determine quadrant
    if x_center < cx and y_center < cy:
        quadrant = "Top-Left"
    elif x_center >= cx and y_center < cy:
        quadrant = "Top-Right"
    elif x_center < cx and y_center >= cy:
        quadrant = "Bottom-Left"
    else:
        quadrant = "Bottom-Right"

    return (x_real, y_real, distance, quadrant, (x_center, y_center))


def color_filter_mask(frame):
    """
    Returns a binary mask of the frame based on the selected color thresholds.
    Works in HSV or RGB mode.
    """
    if COLOR_MODE == "HSV":
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    elif COLOR_MODE == "RGB":
        mask = cv2.inRange(frame, RGB_LOWER, RGB_UPPER)
    else:
        mask = None
    return mask


# ==========================================
# Data Loading and Preprocessing for Training
# ==========================================
def load_dataset():
    """
    Loads the dataset from DATASET_DIR using the ANNOTATIONS_FILE.
    Assumes the CSV file has columns: filename,x,y,w,h (all normalized values between 0 and 1).
    Returns:
        X: list of images resized to (IMAGE_WIDTH, IMAGE_HEIGHT)
        y: numpy array of shape (num_samples, 5) where each row is [1, x, y, w, h]
           (we set the objectness score to 1 since every sample is a balloon)
    """
    data = pd.read_csv(ANNOTATIONS_FILE)
    X = []
    y = []
    for index, row in data.iterrows():
        img_path = os.path.join(DATASET_DIR, row['filename'])
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        # Resize image to training size
        img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
        # Normalize image pixels to [0,1]
        img = img.astype('float32') / 255.0
        X.append(img)
        # Get bounding box annotation (normalized)
        # Assuming annotations are normalized with respect to width/height.
        # Our output vector is: [objectness, x, y, w, h]
        bbox = [1.0, row['x'], row['y'], row['w'], row['h']]
        y.append(bbox)
    X = np.array(X)
    y = np.array(y)
    return X, y


# ==========================================
# Model Definition (Simplified YOLO-like Model)
# ==========================================
def build_model():
    """
    Builds a simple CNN that regresses a 5-element vector: [objectness, x, y, w, h] (all normalized).
    This is a simplified model for single-object detection.
    """
    inputs = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))

    # Convolutional layers
    x = Conv2D(16, (3, 3), padding="same")(inputs)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(2, 2)(x)

    x = Conv2D(32, (3, 3), padding="same")(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(2, 2)(x)

    x = Conv2D(64, (3, 3), padding="same")(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(2, 2)(x)

    x = Flatten()(x)
    x = Dense(128)(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Output layer: 5 neurons with sigmoid activation (all outputs in [0,1])
    outputs = Dense(5, activation="sigmoid")(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(lr=1e-4), loss="mse")
    return model


# ==========================================
# Model Training
# ==========================================
def train_model():
    print("Loading dataset...")
    X, y = load_dataset()
    print("Dataset loaded. Samples:", len(X))
    model = build_model()
    print("Starting training...")
    model.fit(X, y, epochs=100, batch_size=8, validation_split=0.1)
    model.save(MODEL_PATH)
    print("Training complete. Model saved as", MODEL_PATH)
    return model


# ==========================================
# Detection Function (Single Frame Processing)
# ==========================================
def detect_and_process(model):
    """
    Captures one frame from the camera, uses the model to predict the bounding box,
    applies color filtering to verify the detection, computes 3D position info,
    draws the annotations (bounding box, center line, quadrant text), saves the image,
    and sends the image & data to the home station.
    """
    # Capture one frame from the default camera
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Failed to capture image from camera.")
        return

    # Resize frame to (IMAGE_WIDTH, IMAGE_HEIGHT) for the model
    orig_frame = cv2.resize(frame, (FRAME_WIDTH_PIXELS, FRAME_HEIGHT_PIXELS))
    proc_frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
    proc_frame_norm = proc_frame.astype('float32') / 255.0

    # Expand dimensions and predict using the model
    pred = model.predict(np.expand_dims(proc_frame_norm, axis=0))[0]
    # The prediction is [objectness, x, y, w, h] (all normalized)
    obj_score, x_norm, y_norm, w_norm, h_norm = pred
    # If objectness score is too low, consider no detection
    if obj_score < 0.5:
        print("No object detected with high confidence.")
        return

    # Convert normalized bbox to image coordinates (using original frame dimensions)
    x = int(x_norm * FRAME_WIDTH_PIXELS)
    y = int(y_norm * FRAME_HEIGHT_PIXELS)
    w = int(w_norm * FRAME_WIDTH_PIXELS)
    h = int(h_norm * FRAME_HEIGHT_PIXELS)

    # Ensure bounding box is within frame
    x = max(0, x)
    y = max(0, y)
    if x + w > FRAME_WIDTH_PIXELS:
        w = FRAME_WIDTH_PIXELS - x
    if y + h > FRAME_HEIGHT_PIXELS:
        h = FRAME_HEIGHT_PIXELS - y

    bbox = [x, y, w, h]

    # Apply color filtering on the detected region (ROI)
    roi = orig_frame[y:y + h, x:x + w]
    if roi.size == 0:
        print("ROI is empty. Aborting detection.")
        return
    mask = color_filter_mask(roi)
    color_pixels = cv2.countNonZero(mask)
    total_pixels = roi.shape[0] * roi.shape[1]
    if total_pixels == 0 or (color_pixels / total_pixels) < 0.5:
        print("Detected object does not match the specified color criteria.")
        return

    # Compute 3D position and quadrant information
    pos_data = compute_3d_position(bbox)
    if pos_data is None:
        print("Error computing 3D position.")
        return
    x_real, y_real, z_real, quadrant, detected_center = pos_data

    # Draw a bounding square around the detected balloon
    cv2.rectangle(orig_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Draw the center of the detected object
    cv2.circle(orig_frame, (int(detected_center[0]), int(detected_center[1])), 5, (0, 0, 255), -1)
    # Draw the center of the frame
    center_frame = (FRAME_WIDTH_PIXELS // 2, FRAME_HEIGHT_PIXELS // 2)
    cv2.circle(orig_frame, center_frame, 5, (255, 0, 0), -1)
    # Draw a line from the center of the frame to the detected object
    cv2.line(orig_frame, center_frame, (int(detected_center[0]), int(detected_center[1])), (0, 255, 255), 2)

    # Overlay text with position (in cm) and quadrant info
    text = f"Pos: ({x_real:.1f}cm, {y_real:.1f}cm, {z_real:.1f}cm), Quad: {quadrant}"
    cv2.putText(orig_frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the image once (waits until a key is pressed)
    cv2.imshow("Balloon Detection", orig_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the annotated image
    output_image_path = "detected_balloon.jpg"
    cv2.imwrite(output_image_path, orig_frame)
    print("Output saved to", output_image_path)

    # Prepare data to send
    data = {
        "x_cm": f"{x_real:.1f}",
        "y_cm": f"{y_real:.1f}",
        "z_cm": f"{z_real:.1f}",
        "quadrant": quadrant
    }
    # Send image and coordinate data to the home station
    send_to_home_station(output_image_path, data)


# ==========================================
# Function: Send Image & Data to Home Station
# ==========================================
def send_to_home_station(image_path, data):
    """
    Sends the image and associated coordinate data to the home station via HTTP POST.
    """
    files = {'image': open(image_path, 'rb')}
    try:
        response = requests.post(HOME_STATION_URL, files=files, data=data)
        print("Image sent successfully, response:", response.text)
    except Exception as e:
        print("Failed to send image:", e)


# ==========================================
# Main Routine
# ==========================================
def main():
    # If model doesn't exist, train it first
    if not os.path.exists(MODEL_PATH):
        print("No trained model found. Starting training...")
        model = train_model()
    else:
        print("Loading trained model...")
        model = load_model(MODEL_PATH)

    # Small delay to allow camera to warm up (if needed)
    time.sleep(0.5)

    # Run detection on one captured frame
    detect_and_process(model)


if __name__ == "__main__":
    main()
