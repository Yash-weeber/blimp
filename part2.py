import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import math
import time

# ===============================
# Camera and Balloon Parameters
# ===============================
FOCAL_LENGTH_MM = 3.04  # in mm
SENSOR_WIDTH_MM = 3.68  # in mm
FRAME_WIDTH_PIXELS = 640
FRAME_HEIGHT_PIXELS = 480
FOCAL_LENGTH_PIXELS = (FOCAL_LENGTH_MM / SENSOR_WIDTH_MM) * FRAME_WIDTH_PIXELS

TARGET_BALLOON_DIAMETER_CM = 12
TOLERANCE = 0.1  # 10% tolerance

TRAIN_DIR = "E:/blimptrain/"
VALID_DIR = "E:/blimpval/"
MODEL_PATH = "balloon_detection_modelpart2.h5"


# ===============================
# Model Training Function
# ===============================
def train_balloon_model():
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(64, 64),
        batch_size=4,
        class_mode="categorical",
    )

    validation_generator = datagen.flow_from_directory(
        VALID_DIR,
        target_size=(64, 64),
        batch_size=4,
        class_mode="categorical",
    )

    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(4, activation="softmax"),  # 4 neurons for 4 classes
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        epochs=1000,
    )

    model.save(MODEL_PATH)
    print("Model training complete and saved as", MODEL_PATH)


if not os.path.exists(MODEL_PATH):
    print("No model found. Training a new model...")
    train_balloon_model()

try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print("Error loading model:", e)
    model = None


# ===============================
# Utility Functions
# ===============================

def detect_balloons(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply GaussianBlur to reduce noise and improve detection accuracy
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Apply Hough Circle Transform
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=frame.shape[0] / 8,
        param1=100,
        param2=30,
        minRadius=10,
        maxRadius=100
    )

    detections = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            x, y, radius = circle
            # Extract the region of interest (ROI) around the detected circle
            x1, y1 = x - radius, y - radius
            x2, y2 = x + radius, y + radius
            roi = frame[y1:y2, x1:x2]

            # Convert ROI to HSV color space for color detection
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            for color_name, (lower_hsv, upper_hsv) in BALLOON_COLORS.items():
                mask = cv2.inRange(hsv_roi, lower_hsv, upper_hsv)
                if cv2.countNonZero(mask) > 0:
                    # Calculate distance
                    distance = calculate_distance(KNOWN_WIDTH_CM, FOCAL_LENGTH_PIXELS, 2 * radius)
                    detections.append((color_name, x, y, 2 * radius, 2 * radius, distance))
                    break  # Stop checking other colors if a match is found
    return detections
def calculate_distance(known_width, focal_length, perceived_width):
    if perceived_width == 0:
        return None
    return (known_width * focal_length) / perceived_width

def classify_balloon(frame, x, y, w, h):
    """Crop the detected region, preprocess, and classify using the trained model."""
    if model is None:
        return "Unknown"

    # Ensure ROI coordinates are within the frame boundaries
    if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
        print("Invalid ROI coordinates. Skipping classification.")
        return "Not Balloon"

    # Extract the region of interest (ROI)
    roi = frame[y:y+h, x:x+w]

    # Check if ROI is empty
    if roi.size == 0:
        print("Empty ROI detected. Skipping classification.")
        return "Not Balloon"

    # Resize and preprocess the ROI
    try:
        roi = cv2.resize(roi, (64, 64))
    except cv2.error as e:
        print(f"Error resizing ROI: {e}")
        return "Not Balloon"

    roi = roi / 255.0
    roi = np.expand_dims(roi, axis=0)

    # Predict using the trained model
    prediction = model.predict(roi)[0][0]
    return "Balloon" if prediction > 0.5 else "Not Balloon"

# def classify_balloon(frame, x, y, w, h):
#     if model is None:
#         return "Unknown"
#
#     roi = frame[y:y + h, x:x + w]
#     roi = cv2.resize(roi, (64, 64))
#     roi = roi / 255.0
#     roi = np.expand_dims(roi, axis=0)
#
#     prediction = model.predict(roi)[0][0]
#     return "Balloon" if prediction > 0.5 else "Not Balloon"


def compute_3d_position(circle, focal_length_pixels, target_diameter_cm):
    x_center, y_center, radius = circle
    perceived_diameter_pixels = 2 * radius

    distance = (target_diameter_cm * focal_length_pixels) / perceived_diameter_pixels

    cx, cy = FRAME_WIDTH_PIXELS / 2, FRAME_HEIGHT_PIXELS / 2
    x_offset = x_center - cx
    y_offset = y_center - cy

    x_real = (x_offset * distance) / focal_length_pixels
    y_real = (y_offset * distance) / focal_length_pixels
    z_real = distance

    yaw = np.degrees(math.atan(x_offset / focal_length_pixels))
    pitch = np.degrees(math.atan(y_offset / focal_length_pixels))
    roll = 0

    if x_center < cx and y_center < cy:
        quadrant = "Top-Left"
    elif x_center >= cx and y_center < cy:
        quadrant = "Top-Right"
    elif x_center < cx and y_center >= cy:
        quadrant = "Bottom-Left"
    else:
        quadrant = "Bottom-Right"

    return (x_real, y_real, z_real, pitch, roll, yaw, quadrant)


# ===============================
# Main Detection Function using OpenCV VideoCapture
# ===============================
def main():
    cap = cv2.VideoCapture(1)  # Use the first camera device (default webcam)

    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Unable to read frame from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                                   param1=50, param2=30, minRadius=10, maxRadius=150)

        cv2.line(frame, (FRAME_WIDTH_PIXELS // 2, 0), (FRAME_WIDTH_PIXELS // 2, FRAME_HEIGHT_PIXELS), (255, 255, 255),
                 1)
        cv2.line(frame, (0, FRAME_HEIGHT_PIXELS // 2), (FRAME_WIDTH_PIXELS, FRAME_HEIGHT_PIXELS // 2), (255, 255, 255),
                 1)

        if circles is not None:
            circles = np.round(circles[0]).astype("int")
            for circle in circles:
                x_center, y_center, radius = circle

                expected_pixel_diameter = (TARGET_BALLOON_DIAMETER_CM * FOCAL_LENGTH_PIXELS) / calculate_distance(
                    TARGET_BALLOON_DIAMETER_CM,
                    FOCAL_LENGTH_PIXELS,
                    radius * 2)
                if abs(radius * 2 - expected_pixel_diameter) / expected_pixel_diameter > TOLERANCE:
                    continue

                x = x_center - radius
                y = y_center - radius
                w = h = radius * 2

                classification = classify_balloon(frame, x, y, w, h)
                if classification != "Balloon":
                    continue

                x_real, y_real, z_real, pitch_, roll_, yaw_, quadrant_ = compute_3d_position(circle,
                                                                                             FOCAL_LENGTH_PIXELS,
                                                                                             TARGET_BALLOON_DIAMETER_CM)

                # Draw the circle's circumference and its center point
                cv2.circle(frame, (x_center, y_center), radius, (0, 255, 0), 2)
                cv2.circle(frame, (x_center, y_center), 2, (0, 0, 255), 3)

                # Create overlay text with position, orientation, and quadrant info
                text = (f"Pos: ({x_real:.1f},{y_real:.1f},{z_real:.1f})cm, "
                        f"Ang: (Pitch:{pitch_:.1f},Yaw:{yaw_:.1f},Roll:{roll_:.1f}), "
                        f"Quad: {quadrant_}")
                cv2.putText(frame, text, (x_center - radius, y_center - radius - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                print(text)

        # Display the annotated frame
        cv2.imshow("Balloon Detection", frame)
        key = cv2.waitKey(1) & 0xFF

        # Exit on 'q' key press
        if key == ord("q"):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
