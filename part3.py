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

# Target balloon physical diameter (in cm) and tolerance
TARGET_BALLOON_DIAMETER_CM = 23
TOLERANCE = 0.1  # 10% tolerance

# ===============================
# Training Data Directories & Model Path
# ===============================
TRAIN_DIR = "E:/blimptrain/"
VALID_DIR = "E:/blimpval/"
MODEL_PATH = "balloon_detection_3model.h5"

# ===============================
# Only these 3 colors should be considered valid balloons
# Adjust HSV ranges as needed
# ===============================
BALLOON_COLORS = {
    "purple": ((130, 50, 50), (160, 255, 255)),
    "green": ((40, 50, 50), (80, 255, 255)),
    "copper": ((10, 100, 100), (20, 255, 255)),
}

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

    num_classes = len(train_generator.class_indices)
    # Ensure the "Balloon" class is index 0 if that matters for your classification logic

    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),  # Categorical output
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    steps_per_epoch = max(1, train_generator.samples // train_generator.batch_size)
    validation_steps = max(1, validation_generator.samples // validation_generator.batch_size)

    model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        epochs=1000,
    )

    model.save(MODEL_PATH)
    print("Model training complete and saved as", MODEL_PATH)

# If no trained model exists, train a new one.
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
def calculate_distance(known_width, focal_length, perceived_width):
    """Compute distance from camera using the pinhole camera model."""
    if perceived_width == 0:
        return None
    return (known_width * focal_length) / perceived_width

def classify_balloon(frame, x, y, w, h):
    """
    Crop the detected region, preprocess, and classify using the trained model.
    Assumes that the model is trained in categorical mode and that index 0 corresponds to "Balloon".
    """
    if model is None:
        return "Unknown"

    # Ensure ROI coordinates are within frame boundaries
    if x < 0 or y < 0 or (x + w) > frame.shape[1] or (y + h) > frame.shape[0]:
        print("Invalid ROI coordinates. Skipping classification.")
        return "Not Balloon"

    roi = frame[int(y):int(y+h), int(x):int(x+w)]
    if roi.size == 0:
        print("Empty ROI detected. Skipping classification.")
        return "Not Balloon"

    try:
        roi = cv2.resize(roi, (64, 64))
    except cv2.error as e:
        print(f"Error resizing ROI: {e}")
        return "Not Balloon"

    roi = roi / 255.0
    roi = np.expand_dims(roi, axis=0)

    prediction = model.predict(roi)[0]
    class_index = np.argmax(prediction)
    # Change the following mapping if necessary
    return "Balloon" if class_index == 0 else "Not Balloon"

def is_valid_balloon_color(frame, circle_center, radius):
    """
    Check if the balloon's circular region in 'frame' matches
    one of the valid balloon colors in BALLOON_COLORS.
    """
    x_center, y_center = circle_center

    # Define a bounding box around the circle
    x1 = int(x_center - radius)
    y1 = int(y_center - radius)
    x2 = int(x_center + radius)
    y2 = int(y_center + radius)

    # Ensure bounding box is within the frame
    if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
        return False

    roi = frame[y1:y2, x1:x2]
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # We'll consider a balloon valid if it matches at least one color range
    for color_name, (lower, upper) in BALLOON_COLORS.items():
        mask = cv2.inRange(roi_hsv, lower, upper)
        # If enough pixels match this color, we consider it valid
        if np.count_nonzero(mask) > 200:  # Adjust threshold as needed
            return True
    return False

def compute_3d_position(circle, focal_length_pixels, target_diameter_cm):
    """
    Compute approximate 3D position (x, y, z) and orientation (pitch, roll, yaw)
    based on a detected circle defined by (x_center, y_center, radius).
    Also determines the quadrant of the frame in which the circle center lies.
    """
    x_center, y_center, radius = circle
    perceived_diameter_pixels = 2 * radius

    # Distance (z) from the camera
    distance = (target_diameter_cm * focal_length_pixels) / perceived_diameter_pixels

    # Offsets from the image center
    cx, cy = FRAME_WIDTH_PIXELS / 2, FRAME_HEIGHT_PIXELS / 2
    x_offset = x_center - cx
    y_offset = y_center - cy

    # Real-world x and y positions (in cm)
    x_real = (x_offset * distance) / focal_length_pixels
    y_real = (y_offset * distance) / focal_length_pixels
    z_real = distance

    # Compute approximate orientation angles (assuming roll = 0)
    yaw = np.degrees(math.atan2(x_offset, focal_length_pixels))
    pitch = np.degrees(math.atan2(y_offset, focal_length_pixels))
    roll = 0

    # Determine quadrant based on center of frame
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
    cap = cv2.VideoCapture(1)  # Use your default webcam (change index if needed)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH_PIXELS)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT_PIXELS)

    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    # We’ll use a time-based filter to reduce spammy output
    last_print_time = 0.0
    MIN_PRINT_INTERVAL = 0.5  # seconds

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from camera.")
            break

        # Convert frame to grayscale and apply blur for circle detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # Detect circles using Hough Circle Transform
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                                   param1=50, param2=30, minRadius=10, maxRadius=150)

        # Draw quadrant divider lines
        cv2.line(frame, (FRAME_WIDTH_PIXELS // 2, 0), (FRAME_WIDTH_PIXELS // 2, FRAME_HEIGHT_PIXELS), (255, 255, 255), 1)
        cv2.line(frame, (0, FRAME_HEIGHT_PIXELS // 2), (FRAME_WIDTH_PIXELS, FRAME_HEIGHT_PIXELS // 2), (255, 255, 255), 1)

        if circles is not None:
            circles = np.round(circles[0]).astype("int")
            for circle in circles:
                x_center, y_center, radius = circle

                # 1) Check color match first
                if not is_valid_balloon_color(frame, (x_center, y_center), radius):
                    continue

                # 2) Compute distance from the circle's pixel diameter
                measured_diameter = 2 * radius
                distance = calculate_distance(TARGET_BALLOON_DIAMETER_CM, FOCAL_LENGTH_PIXELS, measured_diameter)
                # Expected pixel diameter if the balloon is exactly TARGET_BALLOON_DIAMETER_CM at computed distance
                expected_pixel_diameter = ((TARGET_BALLOON_DIAMETER_CM * FOCAL_LENGTH_PIXELS) / distance
                                           if distance else 0)

                # 3) Validate detection based on size tolerance
                if expected_pixel_diameter == 0 or abs(measured_diameter - expected_pixel_diameter) / expected_pixel_diameter > TOLERANCE:
                    continue

                # 4) Classify the balloon
                x = x_center - radius
                y = y_center - radius
                w = h = measured_diameter
                classification = classify_balloon(frame, x, y, w, h)
                if classification != "Balloon":
                    continue

                # 5) Compute 3D position and orientation
                x_real, y_real, z_real, pitch, roll, yaw, quadrant = compute_3d_position(
                    (x_center, y_center, radius),
                    FOCAL_LENGTH_PIXELS,
                    TARGET_BALLOON_DIAMETER_CM
                )

                # 6) Draw the circle's circumference and center point
                cv2.circle(frame, (x_center, y_center), radius, (0, 255, 0), 2)
                cv2.circle(frame, (x_center, y_center), 2, (0, 0, 255), 3)

                # 7) Only print once per interval to reduce spam
                current_time = time.time()
                if (current_time - last_print_time) >= MIN_PRINT_INTERVAL:
                    overlay_text = (f"x:{x_real:.1f}cm y:{y_real:.1f}cm z:{z_real:.1f}cm "
                                    f"Pitch:{pitch:.1f}° Roll:{roll:.1f}° Yaw:{yaw:.1f}° "
                                    f"Quad:{quadrant}")
                    print(overlay_text)
                    last_print_time = current_time

                # Optionally overlay text on the frame (if you want to see it on-screen)
                cv2.putText(frame, f"{classification}", (x_center - radius, y_center - radius - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Balloon Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
