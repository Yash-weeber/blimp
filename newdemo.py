import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
# from picamera import PiCamera
# from picamera.array import PiRGBArray
import math
import time

# ===============================
# Camera and Balloon Parameters
# ===============================
FOCAL_LENGTH_MM = 3.04           # in mm
SENSOR_WIDTH_MM = 3.68           # in mm
FRAME_WIDTH_PIXELS = 640
FRAME_HEIGHT_PIXELS = 480
FOCAL_LENGTH_PIXELS = (FOCAL_LENGTH_MM / SENSOR_WIDTH_MM) * FRAME_WIDTH_PIXELS

# Target balloon physical diameter (cm) and tolerance (e.g., 10% tolerance)
TARGET_BALLOON_DIAMETER_CM = 23
TOLERANCE = 0.1  # 10% tolerance

# ===============================
# Training Data Directories & Model Path
# ===============================
TRAIN_DIR = "E:/blimptrain/"
VALID_DIR = "E:/blimpval/"
MODEL_PATH = "balloon_detection_modelpart2.h5"

# ===============================
# Model Training Function
# ===============================
def train_balloon_model():
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    
    # train_generator = datagen.flow_from_directory(
    #     TRAIN_DIR,
    #     target_size=(64, 64),
    #     batch_size=4,
    #     class_mode="binary",
    # )
    #
    # validation_generator = datagen.flow_from_directory(
    #     VALID_DIR,
    #     target_size=(64, 64),
    #     batch_size=32,
    #     class_mode="binary",
    # )
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

    # model = Sequential([
    #     Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)),
    #     MaxPooling2D(2, 2),
    #     Conv2D(64, (3, 3), activation="relu"),
    #     MaxPooling2D(2, 2),
    #     Flatten(),
    #     Dense(128, activation="relu"),
    #     Dropout(0.5),
    #     Dense(1, activation="sigmoid"),
    # ])
    #
    # model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
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

# If no trained model exists, train a new one.
if not os.path.exists(MODEL_PATH):
    print("No model found. Training a new model...")
    train_balloon_model()

# Try to load the model.
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print("Error loading model:", e)
    model = None

# ===============================
# Utility Functions
# ===============================
def calculate_distance(known_width, focal_length, perceived_width):
    """Compute distance from camera using pinhole camera model."""
    if perceived_width == 0:
        return None
    return (known_width * focal_length) / perceived_width

def classify_balloon(frame, x, y, w, h):
    """Crop the detected region, preprocess, and classify using the trained model."""
    if model is None:
        return "Unknown"
    
    roi = frame[y:y+h, x:x+w]
    roi = cv2.resize(roi, (64, 64))
    roi = roi / 255.0
    roi = np.expand_dims(roi, axis=0)
    
    prediction = model.predict(roi)[0][0]
    return "Balloon" if prediction > 0.5 else "Not Balloon"

def compute_3d_position(circle, focal_length_pixels, target_diameter_cm):
    """
    Compute approximate 3D position (x, y, z) and orientation (pitch, roll, yaw)
    based on a detected circle defined by (x_center, y_center, radius).
    """
    x_center, y_center, radius = circle
    perceived_diameter_pixels = 2 * radius

    # Distance (z) using pinhole camera model:
    distance = (target_diameter_cm * focal_length_pixels) / perceived_diameter_pixels

    # Offsets from image center:
    cx, cy = FRAME_WIDTH_PIXELS / 2, FRAME_HEIGHT_PIXELS / 2
    x_offset = x_center - cx
    y_offset = y_center - cy

    # Real-world x and y positions (in cm)
    x_real = (x_offset * distance) / focal_length_pixels
    y_real = (y_offset * distance) / focal_length_pixels
    z_real = distance

    # Orientation: approximate pitch and yaw angles (roll assumed 0 for a circle)
    yaw = np.degrees(math.atan(x_offset / focal_length_pixels))
    pitch = np.degrees(math.atan(y_offset / focal_length_pixels))
    roll = 0

    # Determine the quadrant
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
# Main Detection Function using PiCamera and Hough Circles
# ===============================
def main():
    camera = PiCamera()
    camera.resolution = (FRAME_WIDTH_PIXELS, FRAME_HEIGHT_PIXELS)
    rawCapture = PiRGBArray(camera, size=(FRAME_WIDTH_PIXELS, FRAME_HEIGHT_PIXELS))

    # Allow camera to warm up
    time.sleep(0.1)
    
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        image = frame.array
        
        # Preprocessing for circle detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Use Hough Circle Transform to detect circles
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                                   param1=50, param2=30, minRadius=10, maxRadius=150)
        
        # Optionally draw quadrant divider lines for visual reference
        cv2.line(image, (FRAME_WIDTH_PIXELS // 2, 0), (FRAME_WIDTH_PIXELS // 2, FRAME_HEIGHT_PIXELS), (255, 255, 255), 1)
        cv2.line(image, (0, FRAME_HEIGHT_PIXELS // 2), (FRAME_WIDTH_PIXELS, FRAME_HEIGHT_PIXELS // 2), (255, 255, 255), 1)
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for circle in circles:
                x_center, y_center, radius = circle
                
                # Validate detected circle using the expected size
                expected_pixel_diameter = (TARGET_BALLOON_DIAMETER_CM * FOCAL_LENGTH_PIXELS) / calculate_distance(TARGET_BALLOON_DIAMETER_CM, FOCAL_LENGTH_PIXELS, 2 * radius)
                if abs(2 * radius - expected_pixel_diameter) / expected_pixel_diameter > TOLERANCE:
                    continue  # Skip circles not matching the target balloon size
                
                # Use a bounding rectangle around the circle for classification
                x = x_center - radius
                y = y_center - radius
                w = h = 2 * radius
                classification = classify_balloon(image, x, y, w, h)
                if classification != "Balloon":
                    continue  # Only process if the classifier confirms a balloon
                
                # Compute 3D position and orientation from the detected circle
                x_real, y_real, z_real, pitch, roll, yaw, quadrant = compute_3d_position(circle, FOCAL_LENGTH_PIXELS, TARGET_BALLOON_DIAMETER_CM)
                
                # Draw the circle's circumference and its center point
                cv2.circle(image, (x_center, y_center), radius, (0, 255, 0), 2)
                cv2.circle(image, (x_center, y_center), 2, (0, 0, 255), 3)
                
                # Create overlay text with position, orientation and quadrant info
                text = (f"Pos: ({x_real:.1f},{y_real:.1f},{z_real:.1f})cm, "
                        f"Ang: (Pitch:{pitch:.1f},Yaw:{yaw:.1f},Roll:{roll:.1f}), "
                        f"Quad: {quadrant}")
                cv2.putText(image, text, (x_center - radius, y_center - radius - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                print(text)
        
        # Display the annotated frame
        cv2.imshow("Balloon Detection", image)
        key = cv2.waitKey(1) & 0xFF
        
        # Clear stream for next frame
        rawCapture.truncate(0)
        if key == ord("q"):
            break
    
    cv2.destroyAllWindows()
    camera.close()

if __name__ == "__main__":
    main()
