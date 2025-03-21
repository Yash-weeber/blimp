import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import math
import time
import requests

# ==========================================
# Global Parameters & Camera Settings
# ==========================================
FOCAL_LENGTH_MM = 3.04  # in mm
SENSOR_WIDTH_MM = 3.68  # in mm
FRAME_WIDTH_PIXELS = 640
FRAME_HEIGHT_PIXELS = 480
FOCAL_LENGTH_PIXELS = (FOCAL_LENGTH_MM / SENSOR_WIDTH_MM) * FRAME_WIDTH_PIXELS

TARGET_BALLOON_DIAMETER_CM = 23
TOLERANCE = 0.1  # 10% tolerance

IMG_SIZE = (64, 64)

# ==========================================
# Dataset Directories & Model File
# ==========================================
TRAIN_DIR = "E:/blimptrain/"  # Folder with subfolders "shiny" and "not shiny"
VALID_DIR = "E:/blimpval/"    # Folder with subfolders "shiny" and "not shiny"
MODEL_PATH = "balloon_detection_model.h5"

# ==========================================
# Color Threshold Parameters (modifiable)
# ==========================================
# Global mode and color selection
COLOR_MODE = "HSV"  # Choose between "HSV" and "RGB"
SELECTED_COLOR = "red"  # Change to 'red', 'blue', 'purple', or 'green'

# For RGB mode (example values, adjust as needed)
RGB_RANGES = {
    "red": (np.array([100, 0, 0]), np.array([255, 100, 100])),
    "blue": (np.array([0, 0, 100]), np.array([100, 100, 255])),
    "purple": (np.array([100, 0, 100]), np.array([255, 100, 255])),
    "green": (np.array([0, 100, 0]), np.array([100, 255, 100]))
}

# ==========================================
# Home Station URL (for sending image & data)
# ==========================================
HOME_STATION_URL = "http://your-home-station-address/api/upload"  # Adjust accordingly


# ==========================================
# Training: Build & Train the Classifier Model
# ==========================================
def train_balloon_model():
    datagen = ImageDataGenerator(rescale=1.0 / 255)
    train_generator = datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=32,
        class_mode="categorical"
    )
    validation_generator = datagen.flow_from_directory(
        VALID_DIR,
        target_size=IMG_SIZE,
        batch_size=32,
        class_mode="categorical"
    )
    num_classes = len(train_generator.class_indices)
    print("Detected classes:", train_generator.class_indices)

    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        epochs=20  # Adjust epochs as needed
    )

    model.save(MODEL_PATH)
    print("Model training complete and saved as", MODEL_PATH)
    return model


# ==========================================
# Detection: Color Mask and Bounding Box Extraction
# ==========================================
def get_color_bbox(frame):
    if COLOR_MODE == "HSV":
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if SELECTED_COLOR.lower() == "red":
            # Red requires two ranges since hue wraps around.
            lower_red1 = np.array([0, 120, 70])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 120, 70])
            upper_red2 = np.array([180, 255, 255])
            mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)
        elif SELECTED_COLOR.lower() == "blue":
            lower_blue = np.array([94, 80, 2])
            upper_blue = np.array([126, 255, 255])
            mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)
        elif SELECTED_COLOR.lower() == "purple":
            lower_purple = np.array([129, 50, 70])
            upper_purple = np.array([158, 255, 255])
            mask = cv2.inRange(hsv_frame, lower_purple, upper_purple)
        elif SELECTED_COLOR.lower() == "green":
            lower_green = np.array([40, 40, 40])
            upper_green = np.array([70, 255, 255])
            mask = cv2.inRange(hsv_frame, lower_green, upper_green)
        else:
            print("Invalid SELECTED_COLOR specified.")
            return None
    elif COLOR_MODE == "RGB":
        if SELECTED_COLOR.lower() in RGB_RANGES:
            lower, upper = RGB_RANGES[SELECTED_COLOR.lower()]
            mask = cv2.inRange(frame, lower, upper)
        else:
            print("Invalid SELECTED_COLOR specified for RGB mode.")
            return None
    else:
        print("Invalid COLOR_MODE specified.")
        return None

    # Clean up the mask using morphological operations.
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    bbox = cv2.boundingRect(largest_contour)
    return bbox


# ==========================================
# Detection: Compute 3D Position & Quadrant
# ==========================================
def compute_3d_position(bbox):
    x, y, w, h = bbox
    if w == 0:
        return None
    distance = (TARGET_BALLOON_DIAMETER_CM * FOCAL_LENGTH_PIXELS) / w
    x_center = x + w / 2
    y_center = y + h / 2
    cx = FRAME_WIDTH_PIXELS / 2
    cy = FRAME_HEIGHT_PIXELS / 2
    x_offset = x_center - cx
    y_offset = y_center - cy
    x_cm = (x_offset * distance) / FOCAL_LENGTH_PIXELS
    y_cm = (y_offset * distance) / FOCAL_LENGTH_PIXELS
    if x_center < cx and y_center < cy:
        quadrant = "Top-Left"
    elif x_center >= cx and y_center < cy:
        quadrant = "Top-Right"
    elif x_center < cx and y_center >= cy:
        quadrant = "Bottom-Left"
    else:
        quadrant = "Bottom-Right"
    return (x_cm, y_cm, distance, quadrant, (x_center, y_center))


# ==========================================
# Sending Data to Home Station
# ==========================================
def send_to_home_station(image_path, x_cm, y_cm, distance, quadrant):
    data = {
        "x_cm": f"{x_cm:.1f}",
        "y_cm": f"{y_cm:.1f}",
        "distance_cm": f"{distance:.1f}",
        "quadrant": quadrant
    }
    try:
        with open(image_path, "rb") as img_file:
            files = {"image": img_file}
            response = requests.post(HOME_STATION_URL, data=data, files=files)
            print("Image sent successfully. Response:", response.text)
    except Exception as e:
        print("Failed to send image:", e)


# ==========================================
# Continuous Detection Routine
# ==========================================
def run_detection(model):
    cap = cv2.VideoCapture(1)  # Adjust your camera index if needed
    if not cap.isOpened():
        print("Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        display_frame = cv2.resize(frame, (FRAME_WIDTH_PIXELS, FRAME_HEIGHT_PIXELS))
        bbox = get_color_bbox(display_frame)
        if bbox is not None:
            x, y, w, h = bbox
            roi = display_frame[y:y+h, x:x+w]
            if roi.size > 0:
                roi_resized = cv2.resize(roi, IMG_SIZE)
                roi_norm = roi_resized.astype("float32") / 255.0
                roi_norm = np.expand_dims(roi_norm, axis=0)
                preds = model.predict(roi_norm)[0]
                predicted_class = np.argmax(preds)
                cv2.putText(display_frame, f"Class: {predicted_class} ({preds[predicted_class]:.2f})", (x, y-30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                if predicted_class == 1:
                    pos_info = compute_3d_position(bbox)
                    if pos_info is not None:
                        x_cm, y_cm, distance, quadrant, detected_center = pos_info
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.circle(display_frame, (int(detected_center[0]), int(detected_center[1])), 5, (0, 0, 255), -1)
                        center_frame = (FRAME_WIDTH_PIXELS // 2, FRAME_HEIGHT_PIXELS // 2)
                        cv2.circle(display_frame, center_frame, 5, (255, 0, 0), -1)
                        cv2.line(display_frame, center_frame, (int(detected_center[0]), int(detected_center[1])), (0, 255, 255), 2)
                        cv2.line(display_frame, (FRAME_WIDTH_PIXELS // 2, 0), (FRAME_WIDTH_PIXELS, FRAME_HEIGHT_PIXELS), (255, 255, 255), 1)
                        cv2.line(display_frame, (0, FRAME_HEIGHT_PIXELS // 2), (FRAME_WIDTH_PIXELS, FRAME_HEIGHT_PIXELS // 2), (255, 255, 255), 1)
                        info_text = f"Pos: ({x_cm:.1f}cm, {y_cm:.1f}cm, {distance:.1f}cm) {quadrant}"
                        cv2.putText(display_frame, info_text, (x, y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(display_frame, "Not a shiny balloon", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(display_frame, "No region with specified color detected", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Balloon Detection", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            output_image_path = "detected_balloon.jpg"
            cv2.imwrite(output_image_path, display_frame)
            print("Output saved to", output_image_path)
            # Uncomment to send data when quitting:
            # send_to_home_station(output_image_path, x_cm, y_cm, distance, quadrant)
            break

    cap.release()
    cv2.destroyAllWindows()


# ==========================================
# Main Routine: Train (if needed) & Run Detection
# ==========================================
def main():
    if os.path.exists(MODEL_PATH):
        print("Loading trained model...")
        model = load_model(MODEL_PATH)
    else:
        print("No trained model found. Starting training...")
        model = train_balloon_model()

    time.sleep(0.5)
    run_detection(model)


if __name__ == "__main__":
    main()
