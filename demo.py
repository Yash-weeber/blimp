import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import math
FOCAL_LENGTH_MM = 3.04
SENSOR_WIDTH_MM = 3.68
FRAME_WIDTH_PIXELS = 640
FOCAL_LENGTH_PIXELS = (FOCAL_LENGTH_MM / SENSOR_WIDTH_MM) * FRAME_WIDTH_PIXELS

KNOWN_WIDTH_CM = 5

BALLOON_COLORS = {
    "purple": ((130, 50, 50), (160, 255, 255)),
    "green": ((40, 50, 50), (80, 255, 255)),
    "copper": ((10, 100, 100), (20, 255, 255)),
}

TRAIN_DIR = "E:/blimptrain/"
VALID_DIR = "E:/blimpval/"
MODEL_PATH = "shiny_balloon_model.h5"

def train_shiny_balloon_model():
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(64, 64),
        batch_size=32,
        class_mode="binary",
    )

    validation_generator = datagen.flow_from_directory(
        VALID_DIR,
        target_size=(64, 64),
        batch_size=32,
        class_mode="binary",
    )
    steps_per_epoch = math.ceil(train_generator.samples / train_generator.batch_size)
    validation_steps = math.ceil(validation_generator.samples / validation_generator.batch_size)
    print("Steps per epoch:", steps_per_epoch)
    print("Validation steps:", validation_steps)
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        epochs=1000,
    )

    model.save(MODEL_PATH)
    print("Model training complete and saved as shiny_balloon_model.h5")

if not os.path.exists(MODEL_PATH):
    print("Training a new model...")
    train_shiny_balloon_model()

try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print("Error loading model: ", e)
    model = None

def calculate_distance(known_width, focal_length, perceived_width):
    if perceived_width == 0:
        return None
    return (known_width * focal_length) / perceived_width

def detect_balloons(frame, colors):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    detections = []

    for color_name, color_range in colors.items():
        mask = cv2.inRange(hsv, color_range[0], color_range[1])
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 50:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                if 0.8 <= aspect_ratio <= 1.2:
                    distance = calculate_distance(KNOWN_WIDTH_CM, FOCAL_LENGTH_PIXELS, w)
                    detections.append((color_name, x, y, w, h, distance))

    return detections

def classify_shiny(frame, x, y, w, h):
    if model is None:
        return "Unknown"

    roi = frame[y:y+h, x:x+w]
    roi = cv2.resize(roi, (64, 64))
    roi = roi / 255.0
    roi = np.expand_dims(roi, axis=0)

    prediction = model.predict(roi)[0][0]
    return "shiny" if prediction > 0.5 else "not_shiny"

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_balloons(frame, BALLOON_COLORS)

        for color_name, x, y, w, h, distance in detections:
            shiny_status = classify_shiny(frame, x, y, w, h)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{color_name} ({shiny_status}) {distance:.2f}cm", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            print(f"Detected {color_name} balloon ({shiny_status}) at distance {distance:.2f}cm")

        cv2.imshow("Balloon Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
