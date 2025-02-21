import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import math

# from picamera.array import PiRGBArray
# from picamera import PiCamera

FOCAL_LENGTH_MM = 3.04
SENSOR_WIDTH_MM = 3.68
FRAME_WIDTH_PIXELS = 640
FOCAL_LENGTH_PIXELS = (FOCAL_LENGTH_MM / SENSOR_WIDTH_MM) * FRAME_WIDTH_PIXELS

KNOWN_WIDTH_CM = 10  # Adjusted for blimp size

TRAIN_DIR = "E:/blimptrain/"
VALID_DIR = "E:/blimpval/"
MODEL_PATH = "blimp_detection_model.h5"


def train_blimp_model():
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

    # Use math.ceil to ensure at least one step is executed even with a small dataset
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
        epochs=10,
    )

    model.save(MODEL_PATH)
    print("Model training complete and saved as", MODEL_PATH)


if not os.path.exists(MODEL_PATH):
    print("Training a new model...")
    train_blimp_model()

try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print("Error loading model:", e)
    model = None


def calculate_distance(known_width, focal_length, perceived_width):
    if perceived_width == 0:
        return None
    return (known_width * focal_length) / perceived_width


def classify_blimp(frame, x, y, w, h):
    if model is None:
        return "Unknown"

    roi = frame[y:y + h, x:x + w]
    roi = cv2.resize(roi, (64, 64))
    roi = roi / 255.0
    roi = np.expand_dims(roi, axis=0)

    prediction = model.predict(roi)[0][0]
    return "Blimp" if prediction > 0.5 else "Not a Blimp"


def main():
    # If using a Raspberry Pi, uncomment the following lines:
    # camera = PiCamera()
    # camera.resolution = (640, 480)
    # rawCapture = PiRGBArray(camera, size=(640, 480))

    # For testing on a desktop, use cv2.VideoCapture
    cap = cv2.VideoCapture(0)

    while True:
        ret, image = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                if 1.5 <= aspect_ratio <= 4.0:  # Adjust for blimp shape
                    distance = calculate_distance(KNOWN_WIDTH_CM, FOCAL_LENGTH_PIXELS, w)
                    classification = classify_blimp(image, x, y, w, h)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(image, f"{classification} {distance:.2f}cm", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    print(f"Detected {classification} at distance {distance:.2f}cm")

        cv2.imshow("Blimp Detection", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
