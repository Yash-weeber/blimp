import cv2
import numpy as np
import os
import math
import time
import requests
from ultralytics import YOLO  # pip install ultralytics

# ==========================================
# Global Parameters & Camera Settings
# ==========================================
FOCAL_LENGTH_MM = 3.04  # in mm
SENSOR_WIDTH_MM = 3.68  # in mm
FRAME_WIDTH_PIXELS = 640
FRAME_HEIGHT_PIXELS = 480
FOCAL_LENGTH_PIXELS = (FOCAL_LENGTH_MM / SENSOR_WIDTH_MM) * FRAME_WIDTH_PIXELS

# Known balloon physical diameter (cm) and tolerance
TARGET_BALLOON_DIAMETER_CM = 23
TOLERANCE = 0.1  # 10% tolerance

# ==========================================
# YOLOv9 Model & Dataset Configurations
# ==========================================
DATA_CONFIG = "C:/Users/thepr/PycharmProjects/PythonProject/balloon2.v2i.yolov9/data.yaml"  # YAML file with dataset paths (train/valid/test)
#MODEL_CONFIG = YOLO("C:/Users/thepr/PycharmProjects/PythonProject/balloon2.v2i.yolov9/yolov9.yaml", imgsz =640)  # YOLOv9 model configuration file
MODEL_WEIGHTS = "balloon_yolov9.pt"  # File to save/load trained weights
# Load model with explicit image size
MODEL_CONFIG = YOLO("C:/Users/thepr/PycharmProjects/PythonProject/balloon2.v2i.yolov9/yolov9.yaml")

# Run inference with matching image size
results = MODEL_CONFIG.predict("C:/Users/thepr/PycharmProjects/PythonProject/balloon2.v2i.yolov9/test/images/16_jpg.rf.daf113ebb3f2106fc29e9f7f4efdfe32.jpg", imgsz=[1080, 1920])

# Color Threshold Parameters
# ==========================================
# Choose "HSV" or "RGB" for color filtering
COLOR_MODE = "HSV"
# Select desired color: "red", "blue", "purple", or "green"
SELECTED_COLOR = "red"

# HSV ranges for different colors (adjust as needed)
HSV_RANGES = {
    "red": ([0, 120, 70], [10, 255, 255], [170, 120, 70], [180, 255, 255]),
    "blue": ([94, 80, 2], [126, 255, 255]),
    "purple": ([129, 50, 70], [158, 255, 255]),
    "green": ([40, 40, 40], [70, 255, 255])
}

# For RGB mode, example ranges (adjust as needed)
RGB_RANGES = {
    "red": (np.array([100, 0, 0]), np.array([255, 100, 100])),
    "blue": (np.array([0, 0, 100]), np.array([100, 100, 255])),
    "purple": (np.array([100, 0, 100]), np.array([255, 100, 255])),
    "green": (np.array([0, 100, 0]), np.array([100, 255, 100]))
}

# ==========================================
# Home Station URL (for sending image & data)
# ==========================================
HOME_STATION_URL = "http://your-home-station-address/api/upload"  # Modify accordingly


# ==========================================
# Utility: Compute 3D Position & Quadrant
# ==========================================
def compute_3d_position(bbox):
    """
    Compute approximate 3D position (in cm) using the pinhole camera model.
    bbox: [x, y, w, h] in pixel coordinates (using width as perceived diameter).
    Returns: (x_real, y_real, distance, quadrant, center_point)
    """
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
    x_real = (x_offset * distance) / FOCAL_LENGTH_PIXELS
    y_real = (y_offset * distance) / FOCAL_LENGTH_PIXELS
    if x_center < cx and y_center < cy:
        quadrant = "Top-Left"
    elif x_center >= cx and y_center < cy:
        quadrant = "Top-Right"
    elif x_center < cx and y_center >= cy:
        quadrant = "Bottom-Left"
    else:
        quadrant = "Bottom-Right"
    return (x_real, y_real, distance, quadrant, (x_center, y_center))


# ==========================================
# Utility: Color Filter Mask Function
# ==========================================
def color_filter_mask(frame):
    """
    Returns a binary mask based on the selected color thresholds.
    Works in HSV or RGB mode.
    """
    if COLOR_MODE == "HSV":
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if SELECTED_COLOR.lower() == "red":
            lower1 = np.array(HSV_RANGES["red"][0])
            upper1 = np.array(HSV_RANGES["red"][1])
            lower2 = np.array(HSV_RANGES["red"][2])
            upper2 = np.array(HSV_RANGES["red"][3])
            mask1 = cv2.inRange(hsv, lower1, upper1)
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)
        elif SELECTED_COLOR.lower() in HSV_RANGES:
            lower = np.array(HSV_RANGES[SELECTED_COLOR.lower()][0])
            upper = np.array(HSV_RANGES[SELECTED_COLOR.lower()][1])
            mask = cv2.inRange(hsv, lower, upper)
        else:
            mask = None
    elif COLOR_MODE == "RGB":
        if SELECTED_COLOR.lower() in RGB_RANGES:
            lower, upper = RGB_RANGES[SELECTED_COLOR.lower()]
            mask = cv2.inRange(frame, lower, upper)
        else:
            mask = None
    else:
        mask = None

    kernel = np.ones((5, 5), np.uint8)
    if mask is not None:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


# ==========================================
# YOLOv9 Training Function
# ==========================================

# Modify your train_model function
def train_model():
    print("Starting YOLOv9 training...")
    # Use the existing MODEL_CONFIG object directly
    MODEL_CONFIG.train(data=DATA_CONFIG, epochs=100, imgsz=640)
    MODEL_CONFIG.save(MODEL_WEIGHTS)
    print("Training complete. Model saved as", MODEL_WEIGHTS)
    return MODEL_CONFIG


# ==========================================
# YOLOv9 Detection Function (Continuous)
# ==========================================
def run_detection(model):
    cap = cv2.VideoCapture(1)  # Adjust camera index if needed
    if not cap.isOpened():
        print("Failed to open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # Resize frame for display
        display_frame = cv2.resize(frame, (FRAME_WIDTH_PIXELS, FRAME_HEIGHT_PIXELS))

        # Run YOLOv9 inference on the frame
        results = model(frame)[0]  # Process the first result
        # Check if any detections are present
        if results.boxes is not None and len(results.boxes) > 0:
            # For simplicity, take the first detected box (assumes one balloon per frame)
            box = results.boxes.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = box.astype(int)
            w = x2 - x1
            h = y2 - y1
            bbox = [x1, y1, w, h]

            # Extract the ROI from the display frame and verify with color filter
            roi = display_frame[y1:y1 + h, x1:x1 + w]
            if roi.size > 0:
                mask = color_filter_mask(roi)
                color_pixels = cv2.countNonZero(mask)
                total_pixels = roi.shape[0] * roi.shape[1]
                if total_pixels > 0 and (color_pixels / total_pixels) >= 0.5:
                    pos_data = compute_3d_position(bbox)
                    if pos_data is not None:
                        x_real, y_real, distance, quadrant, detected_center = pos_data
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(display_frame, (int(detected_center[0]), int(detected_center[1])), 5, (0, 0, 255),
                                   -1)
                        center_frame = (FRAME_WIDTH_PIXELS // 2, FRAME_HEIGHT_PIXELS // 2)
                        cv2.circle(display_frame, center_frame, 5, (255, 0, 0), -1)
                        cv2.line(display_frame, center_frame, (int(detected_center[0]), int(detected_center[1])),
                                 (0, 255, 255), 2)
                        cv2.line(display_frame, (FRAME_WIDTH_PIXELS // 2, 0),
                                 (FRAME_WIDTH_PIXELS // 2, FRAME_HEIGHT_PIXELS), (255, 255, 255), 1)
                        cv2.line(display_frame, (0, FRAME_HEIGHT_PIXELS // 2),
                                 (FRAME_WIDTH_PIXELS, FRAME_HEIGHT_PIXELS // 2), (255, 255, 255), 1)
                        info_text = f"Pos: ({x_real:.1f}cm, {y_real:.1f}cm, {distance:.1f}cm) {quadrant}"
                        cv2.putText(display_frame, info_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                                    2)
                    else:
                        cv2.putText(display_frame, "Position error", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 0, 255), 2)
                else:
                    cv2.putText(display_frame, "Color mismatch", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 0, 255), 2)
            else:
                cv2.putText(display_frame, "Empty ROI", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(display_frame, "No object detected", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Balloon Detection", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cv2.imwrite("detected_balloon.jpg", display_frame)
            print("Output saved as detected_balloon.jpg")
            # Optionally send to home station here:
            # send_to_home_station("detected_balloon.jpg", {...})
            break

    cap.release()
    cv2.destroyAllWindows()


# ==========================================
# Main Routine: Train (if needed) & Run Detection
# ==========================================
def main():
    if not os.path.exists(MODEL_WEIGHTS):
        print("No trained model found. Starting training...")
        model = train_model()
    else:
        print("Loading trained model...")
        model = YOLO(MODEL_WEIGHTS)

    time.sleep(0.5)  # Allow camera to warm up
    run_detection(model)


if __name__ == "__main__":
    main()
