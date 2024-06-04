import cv2
from ultralytics import YOLO  # Import YOLO from ultralytics
import torch
import numpy as np
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set Seaborn style
sns.set(style="whitegrid", context="talk")

# Load YOLO model
model = YOLO('../Weights/yolov8n.pt')

# Initialize ORB
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Placeholder for the last known descriptors and keypoints for tracked objects
last_known_descriptors = {}
last_known_keypoints = {}

# Directory containing the image frames
frames_directory = "Jet2/img"

# Get list of image files in sorted order
frame_files = sorted([f for f in os.listdir(frames_directory) if f.endswith('.jpg')])

# Real-time speed simulation: 30 FPS
fps = 30
frame_duration = 1.0 / fps

# Initialize counters and lists for plotting
detection_count = 0
non_detection_count = 0
frame_times = []
confidence_levels = []
keypoint_counts = []
processing_times = []
cumulative_confidence = 0
cumulative_confidences = []
detection_rates = []
keypoint_survival_rates = []

# Process each frame in the directory
for frame_file in frame_files:
    start_time = time.time()

    frame_path = os.path.join(frames_directory, frame_file)
    frame = cv2.imread(frame_path)

    # Resize frame for faster processing
    frame = cv2.resize(frame, (640, 480))
    YOLO_display_frame = frame.copy()
    orb_display_frame_raw = frame.copy()
    original = frame.copy()

    # Preprocess the frame for YOLOv8 and perform inference
    results = model(frame)

    # Initial computing of keypoints in entire frame
    kp_all, des_all = orb.detectAndCompute(original, None)

    # Dictionary to keep track of current frame's descriptors
    current_descriptors = {}
    current_keypoints = {}

    # To keep track of if object is detected
    detection_status = False
    frame_confidences = []

    # Check if results is a list and handle accordingly
    if isinstance(results, list) and len(results) > 0:
        result = results[0]

        if hasattr(result, 'boxes'):
            boxes = result.boxes
            for box in boxes:
                if isinstance(box.xyxy, torch.Tensor):
                    bbox = box.xyxy.cpu().numpy()
                else:
                    bbox = box.xyxy

                x1, y1, x2, y2 = bbox[0, :4].astype(int)
                conf = box.conf.cpu().numpy()[0]
                cls_id = int(box.cls.cpu().numpy()[0])

                confidence_threshold = 0.3

                if conf < confidence_threshold:
                    detection_status = False
                else:
                    detection_status = True

                label = f"{result.names[cls_id]}: {conf:.2f}"

                if conf > confidence_threshold:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.rectangle(YOLO_display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(YOLO_display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    kp_within_box = []
                    des_within_box = []
                    for kp, des in zip(kp_all, des_all):
                        if x1 <= kp.pt[0] <= x2 and y1 <= kp.pt[1] <= y2:
                            kp_within_box.append(kp)
                            des_within_box.append(des)

                    if kp_within_box and des_within_box:
                        current_keypoints[label] = kp_within_box
                        current_descriptors[label] = np.array(des_within_box)

                    frame_confidences.append(conf)

            if detection_status:
                detection_count += 1
            else:
                non_detection_count += 1
        else:
            non_detection_count += 1
    else:
        non_detection_count += 1

    matched_kps = []  # Initialize matched_kps here
    if not detection_status:
        kp_current, des_current = orb.detectAndCompute(original, None)

        for label, des_last in last_known_descriptors.items():
            if des_last is not None and des_current is not None:
                matches = bf.match(des_last, des_current)
                matched_kps = []
                matched_dess = []

                for match in matches:
                    matched_kp = kp_current[match.trainIdx]
                    matched_des = des_current[match.trainIdx]
                    matched_kps.append(matched_kp)
                    matched_dess.append(matched_des)

                    cv2.circle(frame, (int(matched_kp.pt[0]), int(matched_kp.pt[1])), 5, (255, 0, 0), 2)
                    cv2.putText(frame, label, (int(matched_kp.pt[0]), int(matched_kp.pt[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                if matched_kps and matched_dess:
                    current_keypoints[label] = matched_kps
                    current_descriptors[label] = np.array(matched_dess)

    last_known_descriptors = current_descriptors
    last_known_keypoints = current_keypoints

    keypoints_all, _ = orb.detectAndCompute(orb_display_frame_raw, None)
    for keypoint in keypoints_all:
        x, y = keypoint.pt
        cv2.circle(orb_display_frame_raw, (int(x), int(y)), 3, (255, 0, 0), -1)

    cv2.imshow('YOLO Output', YOLO_display_frame)
    cv2.imshow('orb_display_frame_raw', orb_display_frame_raw)
    cv2.imshow('YOLOv8 + ORB Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_end_time = time.time()
    frame_duration = frame_end_time - start_time
    frame_times.append(frame_duration)
    confidence_levels.append(np.mean(frame_confidences) if frame_confidences else 0)
    keypoint_counts.append(len(kp_all))
    processing_times.append(frame_duration)
    cumulative_confidence += np.mean(frame_confidences) if frame_confidences else 0
    cumulative_confidences.append(cumulative_confidence)
    detection_rate = detection_count / (detection_count + non_detection_count)
    detection_rates.append(detection_rate)

    # Keypoint survival rate
    keypoint_survival_rate = len(matched_kps) / len(kp_all) if len(kp_all) > 0 else 0
    keypoint_survival_rates.append(keypoint_survival_rate)

cv2.destroyAllWindows()

print(f"Total detections: {detection_count}")
print(f"Total non-detections: {non_detection_count}")

total_frames = len(frame_files)
accuracy = detection_count / total_frames
print(f"Final accuracy: {accuracy:.2f} ({detection_count}/{total_frames} frames with detections)")

# Data processing for plots
df = pd.DataFrame({
    'Frame': [f for f in range(len(frame_times))],
    'Confidence': confidence_levels,
    'Keypoints': keypoint_counts,
    'Processing Time': processing_times,
    'Cumulative Confidence': cumulative_confidences,
    'Detection Rate': detection_rates,
    'Keypoint Survival Rate': keypoint_survival_rates
})

# Plotting individual plots
plt.figure(figsize=(12, 6))
sns.lineplot(x='Frame', y='Confidence', data=df, label='Confidence')
sns.lineplot(x='Frame', y='Keypoints', data=df, label='Keypoints')
plt.title('Detection Confidence and Keypoints Over Time')
plt.xlabel('Frame')
plt.ylabel('Value')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
sns.histplot(df['Keypoints'], bins=20, kde=True)
plt.title('Keypoint Distribution in Detected Objects')
plt.xlabel('Keypoint Count')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(12, 6))
sns.lineplot(x='Frame', y='Processing Time', data=df, color='red')
plt.title('Frame Processing Time')
plt.xlabel('Frame')
plt.ylabel('Time (seconds)')
plt.show()

plt.figure(figsize=(12, 6))
sns.scatterplot(x='Confidence', y='Keypoints', data=df)
plt.title('Confidence vs. Keypoint Count')
plt.xlabel('Confidence')
plt.ylabel('Keypoint Count')
plt.show()

plt.figure(figsize=(12, 6))
sns.histplot(df['Processing Time'], bins=20, kde=True, color='green')
plt.title('Processing Time Distribution')
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(12, 6))
sns.lineplot(x='Frame', y='Cumulative Confidence', data=df, color='purple')
plt.title('Cumulative Confidence Over Time')
plt.xlabel('Frame')
plt.ylabel('Cumulative Confidence')
plt.show()

plt.figure(figsize=(12, 6))
sns.lineplot(x='Frame', y='Detection Rate', data=df, color='orange')
plt.title('Detection Rate Over Time')
plt.xlabel('Frame')
plt.ylabel('Detection Rate')
plt.show()

plt.figure(figsize=(12, 6))
sns.lineplot(x='Frame', y='Keypoint Survival Rate', data=df, color='brown')
plt.title('Keypoint Survival Rate Over Time')
plt.xlabel('Frame')
plt.ylabel('Keypoint Survival Rate')
plt.show()

# Keypoints Detected Per Object
object_keypoints = {key: len(value) for key, value in last_known_keypoints.items()}
objects = list(object_keypoints.keys())
keypoints = list(object_keypoints.values())

plt.figure(figsize=(12, 6))
sns.barplot(x=keypoints, y=objects, palette='coolwarm')
plt.title('Keypoints Detected Per Object')
plt.xlabel('Keypoints')
plt.ylabel('Object')
plt.show()

plt.figure(figsize=(12, 6))
sns.scatterplot(x='Confidence', y='Keypoint Survival Rate', data=df)
plt.title('YOLO Confidence vs. Keypoint Survival Rate')
plt.xlabel('Confidence')
plt.ylabel('Keypoint Survival Rate')
plt.show()

plt.figure(figsize=(12, 6))
sns.lineplot(x='Frame', y='Confidence', data=df, label='YOLO Confidence', color='blue')
sns.lineplot(x='Frame', y='Keypoint Survival Rate', data=df, label='Keypoint Survival Rate', color='green')
plt.title('YOLO Confidence vs. Keypoint Survival Rate Over Time')
plt.xlabel('Frame')
plt.ylabel('Value')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
sns.lineplot(x='Frame', y='Detection Rate', data=df, label='Detection Rate', color='orange')
sns.lineplot(x='Frame', y='Keypoint Survival Rate', data=df, label='Keypoint Survival Rate', color='green')
plt.title('Detection Rate vs. Keypoint Survival Rate Over Time')
plt.xlabel('Frame')
plt.ylabel('Rate')
plt.legend()
plt.show()
