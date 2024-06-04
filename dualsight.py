"""
DualSight: Integrated YOLO and ORB Tracking

This script combines YOLO (You Only Look Once) for object detection with
ORB (Oriented FAST and Rotated BRIEF) for feature tracking.
It processes a series of image frames, performs object detection using YOLO,
and tracks features using ORB. The results are displayed in real-time,
showing detected objects and tracked keypoints.

Requirements:
- Python 3.8+
- OpenCV 4.5+
- Ultralytics YOLO library
- PyTorch
- NumPy

Install the necessary dependencies using:
pip install opencv-python ultralytics torch numpy

Setup Instructions:
1. Download the YOLO model file (yolov8n.pt) from the Ultralytics YOLO repository.
2. Place the model file in the same directory as the script or update the script to point to its location.
3. Organize your image frames in a directory (e.g., Jet2/img).

Usage Instructions:
1. Place your image frames in the frames_directory specified in the script.
2. Run the script using Python:
   python dualsight.py
3. The script will process each frame, perform object detection and tracking,
and display the results in real-time.

Performance Metrics:
The script calculates and prints the detection count, non-detection count,
and final accuracy based on the processed frames.

Limitations and Future Work:
- The current implementation is designed for single-object tracking.
- Future work includes supporting multi-object tracking and integrating more advanced tracking algorithms.

Acknowledgments
We would like to thank the creators of YOLO (You Only Look Once) and ORB (Oriented FAST and Rotated BRIEF)
for providing the foundational tools that made this research possible.
YOLO is developed by Joseph Redmon et al. and is available under the MIT License.
The code and further details can be found at https://github.com/AlexeyAB/darknet.
ORB is developed by Ethan Rublee et al. and is available under the BSD License.
The original paper can be referenced as "ORB: an efficient alternative to SIFT or SURF" by Rublee et al. (2011),
and the implementation can be found in OpenCV at https://github.com/opencv/opencv.
We have adhered to the licenses and terms of use for both YOLO and ORB in this research.
"""

import cv2
from ultralytics import YOLO  # Import YOLO from ultralytics
import numpy as np
import os
import torch

# Load YOLO model
model = YOLO('yolov8n.pt')

# Initialize ORB
orb = cv2.ORB_create()
#orb = cv2.ORB_create(scoreType=cv2.ORB_HAMMING)
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

# Initialize counters
detection_count = 0
non_detection_count = 0

# Process each frame in the directory
for frame_file in frame_files:
    frame_path = os.path.join(frames_directory, frame_file)
    frame = cv2.imread(frame_path)

    #frame_count += 1
    #if frame_count % frame_skip != 0:
    #    continue  # Skip frame

    # Resize frame for faster processing
    frame = cv2.resize(frame, (640, 480))
    YOLO_display_frame = frame.copy()  # This is where we will display YOLO
    orb_display_frame_raw = frame.copy()  # This is where we will draw all Raw ORB keypoints
    original = frame.copy()  # This is where we will compare ORB keypoints detected within bbox area to all Keypoints

    # Preprocess the frame for YOLOv8 and perform inference
    results = model(frame)

    # Initial computing of keypoints in entire frame
    kp_all, des_all = orb.detectAndCompute(original, None)

    # Dictionary to keep track of current frame's descriptors
    current_descriptors = {}
    current_keypoints = {}

    # To keep track of if object is detected
    detection_status = False

    # Check if results is a list and handle accordingly
    if isinstance(results, list) and len(results) > 0:
        result = results[0]

        if hasattr(result, 'boxes'):

            boxes = result.boxes
            for box in boxes:
                # Convert tensor to numpy array if it's not already
                if isinstance(box.xyxy, torch.Tensor):
                    bbox = box.xyxy.cpu().numpy()
                else:
                    bbox = box.xyxy

                # Extract coordinates
                x1, y1, x2, y2 = bbox[0, :4].astype(int)

                # Extract confidence and class ID
                conf = box.conf.cpu().numpy()[0]
                cls_id = int(box.cls.cpu().numpy()[0])

                confidence_threshold = 0.3

                # Check if no object detected (all confidence scores below threshold)
                if (conf < confidence_threshold):
                    print("\nNo objects detected in this frame.")
                    detection_status = False
                else:
                    detection_status = True

                # Should this be here or within loop?
                label = f"{result.names[cls_id]}: {conf:.2f}"

                if (conf > confidence_threshold):
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    # Detect and store ORB keypoints and descriptors for the current object

                    cv2.rectangle(YOLO_display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(YOLO_display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    # Label keypoints within the bounding box
                    kp_within_box = []
                    des_within_box = []
                    for kp, des in zip(kp_all, des_all):
                        if x1 <= kp.pt[0] <= x2 and y1 <= kp.pt[1] <= y2:
                            kp_within_box.append(kp)
                            des_within_box.append(des)
                            # Optionally, draw these keypoints
                            #cv2.circle(frame, (int(kp.pt[0]), int(kp.pt[1])), 3, (255, 0, 0), -1)

                    if kp_within_box and des_within_box:
                        current_keypoints[label] = kp_within_box
                        current_descriptors[label] = np.array(des_within_box)

            if detection_status:
                detection_count += 1
            else:
                non_detection_count += 1
        else:
            non_detection_count += 1
            print("\nNo objects detected in this frame.")
    else:
        non_detection_count += 1
        print("\nNo objects detected in this frame.")

    # Only proceed if no objects were detected by YOLO in the current frame
    if not detection_status:
        # Detect current keypoints and descriptors
        kp_current, des_current = orb.detectAndCompute(original, None)

        for label, des_last in last_known_descriptors.items():
            if des_last is not None and des_current is not None:
                # Perform the matching
                matches = bf.match(des_last, des_current)

                # To store matched keypoints and descriptors for the current object
                matched_kps = []
                matched_dess = []

                # print("uh")

                for match in matches:
                    # print("MATCH!!")
                    matched_kp = kp_current[match.trainIdx]  # Get the matched keypoint object
                    matched_des = des_current[match.trainIdx]  # Get the matched descriptor

                    # Store the matched keypoint and descriptor
                    matched_kps.append(matched_kp)
                    matched_dess.append(matched_des)

                    # Draw the matched keypoint
                    cv2.circle(frame, (int(matched_kp.pt[0]), int(matched_kp.pt[1])), 5, (255, 0, 0), 2)
                    cv2.putText(frame, label, (int(matched_kp.pt[0]), int(matched_kp.pt[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (0, 0, 255), 2)

                # If there are matched keypoints and descriptors, update the current dictionaries
                if matched_kps and matched_dess:
                    current_keypoints[label] = matched_kps
                    current_descriptors[label] = np.array(matched_dess)


    # Update last known descriptors and keypoints, these will be matched in next frame if no object is detected,
    # they also carry the labels attached to the current_descriptors
    last_known_descriptors = current_descriptors
    last_known_keypoints = current_keypoints

    # This is for raw display of points
    keypoints_all, _ = orb.detectAndCompute(orb_display_frame_raw, None)
    for keypoint in keypoints_all:
        x, y = keypoint.pt
        cv2.circle(orb_display_frame_raw, (int(x), int(y)), 3, (255, 0, 0), -1)  # Draw all keypoints as red dots

    cv2.imshow('YOLO Output', YOLO_display_frame)  # Display the frame with all ORB keypoints
    cv2.imshow('orb_display_frame_raw', orb_display_frame_raw)
    cv2.imshow('YOLOv8 + ORB Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()

# After processing all frames
print(f"Total detections: {detection_count}")
print(f"Total non-detections: {non_detection_count}")

# Print final accuracy based on detections
total_frames = len(frame_files)
accuracy = detection_count / total_frames
print(f"Final accuracy: {accuracy:.2f} ({detection_count}/{total_frames} frames with detections)")

