import cv2
from ultralytics import YOLO  # Import YOLO from ultralytics
import torch
import numpy as np
import os
import pandas as pd

# Load YOLO model
model = YOLO('yolov8n.pt')  # Ensure you have this model downloaded or change the path accordingly

# Initialize ORB
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Placeholder for the last known descriptors and keypoints for tracked objects
last_known_descriptors = {}
last_known_keypoints = {}

# Directory containing the image frames
frames_directory = "Jet2/img"  # Update this path to your directory containing the frames

# Get list of image files in sorted order
frame_files = sorted([f for f in os.listdir(frames_directory) if f.endswith('.jpg')])

# Data collection list
data_points = []
keypoint_matches = []

# Ensure the directory exists for descriptor files
descriptors_folder = "descriptors_folder"
os.makedirs(descriptors_folder, exist_ok=True)

# Process each frame in the directory
for frame_idx, frame_file in enumerate(frame_files):
    frame_path = os.path.join(frames_directory, frame_file)
    frame = cv2.imread(frame_path)

    # Resize frame for faster processing
    frame = cv2.resize(frame, (640, 480))
    original = frame.copy()  # Make a copy of the original frame

    # Preprocess the frame for YOLOv8 and perform inference
    results = model(frame)

    # Initial computing of keypoints in entire frame
    kp_all, des_all = orb.detectAndCompute(original, None)

    # Dictionary to keep track of current frame's descriptors and keypoints
    current_descriptors = {}
    current_keypoints = {}

    if isinstance(results, list) and len(results) > 0:
        for result in results:
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

                    # Check for detection confidence
                    if conf > confidence_threshold:
                        label = f"{model.names[cls_id]}: {conf:.2f}"

                        # Unique identifier for the descriptor file
                        descriptor_file_name = f"descriptors_{frame_file}_{model.names[cls_id]}_{conf:.2f}.npy"

                        # Process keypoints within the bounding box
                        kp_within_box = []
                        des_within_box = []
                        for kp, des in zip(kp_all, des_all):
                            if x1 <= kp.pt[0] <= x2 and y1 <= kp.pt[1] <= y2:
                                kp_within_box.append(kp)
                                des_within_box.append(des)

                        if kp_within_box and des_within_box:
                            current_keypoints[label] = kp_within_box
                            current_descriptors[label] = np.array(des_within_box)

                            # Save descriptors to a NumPy file for each detected object
                            np.save(os.path.join(descriptors_folder, descriptor_file_name), np.array(des_within_box))

                            data_points.append({
                                'Frame': frame_file,
                                'Class': model.names[cls_id],
                                'Confidence': conf,
                                'BoundingBox': (x1, y1, x2, y2),
                                'Keypoints': len(kp_within_box),
                                'DescriptorFile': descriptor_file_name
                            })

                            # Save additional information for each keypoint detected within the YOLO bounding box
                            for kp_idx, (kp, des) in enumerate(zip(kp_within_box, des_within_box)):
                                kp_match_data = {
                                    'Frame': frame_file,
                                    'Class': model.names[cls_id],
                                    'Keypoint_ID': kp_idx,
                                    'Keypoint_Position': (kp.pt[0], kp.pt[1]),
                                    'Descriptor': des,
                                    'Confidence': conf
                                }
                                keypoint_matches.append(kp_match_data)

    # Update last known descriptors and keypoints
    last_known_descriptors = current_descriptors
    last_known_keypoints = current_keypoints

# Exporting to Excel
df = pd.DataFrame(data_points)
df.to_excel('tracking_results_with_descriptors.xlsx', index=False)

# Export keypoint match data to a separate Excel file
kp_df = pd.DataFrame(keypoint_matches)
kp_df.to_excel('keypoint_tracking_results.xlsx', index=False)

# Make sure to close any open windows if you're using cv2.imshow() to display the frames
cv2.destroyAllWindows()
