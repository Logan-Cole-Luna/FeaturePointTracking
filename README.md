DualSight Documentation
May 22, 2024


Introduction:
DualSight is a real-time object tracking system that combines YOLO (You Only Look Once) for object detection
and ORB (Oriented FAST and Rotated BRIEF) for feature tracking. This integration enhances the tracking accuracy
and robustness in dynamic environments.


Prerequisites and Dependencies:
- Python 3.8+
- OpenCV 4.5+
- Ultralytics YOLO library
- NumPy
Install the necessary dependencies using pip:

    pip install opencv-python ultralytics numpy


Setup Instructions:
Download the YOLO model file (yolov8n.pt) from the Ultralytics YOLO repository.
Place the model file in the same directory as the script or update the script to point to its location.
Organize your image frames in a directory (e.g., Jet2/img) or utilize the provided dataset Jet2.


Usage Instructions:
Place your image frames in the frames_directory specified in the script. Run the script using Python:

    python dualsight.py

The script will process each frame, perform object detection and tracking, and display the results in real-time.


Code Explanation:
- Loading YOLO Model: The script initializes the YOLO model using the Ultralytics library.
- Initializing ORB: ORB is initialized to detect and describe keypoints.
- Processing Frames: The script processes each frame in the specified directory, performs object detection using YOLO, 
  and tracks features using ORB.
- Displaying Results: The results are displayed in real-time, showing detected objects and tracked keypoints.


Example:
Place your image frames in the Jet2/img directory and run the script. You should see real-time detection and
tracking results displayed in separate windows.


Performance Metrics:
The script calculates and prints the detection count, non-detection count, and final accuracy based on the processed frames.
Additionally, running PlottingFunct.py will produce graphs displaying the performance of DualSight, and running DataExtraction.py
will provide a large amount of data regarding the run.


Limitations and Future Work:
- The current implementation is designed for single-object tracking.
- Future work includes supporting multi-object tracking and integrating more advanced tracking algorithms.

This asset is built upon YOLO (You Only Look Once) and ORB (Oriented FAST and Rotated BRIEF).

YOLO is developed by Joseph Redmon et al. and is available under the MIT License.
The code and further details can be found at https://github.com/ AlexeyAB/darknet.

ORB is developed by Ethan Rublee et al. and is available under the BSD License.
The original paper can be referenced as "ORB: an efficient alternative to SIFT or SURF" by Rublee et al. (2011),
and the implementation can be found in OpenCV at https://github.com/opencv/opencv.
This research has adhered to the licenses and terms of use for both YOLO and ORB.