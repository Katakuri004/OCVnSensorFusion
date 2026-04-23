Overview

At Trekion.ai, we are building data infrastructure for Physical AI. Our work involves processing multi-modal sensor data from robotic systems, including synchronized video streams, IMU (Inertial Measurement Unit) readings, and depth information. This assessment is designed to evaluate your ability to work with real-world robotics sensor data.

You will be given raw data captured from a multi-sensor camera rig. Your task is to parse the proprietary binary formats, synchronize the data streams, and produce meaningful visual outputs that demonstrate your understanding of computer vision and sensor fusion concepts.

Input Data

You will receive three files, all from the same recording session:

File

Format

Description

recording2.mp4

H.264 video

Raw video from camera. 1920x1080 resolution, 30 fps, approximately 44 seconds duration. Wide-angle / fisheye lens.

recording2.imu

Binary

IMU sensor data. Binary file with header magic "TRIMU001". Contains accelerometer (m/s2), gyroscope (deg/s), and magnetometer (uT) readings sampled at approximately 568 Hz. Also includes temperature data.

recording2.vts

Binary

Video timestamp file. Binary file with header magic "TRIVTS01". Maps each video frame number to a hardware timestamp, enabling synchronization between the video frames and IMU readings.

Note: The .imu and .vts files use proprietary binary formats. Parsing these formats is part of the assessment. The header signatures and the data patterns within the files should give you enough information to reverse-engineer the structure. You are expected to inspect the binary data (using hex editors, Python struct module, etc.) and figure out the record layout yourself.

Tasks

Task 1: IMU Data Parsing and Synchronized Visualization

Parse the binary .imu and .vts files, and create a synchronized visualization that overlays IMU telemetry alongside the video frames.

Requirements

Reverse-engineer the .imu binary format to extract accelerometer (3-axis), gyroscope (3-axis), and magnetometer (3-axis) readings with their timestamps

Parse the .vts file to obtain frame-level timestamps for the video

Synchronize IMU data to video frames using the shared timestamp domain

Generate an output video that displays the camera feed alongside real-time plots of accelerometer, gyroscope, and magnetometer data (X, Y, Z axes), with a scrolling time window that tracks the current video frame

Include an on-screen HUD showing: frame number, timestamp, current sensor values, IMU sampling rate, camera FPS, temperature, and sync delay statistics (mean, median, max)

Task 2: Monocular Dense Depth Estimation

Generate dense depth maps from the video using a monocular depth estimation model.

Requirements

Apply a pre-trained monocular depth estimation model (e.g., MiDaS, Depth Anything, ZoeDepth, or similar) to each video frame

Produce a side-by-side output video: original RGB frame on the left, colorized depth map on the right

Use a perceptually meaningful colormap (e.g., inferno, magma, or turbo) for depth visualization

Handle the wide-angle / fisheye distortion gracefully (note any lens correction steps you apply, if any)

Task 3: Object and Scene Segmentation

Run object detection and/or scene segmentation on the video to identify and label objects and surfaces in the scene.

Requirements

Use a pre-trained model (e.g., YOLOv8/v9, Grounding DINO, SAM, Detectron2, or similar) for object detection and/or instance segmentation

Detect and label objects present in the scene with bounding boxes and class labels

If hands are visible in the video, detect and highlight them (hand detection / pose estimation is a bonus)

Produce an output video with detection overlays (bounding boxes, labels, confidence scores, and segmentation masks if applicable)


Confidential | Page 

Trekion.ai | Technical Assignment

Deliverables

Submit the following:

#

Deliverable

Details

1

Source code

All Python scripts / notebooks used. Code should be clean, well-commented, and reproducible. Include a README with setup instructions and dependencies.

2

IMU Sync Video

Output video (.mp4) showing camera feed with synchronized IMU telemetry plots and HUD overlay.

3

Depth Map Video

Output video (.mp4) showing side-by-side original frames and dense depth estimation.

4

Segmentation Video

Output video (.mp4) showing object detection / segmentation overlays on the original frames.

5

Brief write-up

A short document (1-2 pages) explaining: your approach to parsing the binary formats, model choices and reasoning, any challenges faced, and ideas for improvement.

Evaluation Criteria

Criterion

Weight

What We Look For

Binary format parsing

25%

Ability to reverse-engineer and correctly parse the .imu and .vts binary formats without documentation. Demonstrates low-level data handling skills.

Data synchronization

20%

Correct temporal alignment between video frames and IMU readings. Handling of different sampling rates and timestamp domains.

Computer vision quality

20%

Quality of depth estimation and segmentation results. Appropriate model selection and parameter tuning.

Code quality

15%

Clean, readable, well-structured code. Proper error handling. Easy to set up and reproduce.

Visual output quality

10%

Professional-looking visualizations. Clear labeling, good color choices, readable plots.

Write-up and communication

10%

Clear explanation of approach, decisions, and trade-offs. Shows depth of understanding.

Guidelines and Constraints

Language: Python is preferred. You may use Jupyter notebooks for exploration, but final scripts should be standalone .py files.

Libraries: You are free to use any open-source libraries (OpenCV, PyTorch, TensorFlow, NumPy, Matplotlib, etc.). Pre-trained models from model hubs (HuggingFace, PyTorch Hub, etc.) are encouraged.

Hardware: If you need GPU access, Google Colab (free tier) or Kaggle notebooks are acceptable. Include any Colab links if used.

AI assistance: Using AI tools (ChatGPT, Copilot, etc.) is allowed for general guidance, but the core logic, especially binary parsing, must reflect your own understanding. Be prepared to explain every line of your code in a follow-up call.

Timeline: Please complete and submit within 5 days of receiving this assignment.

Submission: Share a GitHub repository link (public or private with access granted to vivek.jaiswal@trekion.ai and itsvivekj@gmail.com) or a Google Drive folder containing all deliverables.

Hints

The .imu file starts with the header TRIMU001. Use Python's struct module to unpack binary data. The sensor values are stored as 32-bit floats in little-endian format.

The .vts file starts with the header TRIVTS01. It provides the mapping between frame indices and hardware timestamps.

Timestamps in both files share the same clock domain, which is how you synchronize them.

The video was captured with a wide-angle lens. You may notice barrel distortion, particularly at the edges. Consider whether this affects your depth/segmentation models and whether you should correct for it.

For depth estimation, Depth Anything V2 and MiDaS are good starting points that work well on monocular input.