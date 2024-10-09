# Counting number of people currently in the room using OpenCV

## Table of Contents
1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Software and Tools](#software-and-tools)
4. [Dataset Preparation](#dataset-preparation)
5. [Configuration File](#configuration-file)
6. [Model Setup](#model-setup)
7. [Training the Model](#training-the-model)
    - [Rename Trained Weights](#Rename-trained-weights)
    - [Model Inference](#model-inference)
7. [Explanation Code](#explanation-code)
    - [Python Code for Box Detection](#python-code-for-box-detection)
8. [Output Explanation](#output-explanation)
6. [Applications](#applications)
7. [Future Scope](#future-scope)
8. [Summary](#summary)


## Introduction

This Python script demonstrates how to use a pre-trained MobileNet SSD model for object detection, specifically to count the number of persons in a video feed.

## Requirements

- **Python 3.x**
- **OpenCV 4.x**
- **NumPy(numpy)**

## Software and Tools

- **Python:** Programming language used for the script.
- **OpenCV:** Library for computer vision tasks.
- **NumPy:** Library for numerical computing
## Dataset Preparation

No specific dataset preparation is needed as the script uses a pre-trained MobileNet SSD model.

## Configuration File

The configuration files used are:

- deploy.prototxt: Defines the architecture of the neural network.
- mobilenet_iter_73000.caffemodel: Contains the weights of the trained MobileNet SSD model.

## Model Setup
1.The MobileNet SSD model is loaded using 'cv2.dnn.readNetFromCaffe'.

2.Install OpenCV: Open a command prompt in the root directory of your project and install OpenCV using the following command(Install them in the root directory after creating the Virtual environment): 
```bash
pip install opencv-python
pip install opencv-contrib-python
```

3. Activate Virtual Environment: Activate your virtual environment using the following command.
```bash
Open a command prompt in the root directory of your project and activate virtual environment using the following command:
-python -m venv myenv
-cd myenv
-cd Scripts
-Activate
 ```

4. Once the virtal environment is activated go back to your root directory using the following command:
```bash
-Use command cd .. till you reach the root directory
-Then run the app.py file using command : python main_file.py
```

## Training the Model
The model has already been pre-trained on the COCO dataset for people detection tasks.

### Rename Trained Weights
Not applicable in this context since the weights are pre-trained.

### Model Inference
The script performs inference on each frame of the video feed to detect persons using the MobileNet SSD model.

## Explanation Code 
### Python Code for counting people currently in the room.
The code iterates through detections made by the model, filters out detections with low confidence, and counts persons while drawing bounding boxes around them:

```python
import cv2
import numpy as np

# Load the pre-trained MobileNet SSD model and the prototxt file
net = cv2.dnn.readNetFromCaffe(
    'deploy.prototxt', 
    'mobilenet_iter_73000.caffemodel'
)

# List of class labels MobileNet SSD was trained to detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Open a video file or an image file or a webcam feed
cap = cv2.VideoCapture(0)  # Use 0 for webcam or replace with 'path/to/video' for a video file

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break

    # Get the height and width of the frame
    (h, w) = frame.shape[:2]

    # Preprocess the frame: resize to 300x300 pixels and normalize it
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    # Pass the blob through the network and obtain the detections and predictions
    net.setInput(blob)
    detections = net.forward()

    # Initialize a counter for people
    person_count = 0

    # Loop over the detections
    for i in range(detections.shape[2]):
        # Extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by ensuring the confidence is greater than a minimum threshold
        if confidence > 0.2:
            # Extract the index of the class label from the detections
            idx = int(detections[0, 0, i, 1])

            # If the class label is "person", increment the person count
            if CLASSES[idx] == "person":
                person_count += 1

                # Compute the (x, y)-coordinates of the bounding box for the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Draw the bounding box around the detected object
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                label = f"{CLASSES[idx]}: {confidence:.2f}"
                cv2.putText(frame, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the output frame
    cv2.putText(frame, f"Person Count: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Frame", frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

```

## Output Explanation
The script displays the video feed with bounding boxes around detected persons and counts the number of persons in the frame.

## Applications
This script can be used for real-time person detection in the following:

- **Public Safety and Security**:Enhance security measures by monitoring and counting individuals in real-time.
- **Education and Workplace Attendance**:  Automate attendance tracking by accurately counting people entering or leaving 
  designated areas. 
- **Retail Analytics and Customer Insights**:  Analyze customer traffic patterns and behavior to optimize store layouts, 
  improve customer service, and enhance marketing strategies based on foot traffic and visitor demographics.
- **Home Security and Personal Safety**:  Enhance home security systems by detecting and alerting residents to unauthorized 
  access or approaching individuals, ensuring personal safety and property protection.

## Future Scope
Future enhancements and extensions to this project could include:

- **Multi-Object Tracking**: Implement algorithms for tracking multiple persons across frames to analyze movement patterns 
  and behavior over time, enhancing applications like crowd management and surveillance.
- **Real-time Analytics and Insights**: Develop capabilities to generate real-time analytics and insights from detected 
  persons, such as crowd density estimation, heat maps, and predictive analytics for proactive decision-making.
- **Integration with IoT Devices**:Connect the system with IoT devices and sensors to enhance data collection and 
  integration for smart city applications, automated alerts, and seamless data sharing across platforms.
- **Improved Accuracy**: Fine-tuning the model and using more sophisticated data augmentation techniques to improve detection accuracy.
- **Deployment**: Creating a web or mobile application to deploy the model for practical use.


## Summary

This project utilizes a pre-trained MobileNet SSD model to detect and count persons in real-time from video feeds. It leverages OpenCV for image processing and inference, displaying bounding boxes around detected persons and providing a count. Applications include enhancing security in public spaces, automating attendance systems, and providing analytics for retail and urban planning. Future enhancements could involve integrating advanced object detection models, real-time analytics, and IoT connectivity for broader applications in smart cities and personalized customer experiences.

