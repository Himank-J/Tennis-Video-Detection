# Tennis-Video-Detection

## 🎯 AIM

The aim of the project is to implement a video detection model which is capable of identification of players involved in the match, tennis ball and the keypoints of the tennis court. Once detected, we will also generate a mini map of the court which will showcase player and ball movements. At the end we also calculate player and ball related statistics such as player speed, ball speed etc.

---
## 📹 Demo

https://github.com/user-attachments/assets/46cf15f6-7cfe-4857-9dca-4cef6970b5eb

--- 
## ⭐ Features

- Player detecion using YoloV8
- Tennis Ball detection using custom trained yolo model
- Tennis court keypoint detection using custom resnet model
- Creation of Tennis mini court and tracking player movement on mini court
- Calculating PLayer and Ball statistics: Avg. speed of Shot, Avg. speed of Players

---

## 🗒️ Requirements

- Python 3.9 and above
- Ultralytics
- Torch
- Open-CV
- Matplotlib
- Numpy
- Pandas

---

## Code Flow & Details

<img width="855" alt="Screenshot 2024-07-18 at 10 27 13 PM" src="https://github.com/user-attachments/assets/f7317b6b-a598-4c78-90ea-8596b0a2b427">


The above image shows different classes inside our code and seuquence in which the different functions from these classes are called (the number in brackets) to get the final output. Below I have mentioned the classes and the core functionality of each:
<br>

1.) [Training](https://github.com/Himank-J/Tennis-Video-Detection/tree/main/training) 🔗

- This directory consists of two files:
  - [Tennis Court Keypoints Detection](https://github.com/Himank-J/Tennis-Video-Detection/blob/main/training/Yolo_tennis_court_keypoints.ipynb) 🔗 
    - Contains script for training custom resnet model for detecting keypoints of a tennis court.
    - [Dataset](https://drive.google.com/file/d/1lhAaeQCmk2y440PmagA0KmIVBIysVMwu/view?usp=drive_link) 🔗 - The dataset consists of 8841 images, which were separeted to train set (75%) and validation set (25%). Each image has 14 annotated points. The resolution of images is 1280×720.

  - [Tennis Ball Detection](https://github.com/Himank-J/Tennis-Video-Detection/blob/main/training/Yolo_tennis_training.ipynb) 🔗
    -  Contains script for training custom yolov8 model for detecting tennis ball
    -  [Dataset](https://universe.roboflow.com/viren-dhanwani/tennis-ball-detection)

2.) [Utils](https://github.com/Himank-J/Tennis-Video-Detection/tree/main/utils) 🔗

- Video utils: for reading input video, writing frames as output video and for annotating frame number in output video
- BBox utils: for all generic functions related to bounding boxes like get center coordinate of a box, get height of box, find distance between two points etc
- Conversions: for converting pixel to meters and vice versa
- Player stats: for drawing the section where stats related to players like Avg. player speed and related to ball like Avg. ball speed are placed

3.) [PlayerTracker](https://github.com/Himank-J/Tennis-Video-Detection/blob/main/trackers/player_tracker.py) 🔗

The PlayerTracker class is designed for detecting, tracking, and managing player movements in video frames using a YOLO (You Only Look Once) model. Here's a summary of its functionalities:

- Initialization: Load the YOLO model for detecting players.
- Player Detection:
  - Detect players in each frame, either by processing the frames or by loading saved results.
  - Save the detection results if a path is provided.
- Drawing Bounding Boxes: Annotate video frames with bounding boxes and player IDs.
- Choosing and Filtering Players: Identify and track the two players closest to the court keypoints across all frames.
  - Calculate the distance of each detected player to court keypoints.
  - Select and filter the closest players for further tracking and analysis.

4.) [BallTracker](https://github.com/Himank-J/Tennis-Video-Detection/blob/main/trackers/ball_tracker.py) 🔗

The BallTracker class is designed for detecting, tracking, and managing tennis ball movements in video frames using a YOLO (You Only Look Once) model. Here's a summary of its functionalities:

- Initialization: Load the YOLO model for detecting balls.
- Ball Detection:
  - Detect ball positions in each frame, either by processing the frames or by loading saved results.
  - Save the detection results if a path is provided.
  - Interpolating Ball Positions: Fill in missing ball positions in the detection results to handle frames where the ball is not detected.
- Identifying Ball Hits: Analyze changes in the ball's vertical position to identify frames where the ball is hit.
  - Calculate the vertical midpoint of the ball's bounding box.
  - Compute rolling mean and changes in vertical position.
  - Detect frames with significant changes indicating a ball hit.
- Drawing Bounding Boxes: Annotate video frames with bounding boxes and ball IDs.

5.) [CourtLineDetector](https://github.com/Himank-J/Tennis-Video-Detection/blob/main/court_line_detectors/court_line_detect.py) 🔗

- Initialization:
  - Load a pre-trained ResNet-50 model.
  - Modify the final fully connected layer to output 28 values.
  - Load the model weights from the specified path.
  - Set up image transformations for preprocessing.
  - Keypoint Prediction:

- Convert the image from BGR to RGB.
  - Transform the image to match the input requirements of the model.
  - Predict keypoints using the ResNet-50 model.
  - Scale keypoints from the model's output dimensions (224x224) to the original image dimensions.

- Drawing Keypoints:
  - Draw circles and labels on the image at the predicted keypoint positions.
  - For video frames, apply the keypoint drawing method to each frame.
 
6.) [MiniCourt](https://github.com/Himank-J/Tennis-Video-Detection/blob/main/mini_court/mini_court.py) 🔗

- Initialization:
  - Set up the mini court's background box, position, and key points for drawing based on the input frame.

- Court Drawing:
  - Draw the mini court, including lines and key points, on a background rectangle in each frame.

- Coordinate Conversion:
  - Convert real court distances and positions to the mini court's pixel coordinates, enabling visualization of player and ball movements.

- Bounding Box Conversion:
  - Convert bounding boxes from the real court to mini court coordinates, facilitating accurate representation of player and ball positions.

- Visualization:
  - Draw the mini court and relevant positions on each frame, providing a clear and concise visual representation of the game dynamics.
 
7.) Player Stats:

- [Identification of ball shot frames](https://github.com/Himank-J/Tennis-Video-Detection/blob/main/analysis/ball_analysis.ipynb) 🔗 :

  - The get_ball_shot_frames function identifies the frames in which a ball is hit during a sequence of video frames. It processes the ball positions frame by frame and uses a rolling mean of the ball's vertical position to detect significant changes, which indicate a ball hit.
  - Initialization: The function prepares the ball positions and computes necessary columns.
  - Rolling Mean Calculation: Smoothes out the ball's vertical position changes to detect significant trends.
  - Delta Calculation: Measures the rate of change in the smoothed vertical position.
  - Change Detection: Identifies frames where the direction of the vertical movement changes significantly and persists for a sufficient number of frames, indicating a ball hit.
  - Result Extraction: Returns the list of frame numbers where ball hits are detected.
 
- Calculation of Player Stats
  - Initialization: Sets up initial player statistics. 
  - Loop Through Ball Shot Frames: Calculates statistics for each detected ball shot interval.
  - Ball Shot Statistics: Measures and calculates ball shot speed.
  - Identify Player Who Hit the Ball: Determines which player hit the ball.
  - Opponent Player Statistics: Measures and calculates the opponent player's speed.
  - Update Player Stats: Updates the player statistics with the current frame's data.
  - Create Final DataFrame: Compiles all statistics into a DataFrame and calculates average speeds.
 
---
## 📰 References

- https://github.com/yastrebksv/TennisCourtDetector
- https://blog.ml6.eu/improving-tennis-court-line-detection-with-machine-learning-90f82dccdf1d
- https://medium.com/@kosolapov.aetp/tennis-analysis-using-deep-learning-and-machine-learning-a5a74db7e2ee
- https://stackoverflow.com/questions/49236489/how-to-calculate-camera-movement-speed-between-to-frames#:~:text=If%20you%20want%20to%20compute,or%20frame%2Dtime)%20estimate.
