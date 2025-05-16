#  Football2FIFA - Football Analysis CV Pipeline ⚽
This project is an end-to-end Computer Vision pipeline for tracking players, referees, and the ball in football match footage. It uses object detection, tracking, clustering, and perspective transformation techniques to calculate player statistics such as speed, distance covered, team ball possession, etc. and provides the output with UI elements similar to a game of FIFA.

> This project was inspired and built upon a video tutorial by YouTube channel "Code in a Jiffy", called ["Build an AI/ML Football Analysis system with YOLO, OpenCV, and Python"](https://www.youtube.com/watch?v=neBZ6huolkg)

> The input videos for this project are from the Kaggle dataset, ["DFL Bundesliga Data Shootout"](https://www.kaggle.com/datasets/saberghaderi/-dfl-bundesliga-460-mp4-videos-in-30sec-csv)

## Features
- Object detection using a YOLOv5s model, trained on a [Roboflow Football Dataset](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/1) for improved player and ball detection.
- Player and referee tracking using Supervision (ByteTrack)
- Team assignment using K-Means Clustering on the players' shirt colors
- Perspective transformation for camera movement estimation
- Visual overlays of elliptical markers for players and triangles for the ball and closest players to the ball.
- Visual overlays for the possession stats and closest players of each team.
- Modular Python structure with reusable utilities

## Tech Stack
- Python, OpenCV, NumPy
- YOLOv5
- Supervision (ByteTrack)
- Roboflow and Kaggle for dataset sourcing and labeling
- Jupyter Notebook (for GPU training of the custom YOLOv5s model)

## Pipeline
1. **Object Detection**: YOLO is used to detect players, referees, and the ball.
2. **Fine-Tuning**: A Roboflow-labeled dataset is used to improve model performance for distinguishing referees, players, and ignoring audience.
3. **Tracking**: ByteTrack is used to assign unique IDs to players/referees across frames.
4. **Team Segmentation**: K-Means Clustering on shirt color pixels is used to assign players to teams.
5. **Perspective Transformation**: Pixel coordinates are mapped to real-world measurements for speed and distance stats.
6. **Visualization**: Visuals are drawn on each frame to match how a game of FIFA would display the players and possession.

## Results
<p align="center">
 <img src="https://github.com/user-attachments/assets/2ebaaba0-a283-48b9-9273-4715b3ccc409" alt="player1"/>
 <img src="https://github.com/user-attachments/assets/7057342e-439a-4659-9d45-f3e028642c05" alt="player2"/>
 <img src="https://github.com/user-attachments/assets/227882bc-3e8e-4325-a73f-573e72d72a64" alt="referee"/>
    <br>
    <em>Players and referee detected as separate objects</em>
</p>

<p align="center">
 <img src="https://github.com/unnamed-catalyst/Football2FIFA/blob/main/output_videos/README_results/001.gif" alt="player1" width=480 height=270/>
 <img src="https://github.com/unnamed-catalyst/Football2FIFA/blob/main/output_videos/README_results/001_output.gif" alt="player2" width=480 height=270/>

 <img src="https://github.com/unnamed-catalyst/Football2FIFA/blob/main/output_videos/README_results/003.gif" alt="player1" width=480 height=270/>
 <img src="https://github.com/unnamed-catalyst/Football2FIFA/blob/main/output_videos/README_results/003_output.gif" alt="player2" width=480 height=270/>
    <br>
    <em>Example output of the pipeline</em>
</p>

## How to Run
### Clone the repository
  ```
   git clone https://github.com/your-username/football2fifa.git
   cd football2fifa
  ```
### Install the required dependencies
```
pip install -r requirements.txt
```
### Prepare your input
- Place your match video inside the ```input_videos/``` folder. The code was only tested using 30 second clips from the [dataset mentioned above](https://www.kaggle.com/datasets/saberghaderi/-dfl-bundesliga-460-mp4-videos-in-30sec-csv).
- Place your trained YOLOv5 (or higher model) weights inside the ```models/``` folder. The trained weights that I used were too big to upload to GitHub.
- Make sure you change the name of the input file and model in ```main.py```.
- For the first run when using a new input video, make sure to change ```read_from_stub``` to ```False``` in line 21 of ```main.py```. This can be set to ```True``` for subsequent runs to save time.
- Change the name of the output file in ```main.py``` as desired.

### Run the main pipeline
```python main.py```

## Future Work
- Improve the custom-trained model from YOLOv5s to a heavier model (maybe YOLOv8).
- Add a function to detect more in-depth stats like team formations and defensive gaps using the players' positions.
- Add a function to convert all the extracted stats into a JSON or CSV file for analysis.
