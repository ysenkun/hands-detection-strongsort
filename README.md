# Track Hands Detection with StrongSORT

This repository can track hands.  
track.py uses MediaPipe Hands.  

※Also, [here](https://github.com/ysenkun/faces-detection-strongsort) is a repository for tracking faces

### :raising_hand: Reference:
https://google.github.io/mediapipe/solutions/hands.html  
https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet


## Seting Up Environment

```bash
$ git clone https://github.com/ysenkun/hands-detection-strongsort.git
```

```bash
$ pip3 install -r requirements.txt
```

## Run
Enter the video path and run it.
```bash
$ python3 track.py --source vid.mp4 # video path
```

![hand](https://user-images.githubusercontent.com/82140392/180654842-fb6bde4e-d152-40d8-8d4c-c674da446108.gif)
