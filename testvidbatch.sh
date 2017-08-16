#!/bin/bash

# Run an example using a video file to interactively classify frames
# With batches
./abc-classify.py --trackerfile ~/IDInteraction/paper/results/OpenFace/P01_front.openface --videofile ~/IDInteractionSmallSet/webcamvideo/P01_front.mp4 --externaltrainingframes 40 --startframe 3000 --targetted --endframe 3050 --windowsize 20 --advancesize 10 --batchsize 10 

