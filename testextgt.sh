#!/bin/bash

# Run an example using external ground truth
./abc-classify.py --trackerfile ~/IDInteraction/paper/results/OpenFace/P01_front.openface --extgt ~/IDInteraction/paper/results/Groundtruth/P01_attention.csv --externaltrainingframes 200 --startframe 3000 --endframe 6000

