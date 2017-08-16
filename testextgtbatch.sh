#!/bin/bash

# Run an example using external ground truth
# With targetted training
./abc-classify.py --trackerfile ~/IDInteraction/paper/results/OpenFace/P01_front.openface --extgt ~/IDInteraction/paper/results/Groundtruth/P01_attention.csv --externaltrainingframes 200 --startframe 3000 --endframe 3500 --targetted --windowsize 20 --advancesize 10 --batchsize 50
