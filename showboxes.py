#!/usr/bin/python

import cv2
import os
import csv
import sys

WINDOW_NAME = 'Playback'

video = cv2.VideoCapture(sys.argv[1])

fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
ow = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
oh = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.cv.CV_FOURCC('X','V','I','D')
videoout = cv2.VideoWriter(sys.argv[2], fourcc, fps,  (ow, oh))
cv2.namedWindow(WINDOW_NAME)

frame = 0
got, img = video.read()

while got:
    print frame


   # cv2.imshow(WINDOW_NAME, img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    got, img = video.read()
    frame = frame + 1
    videoout.write(img)
    
video.release()
videoout.release()
cv2.destroyAllWindows()

