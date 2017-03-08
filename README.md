# abc-display-tool
Classify behaviours and superimpose the results on video

## abc-classify.py
This program takes an input video, and tracking data derived from CppMT (https://github.com/gnebehay/CppMT)
or OpenFace (https://github.com/TadasBaltrusaitis/OpenFace).  The user classifies randomly chosen frames from the period of
interest by selecting a numeric value corresponding to each behaviour that is being encoded (e.g. 1 for looking at an iPad,
0 for looking at a TV).   Once an appropriate number of frames have been classified, the code attempts to predict the behaviours
being observed in the remaining frames in the video.

## showboxes.py
Given an input video, outputvideo and 1+ bounding-box/prediction files, this program will superimpose (a) box(es) with the
coordinates given in the bounding-box/prediction file(s), whose colour depends on the predicted state for each frame included in
the file.  It is used to check the predictions from, e.g. abc-classify.py

