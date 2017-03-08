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


## Acknowledgements

The IDInteraction Processing Pipelines were developed in the IDInteraction project, funded by the Engineering and Physical Sciences Research Council, UK through grant agreement number [EP/M017133/1][gow].

## Licence

Copyright (c) 2017 The University of Manchester, UK.

Licenced under LGPL version 2.1. See LICENCE for details.

[gow]: http://gow.epsrc.ac.uk/NGBOViewGrant.aspx?GrantRef=EP/M017133/1
