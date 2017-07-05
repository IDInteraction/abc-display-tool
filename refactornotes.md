## abc-classify refactoring

Separate out interactive part - e.g. user classifying frames by hand.  This should call the showboxes.py code too, since the playback
via OpenCV isn't great

The way the results are returned is a mess

Need to add a flexible interface to return other metrics

Main loop is over complex - handles interactive and non interactive in the same place

Random forests - whether to use or not. How to integrate into code?

Will want to include Rob's idea of focussing on the parts of the video where the classification hasn't worked very well


