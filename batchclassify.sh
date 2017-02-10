#!/bin/bash

# Run abc-classify in non-interactive mode
# 
# $1 participantCode
# $2 trainingframes

videodir=/media/sf_spot_the_difference/video/
trackerdir=/media/sf_spot_the_difference/openface/
extgtdir=/home/zzalsdme/IDInteraction/spot_the_difference/part1/

pcs=_front
extgts=checkGTwebcam2.csv 

startframe=`head -1 ${extgtdir}${1}${extgts} |awk -F"," '{print $1}'`

endframe=`tail -1 ${extgtdir}${1}${extgts} |awk -F"," '{print $1}'`
echo $startframe $endframe


./abc-classify.py --videofile ${videodir}${1}${pcs}.mp4 --trackerfile ${trackerdir}${1}${pcs}.openface --startframe $startframe --endframe $endframe --extgt ${extgtdir}${1}${extgts} --externaltrainingframes $2 --useexternalgt --participantcode ${1}shuffle --summaryfile summaryresults.csv


./abc-classify.py --videofile ${videodir}${1}${pcs}.mp4 --trackerfile ${trackerdir}${1}${pcs}.openface --startframe $startframe --endframe $endframe --extgt ${extgtdir}${1}${extgts} --externaltrainingframes $2 --useexternalgt --participantcode ${1}noshuffle --summaryfile summaryresults.csv --noshuffle


