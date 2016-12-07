
# Copyright (c) 2016 The University of Manchester, UK.
#
# Licenced under LGPL version 2.1. See LICENCE for details.
#
# The IDInteraction Attention Classification was developed in the IDInteraction project, funded by the Engineering and Physical Sciences Research Council, UK through grant agreement number EP/M017133/1.
#
# Author: David Mawdsley

# Generate predictions given a set of tracking (ground truth), bounding box and 
# a training time


cloptions <- commandArgs(trailingOnly = TRUE)



library(rpart)
library(REEMtree)
library(formula.tools)
library(IDInteraction)
library(party)
library(sqldf)


print(cloptions)

participant = cloptions[1]
annoteloc = cloptions[2]
trackingloc = cloptions[3]
trainingtime = as.numeric(cloptions[4])
predmethod = cloptions[5]
outfile = cloptions[6]

if(predmethod != "rpart"){
  stop("Only recursive partioning trees implemented so far")
}


table1formula <- attentionName ~ boxHeight +  boxRotation + boxArea + boxWidth + widthHeightRatio + boxYcoordRel

allparticipants <- loadExperimentData(participant,
                                      trackingLoc = trackingloc,
                                      annoteLoc = annoteloc)


allresults <- NULL
allpredictions <- NULL
for(p in participant){
  thisparticipant <- subset(allparticipants, allparticipants$participantCode == p)

  for(tt in trainingtime){


    thispred <- getPartitionPredictions(thisparticipant, tt, table1formula)

    accuracy <- getAccuracy(getConfusionMatrixPreds(thispred))

    modelpredictions  <- cbind(
      traintime = tt,
      thisparticipant[!flagtraining(thisparticipant, tt),
                      c("participantCode", "frame","timestampms")],
      thispred)

    modelpredictions$numpredclass <- as.numeric(modelpredictions$predclass)

   write.csv(modelpredictions[,c("traintime","participantCode",
                                "frame", "timestampms",
                                "predclass", "numpredclass")],
             file = outfile)



  }

  print(p)
}


