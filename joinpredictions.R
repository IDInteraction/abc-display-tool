library(IDInteraction)


options <- commandArgs(trailingOnly = TRUE)
if(length(options) != 3){
  print("Join model predictions to a tracking file, ready for plotting")
  stop("Expecting tracking file, prediction file, output file ")
}



intrack <- options[1]

inpred <- options[2]

trackpreds <- addPredictionsToTracking(intrack, inpred)

write.csv(trackpreds, file = options[3], row.names = FALSE)

