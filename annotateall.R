library(stringr)


checkpath <- function(indir){
  if(!dir.exists(indir)){
    stop(paste(indir, "does not exist"))
  }
  
  if(is.na(str_match(indir, "/$"))){
    indir <- paste0(indir, "/")
  }
  
  return(indir)
}



options <- commandArgs(trailingOnly = TRUE)
if(length(options) != 4){
  stop("Expecting input video, output folders, face detection folder, object tracking folder ")
}

inputdir <- checkpath(options[1])
outputdir <- checkpath(options[2])
facedir <- checkpath(options[3])
objectdir <- checkpath(options[4])


videofiles <- list.files(inputdir, pattern = "P\\d+_video.mp4$")

for(v in videofiles){
  rootname <- str_extract(v, "(P\\d+)")
  outname <- paste0(rootname, "_bboxes.avi")
  facename <- paste0("pface", rootname, ".csv")
  objectname <- paste0(rootname, "_video.csv")
  
  
  cmd <- paste0("python showboxes.py ",
                inputdir, v, " ",
                outputdir, outname, " ",
                facedir, facename, " ",
                objectdir, objectname
  )
  print(cmd)
  system(cmd)
  
  
  
  
}

