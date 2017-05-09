# Count the number of missing frames in a framemap extracted
# from the Kinect Data

import argparse

import pandas as pd

parser=argparse.ArgumentParser(description = "Count the number of missing frames")

parser.add_argument("--infile", type = str, required = True)
parser.add_argument("--fps", type = int, default = 30, required = False)
args = parser.parse_args()


framemap = pd.read_csv(args.infile, header = None ,
    names = ["frame", "time"])

framemap["reltime"] = framemap["time"] - framemap["time"][0] 
framemap["expectedframe"] = (framemap["reltime"] * args.fps  + 1).round()
framemap["delta"] = framemap["time"] - framemap["time"].shift(1)

biggestgap = max(framemap["delta"].dropna())
lastrow = framemap.iloc[-1]
droppedframes = lastrow["expectedframe"] - lastrow["frame"]


print ("%s, %f, %i" % (args.infile, biggestgap, droppedframes))

