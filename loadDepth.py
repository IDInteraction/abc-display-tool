""" Load depth data

"""
import pandas as pd



def loadDepth(infile, width=0, height=0):
    
    depths=pd.read_csv(infile, header=None, names=["depth"])
    if width == 0 and height == 0: # Guess 
        if len(depths) == 512 * 424:
            width = 512
            height = 424
        elif len(depths) == 1920 * 1080:
            width = 1920
            height = 1080
        else:
            raise Exception("Cannot guess frame dimensions")
    depths["x"] =  range(width)  * height 
    depths["y"] = [item for item in range(height) for i in range(width)]

    return depths



#parser = argparse.ArgumentParser(description="Load depth data")
#parser.add_argument("--depthfile", dest="depthfile", type = str, required = True);
#
#args = parser.parse_args()
#
#
#depthData = loadDepth(args.depthfile)
#

