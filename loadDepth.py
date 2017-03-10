""" Load depth data

"""
import pandas as pd


def getDepthDimensions(depthfile):
    """ Guess the dimensions of a (depth) file from the number of rows in it
    """
    depths = pd.read_csv(depthfile, header=None, names=["depth"])
    numrows = len(depths)
    if numrows == 512 * 424:
        width = 512
        height = 424
    elif numrows == 1920 * 1080:
        width = 1920
        height = 1080
    else:
        raise Exception("Cannot guess frame dimensions")

    return (width, height)




def loadDepth(infile, width=0, height=0):
    """Load a depth file"""

    depths=pd.read_csv(infile, header=None, names=["depth"])
    if width == 0 and height == 0:
        (width, height) = getDepthDimensions(infile)
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


