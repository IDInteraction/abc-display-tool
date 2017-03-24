""" Load depth data

"""
import pandas as pd
import numpy as np
import matplotlib.path as mpltPath


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

    depths=pd.read_csv(infile, header=None, names=["depth"],
        memory_map=True, engine="c")
    if width == 0 and height == 0:
        print "Getting width and height - for speed, prespecify"
        (width, height) = getDepthDimensions(infile)
    depths["x"] =  range(width)  * height 
    depths["y"] = [item for item in range(height) for i in range(width)]

    return depths


def filterFrame(inframe, mindepth = None, maxdepth = None, polygon = None,
        recodenulls = False, recodevalue = np.nan):
    """Filter a frame by depth and / or masking polygon"""

    filterframe = inframe
    #TODO could do these in one go if min and max defined
    if mindepth is not None:
        filterframe.loc[ filterframe["depth"] <= mindepth ,"depth"] = np.nan

    if maxdepth is not None:
        filterframe.loc[ filterframe["depth"] >= maxdepth,"depth"] = np.nan

    if polygon is not None:
        if recodenulls == True:
            sys.exit("Still need to implement recodenulls with polygon filtering")
            
        # We filter the points in the bounding box in two stages;
        # first a crude box based on the maximum extent of the polygon
        # then using matplotlib to test the remaining points properly
        # TODO Don't do part 2 if rotation is 0
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]

        minx=min(xs)
        maxx=max(xs)
        miny=min(ys)
        maxy=max(ys)

        filterframe = filterframe[(filterframe["x"] <= maxx) & 
                (filterframe["x"] >= minx) &
                (filterframe["y"] <= maxy) & 
                (filterframe["y"] >= miny)]

        path = mpltPath.Path(polygon)
        pointarr =[filterframe["x"].values, filterframe["y"].values]
        tpointarr = map(list, zip(*pointarr))
        framefilter = path.contains_points(tpointarr)
        filterframe = filterframe[framefilter]

    if recodenulls == False:
        filterframe = filterframe[pd.notnull(filterframe["depth"])]
    else:
        filterframe.loc[pd.isnull(filterframe["depth"]),"depth"] = recodevalue

    return filterframe
