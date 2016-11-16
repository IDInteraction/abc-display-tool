# From http://stackoverflow.com/questions/14440400/creating-a-video-using-opencv-2-4-0-in-python

import cv2

img1 = cv2.imread('1.jpg')
img2 = cv2.imread('2.jpg')
img3 = cv2.imread('3.jpg')

height , width , layers =  img1.shape
fourcc = cv2.cv.CV_FOURCC('D', 'I', 'V', 'X')
video = cv2.VideoWriter('video.avi',fourcc,20,(width,height))

video.write(img1)
video.write(img2)
video.write(img3)

cv2.destroyAllWindows()
video.release()
