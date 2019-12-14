# EECS4422 Assignment 1
# yucheng zhou 
# 213169636
# CSE: yucheng3

import cv2
import numpy as np

def blackandwhite(img_in):   
	# the result image will be stored as img_out.png at the current dir  
	# color the dark objects
	img_out = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
	lower = np.array([0, 40, 90])
	upper = np.array([20, 255, 255])   
	mask = cv2.inRange(img_out, lower, upper)
	dim = (540, 960)
	img_out = cv2.resize(mask, dim, interpolation = cv2.INTER_AREA)
	
	return img_out
    
s1 = cv2.imread('image_std/originals/1.jpg',1)
cv2.imwrite('image_std/masks/s1.jpg',blackandwhite(s1))
s2 = cv2.imread('image_std/originals/2.jpg',1)
cv2.imwrite('image_std/masks/s2.jpg',blackandwhite(s2))
s3 = cv2.imread('image_std/originals/3.jpg',1)
cv2.imwrite('image_std/masks/s3.jpg',blackandwhite(s3))
s4 = cv2.imread('image_std/originals/4.jpg',1)
cv2.imwrite('image_std/masks/s4.jpg',blackandwhite(s4))
