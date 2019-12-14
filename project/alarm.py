# EECS4422 Project
# yucheng zhou 
# 213169636
# CSE: yucheng3

import cv2
import numpy as np
from skimage.measure import compare_ssim
import argparse
import imutils
import PIL
from PIL import Image

def Filter(frame):
	cur_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	lower = np.array([0, 50, 90])
	upper = np.array([20, 230, 230])   
	cur_frame = cv2.inRange(cur_frame, lower, upper)
	return cur_frame
	
def rotate(frame, angle):
    #grab the dimensions of the image and then determine the
    #center
    (h, w) = frame.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    #grab the rotation matrix (applying the negative of the
    #angle to rotate clockwise), then grab the sine and cosine
    #(i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    #compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    #adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    #perform the actual rotation and return the image
    return cv2.warpAffine(frame, M, (nW, nH))

def detect_similarity(cur_frame):
	frame_w, frame_h = cur_frame.shape[::-1] 
	threshold = 0.49
	detected = False
	maxs = []
	locs = []
	for temp in templates:
		temp_w, temp_h = temp.shape[::-1] 
		step = int((min(frame_h, frame_w) - min(temp_h, temp_w)) / 5) # min(h,w) guarantees template wont be bigger than the frame
		for i in range(1,6):
			temp_w = temp_w + step
			temp_h = temp_h + step
			resized = cv2.resize(temp, (temp_w, temp_h), interpolation = cv2.INTER_AREA) 
			res = cv2.matchTemplate(cur_frame,temp,cv2.TM_CCOEFF) 
			min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
			maxs.append(max_val)
			locs.append(max_loc)
	# select best matches		
	max_val = max(maxs)	
	print(maxs)
	i_max = maxs.index(max_val)
	max_loc = locs[i_max]
	if max_val > threshold: 
		detected = True
		top_left = max_loc
		bottom_right = (top_left[0] + temp_w, top_left[1] + temp_h)
		cv2.rectangle(cur_frame,top_left, bottom_right, 255, 2)
		cv2.imwrite('gesture-found.jpg',cur_frame)
	return detected

def kpmatch(cur_frame):
	result = False
	orb = cv2.ORB_create(nfeatures=1500)
	kp1, des1 = orb.detectAndCompute(cur_frame,None)
	kps = []
	dess = []
	threshold = 35
	for i in range (0,4):
		k, d = orb.detectAndCompute(templates[i],None)
		kps.append(k)
		dess.append(d)
		
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	for i in range (0,4):
		matches = bf.match(des1,dess[i])
		matches = sorted(matches, key = lambda x:x.distance)
		gkps = []
		for match in matches: 
			if match.distance < threshold: 
				gkps.append(match.distance)
		if len(kps) > 5: 
			result = True
	print(len(kps))
	return result

def Activate_alarm():
	print("EMERGENCY REPORTED!!!")
	return

templates = [] 
template1 = cv2.imread('image_std/masks/s1.jpg',0)	
template2 = cv2.imread('image_std/masks/s2.jpg',0)	
template3 = cv2.imread('image_std/masks/s3.jpg',0)	
template4 = cv2.imread('image_std/masks/s4.jpg',0)	
templates.append(template1)
templates.append(template2)
templates.append(template3)
templates.append(template4)	
inputstream = cv2.VideoCapture('input_lib/input1.mp4')
gesture_detected = False
while(inputstream.isOpened()):
	ret, frame = inputstream.read()
	# rotate
	rotated = rotate(frame, 90)
	frame = rotated
	detected_count = 0
	# get frame rate
	fps = inputstream.get(cv2.CAP_PROP_FPS)
	detected_need = fps * 2
	if ret == False:
		break
	frame = Filter(frame)
	cv2.imwrite( "current_frame.jpg", frame );
	gesture_detected = detect_similarity(frame)
	# gesture_detected = kpmatch(frame)
	if gesture_detected == True: 
		detected_count = detected_count + 1
		print("Detected!")
		if detected_count >= 30:
			Activate_alarm()
			detected_count = 0
	else: 
		detected_count = 0
	#print(detected_count)
		
		
		
		
