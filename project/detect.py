from matplotlib import pyplot as plt
import imutils
import cv2
import numpy as np

templates = [] 
template1 = cv2.imread('image_std/originals/1.jpg',0)	
template2 = cv2.imread('image_std/originals/2.jpg',0)	
template3 = cv2.imread('image_std/originals/3.jpg',0)	
template4 = cv2.imread('image_std/originals/4.jpg',0)	
templates.append(template1)
templates.append(template2)
templates.append(template3)
templates.append(template4)
cur_frame = cv2.imread('cur_frame.png',0)

def detect_similarity(cur_frame):
	frame_w, frame_h = cur_frame.shape[::-1] 
	threshold = 0.49
	detected = False
	maxs = []
	locs = []
	for temp in templates[:4]:
		temp_w, temp_h = temp.shape[::-1] 
		step = int((min(frame_h, frame_w) - min(temp_h, temp_w)) / 5) # min(h,w) guarantees template wont be bigger than the frame
		for i in range(1,5):
			temp_w = temp_w + step
			temp_h = temp_h + step
			resized = cv2.resize(temp, (temp_w, temp_h), interpolation = cv2.INTER_AREA) 
			res = cv2.matchTemplate(cur_frame,temp,cv2.TM_CCOEFF) 
			min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
			maxs.append(max_val)
			locs.append(max_loc)
	# select best matches		
	max_val = max(maxs)	
	i_max = maxs.index(max_val)
	max_loc = locs[i_max]
	if max_val > threshold: 
		detected = True
		top_left = max_loc
		bottom_right = (top_left[0] + temp_w, top_left[1] + temp_h)
		cv2.rectangle(cur_frame,top_left, bottom_right, 255, 2)
		cv2.imwrite('gesture-found.jpg',cur_frame)
		# loc = np.where( res >= threshold)
		# for pt in zip(*loc[::-1]): 
			# cv2.rectangle(cur_frame, pt, (pt[0] + temp_w, pt[1] + temp_h), (0,0,255), 3) 
			# cv2.imwrite('gesture-found.jpg',cur_frame)

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
		# img = cv2.drawMatches(templates[i],kps[i],cur_frame,kp1,matches[:4], None, flags=2)
		# plt.imshow(img),plt.show()		
		
#kpmatch(cur_frame)
detect_similarity(cur_frame)