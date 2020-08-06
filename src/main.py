import cv2
import numpy as np
from sklearn.metrics import pairwise
import config
version = cv2.__version__

background = None

#Function calculate the weighted sum of the input image src and the accumulator dst so that dst becomes a running average of a frame sequence
def calc_accum_avg(frame,accumulated_weight):
    
    global background
    
    if background is None:
        background = frame.copy().astype('float')
        return None
    
    cv2.accumulateWeighted(frame,background,accumulated_weight)

#Function make the program know the hand location and find the contours of the hand
def segment(frame,threshold_min):
    diff = cv2.absdiff(background.astype('uint8'),frame)
    _,thresholded = cv2.threshold(diff,threshold_min,255,cv2.THRESH_BINARY)
    
    
    #Use erode to delete small black threshold then dilate write threshold then.
    kernel = np.ones((5,5),np.uint8)
    thresholded = cv2.erode(thresholded,kernel,iterations = 1)
    thresholded = cv2.dilate(thresholded,kernel,iterations = 2)
    
    #Opencv version causes the function return different values
    if version.startswith("4."):
        contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        image,contours,hierarchy = cv2.findContours(thresholded.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


    if len(contours) == 0:
        return None
    else:
        hand_segment = max(contours,key=cv2.contourArea)
        return (thresholded,hand_segment)

#calculated the external contour of the hand. Counting Fingers with a Convex Hull
def count_fingers(thresholded,hand_segment):
    
    conv_hull =cv2.convexHull(hand_segment)
    
    top    = tuple(conv_hull[conv_hull[:, :, 1].argmin()][0])
    bottom = tuple(conv_hull[conv_hull[:, :, 1].argmax()][0])
    left   = tuple(conv_hull[conv_hull[:, :, 0].argmin()][0])
    right  = tuple(conv_hull[conv_hull[:, :, 0].argmax()][0])
    
    cX = (left[0] + right[0]) // 2
    cY = (top[1] + bottom[1]) // 2
    
    distance = pairwise.euclidean_distances([(cX, cY)], Y=[left, right, top, bottom])[0]
    max_distance = distance.max()
    radius = int(config.rate*max_distance)
    circumfrence = (2*np.pi*radius)
    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
    cv2.circle(circular_roi,(cX,cY),radius,255,10)
    circular_roi = cv2.bitwise_and(thresholded,thresholded,mask=circular_roi)

    #Opencv version causes the function return different values
    if version.startswith("4."):
        contours, hierarchy = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        image,contours,hierarchy = cv2.findContours(circular_roi.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    count = 0
    
    for cnt in contours:
        (x,y,w,h) = cv2.boundingRect(cnt)
        out_of_wrist = (cY + (cY*0.25)) > (y+h)
        limit_points = ((circumfrence*0.25) > cnt.shape[0])
        if out_of_wrist and limit_points:
            count += 1
            
    return count


cam = cv2.VideoCapture(0)
num_frames = 0

# keep looping, until interrupted
while True:
    ret, frame = cam.read()
    frame_copy = frame.copy()
    roi = frame[config.roi_top:config.roi_bottom,config.roi_right:config.roi_left]
    gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(7,7),0)
    
    if num_frames < 60:
        calc_accum_avg(gray,config.accumulated_weight)
        
        if num_frames <= 59:
            cv2.putText(frame_copy,'WAIT. GETTING BACKGROUND',(200,300),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            cv2.imshow('Finger Count',frame_copy)
    else:
        hand = segment(gray, config.threshold_min)
        if hand is not None:
            thresholded , hand_segment = hand
            
            # Background Removal
            fg_mask =  thresholded.astype(np.float32) / 255.0
            fg_mask = np.stack([fg_mask, ] * 3, axis=2)
            fg_only = fg_mask * roi.astype(np.float32)
            # Display the Video after Background remove in real time
            cv2.imshow('Foreground', fg_only.astype(np.uint8))

            cv2.drawContours(frame_copy,[hand_segment+(config.roi_right,config.roi_top)],-1,(255,0,0),5)
            fingers = count_fingers(thresholded,hand_segment)
            cv2.putText(frame_copy,str(fingers),(70,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            cv2.imshow('Thresholded',thresholded)
   
    cv2.rectangle(frame_copy,(config.roi_left,config.roi_top),(config.roi_right,config.roi_bottom),(0,0,255),5)
    num_frames += 1
    cv2.imshow('Finger Count',frame_copy)
    
    k = cv2.waitKey(1) & 0xFF
    
    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()
    