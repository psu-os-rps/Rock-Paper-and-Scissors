import cv2
from datetime import datetime
import numpy as np
from sklearn.metrics import pairwise
import config
version = cv2.__version__

def count(threshold, contour):
    hull = cv2.convexHull(contour)
    vertical = hull[:, :, 1]
    horizon = hull[:, :, 0]
    min_vertical = tuple(hull[vertical.argmin()][0])
    max_vertical = tuple(hull[vertical.argmax()][0])
    min_horizon = tuple(hull[horizon.argmin()][0])
    max_horizon = tuple(hull[horizon.argmax()][0])
    
    center_x = (max_horizon[0] + min_horizon[0]) // 2
    center_y = (max_vertical[1] + min_vertical[1]) // 2
    circle_rad = (pairwise.euclidean_distances([(center_x,center_y)], 
    Y = [min_horizon,max_horizon,min_vertical,max_vertical])[0]).max() * config.rate
    circle_rad = int(circle_rad)
    detector_circle = np.zeros(threshold.shape[:2], dtype="uint8")
    cv2.circle(detector_circle,(center_x,center_y),circle_rad,config.RGB_INT_MAX,config.circle_thickness)
    detector_circle = cv2.bitwise_and(threshold,threshold,mask=detector_circle)
    
    #Opencv version causes the function return different values
    if version.startswith("4."):
        contours, _ = cv2.findContours(detector_circle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        _,contours, _ = cv2.findContours(detector_circle.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    numbers = 0

    for o in contours:
        (_,button,_,right) = cv2.boundingRect(o)
        if (center_y + (center_y * config.circle_rate)) > (button + right) and (2*circle_rad*np.pi*config.circle_rate) > o.shape[0]:
            numbers += 1 

    return numbers

def main():
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    if cap is None or not cap.isOpened():
        print("Default camera is not working, check connection of camera")
        return 
    background = None
    frames = 0

    while cap.isOpened():
        _, image = cap.read()
        background_copy = image.copy()
        detector_area = image[config.detector_u:config.detector_b,config.detector_r:config.detector_l]
        gray = cv2.cvtColor(detector_area, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(config.Blur_value,config.Blur_value),0)

        if frames < config.processing_frame:
            
            if background is None:
                background = blur.astype('float')
            else:
                cv2.accumulateWeighted(blur, background, config.accumulated_weight)
            
            message = "Calculating the background processing, wait for: "+str(config.processing_frame - frames)
            cv2.putText(image,message,(config.message_x, config.message_y), cv2.FONT_HERSHEY_DUPLEX, config.font_scale, config.text_color, config.thickness)
        
        else:
            different = cv2.absdiff(background.astype('uint8'), blur)
            #thresholded = cv2.adaptiveThreshold(different,config.RGB_INT_MAX,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,config.cv2adaptive_block,config.cv2adaptive_param)
            _,thresholded = cv2.threshold(different,config.threshold_min,config.RGB_INT_MAX,cv2.THRESH_BINARY)
            
            #Use erode to delete small black threshold then dilate write threshold then.
            kernel = np.ones((config.Blur_value,config.Blur_value),np.uint8)
            thresholded = cv2.erode(thresholded,kernel,iterations = config.erodtime)
            thresholded = cv2.dilate(thresholded,kernel,iterations = config.dilatetime)
            
            #Opencv version causes the function return different values
            if version.startswith("4."):
                contours, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            else:
                _,contours,_ = cv2.findContours(thresholded.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) == 0:
                message_ready = "Background information finished, try put hand in"
                cv2.putText(image,message_ready,(config.message_x, config.message_y), cv2.FONT_HERSHEY_DUPLEX, config.font_scale,config.text_color, config.thickness)
            else:
                hand_counter = max(contours,key=cv2.contourArea)
                 # Background Removal
                fg_mask =  thresholded.astype(np.float32) / config.RGB_FLT_MAX
                fg_mask = np.stack([fg_mask, ] * 3, axis=2)
                fg_only = fg_mask * detector_area.astype(np.float32)

                counters = count(thresholded, hand_counter)
                message_result = "Number of Fingers: "+str(counters)
                cv2.putText(image, message_result, (config.message_x, config.message_y), cv2.FONT_HERSHEY_DUPLEX, config.font_scale,config.text_color, config.thickness)
                # Display the Video after Background remove in real time
                cv2.imshow('Background Removel', fg_only.astype(np.uint8))
                cv2.imshow('Threshholded', thresholded)

        frames += 1
        cv2.rectangle(image,(config.detector_l,config.detector_u),(config.detector_r,config.detector_b),config.rectangle_color,config.rectangle_thickness)
        message_time = str(datetime.now())
        cv2.putText(image, message_time, (config.date_x, config.date_y), cv2.FONT_HERSHEY_DUPLEX, config.font_scale,config.text_color, config.thickness)
        cv2.imshow('Rock-Paper-and-Scissors',image)
        
        
        order = cv2.waitKey(10)
        if order == 27:
            break

        if order == ord('q'):
            break

        if order == ord('r'):
            background = None
            frames = 0

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()