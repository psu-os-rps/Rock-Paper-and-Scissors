import cv2
import numpy as np
from sklearn.metrics import pairwise
import config
version = cv2.__version__






def main():
    cap = cv2.VideoCapture(0)
    if cap is None or not cap.isOpened():
        print("Default camera is not working, check connection of camera")
        return 
    background = None
    frames = 0

    while cap.isOpened():
        _, image = cap.read()
        background_temp = image.copy().astype('float')
        detector = image[config.detector_u:config.detector_b,config.detector_r:config.detector_l]
        gray = cv2.cvtColor(detector, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(config.Blur_value,config.Blur_value),0)

        if frames < config.processing_frame:
            """
            global background
            if background is None:
                background = background_temp
            else:
                cv2.accumulateWeighted(blur, background, config.accumulated_weight)
            """
            message = "Calculating the background processing, wait for: "+str(config.processing_frame - frames)
            cv2.putText(image,message,(config.detector_b-200, config.detector_r), config.processing_frame, config.font_scale, config.thickness)
        
        frames += 1
        cv2.imshow('frame',image)
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()