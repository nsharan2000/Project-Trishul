# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 12:12:19 2021

@author: sharan nagarajan
"""

import cv2

#initialize background subtractor object
foog=cv2.createBackgroundSubtractorMOG2(detectShadows = True, varThreshold = 50, history = 2000)

#noise fileter threshold
thresh = 1000

video = cv2.VideoCapture(0)

reference_frame = None # first_frame is the reference frame
while True:
    check,frame = video.read()
    
    frame = foog.apply(frame)
    
    
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(21,21),0) #21,21 is the blurring kernel size
    if reference_frame is None:
        reference_frame = gray
        continue
    delta_frame = cv2.absdiff(reference_frame,gray)
    threshold_frame = cv2.threshold(delta_frame,50,255,cv2.THRESH_BINARY)[1]
    threshold_frame = cv2.dilate(threshold_frame,None,iterations=4)
    
    cntr,_ =cv2.findContours(threshold_frame.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #WITHOUT SPECIFYING CONTOUR WIDTH EVEN A SMALL NOISE WILL BE AFFECTED AS THE MOTION AREA WHICH CAN CAUSE FALSE POSITIVES OF INTRUDERS
    
    for contour in cntr:
        if cv2.contourArea(contour)<1000:
            continue
        (x,y,w,h)=cv2.boundingRect(contour)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        # frame is the image in which we are choosing as image
        # x and y are dimensions of the bounding recangle
        # 0,255,0 is the colour of the box - pure green
        # 3 is the size of the rectangle that is used to draw the rectangle
        
    cv2.imshow("intruder detector", frame)
    key = cv2.waitKey(1)
    if key ==ord("q"):
        exit()
    
        