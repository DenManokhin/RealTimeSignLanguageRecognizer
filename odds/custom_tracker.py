#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import keyboard


# In[ ]:


cap=cv2.VideoCapture(0)
# cv2.TrackerKCF_create()
# cv2.TrackerKCF_CUSTOM()
term_criteria=(cv2.TERM_CRITERIA_EPS| cv2.TERM_CRITERIA_COUNT,10,1)
_, frame=cap.read()
roi=frame[30:300,60:200]
hsv_roi=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
roi_hist=cv2.calcHist([hsv_roi],[0],None,[180],[0,180])
roi_hist=cv2.normalize(roi_hist, roi_hist,0,255,cv2.NORM_MINMAX)
while True:
    _,frame1=cap.read()
    frame=cv2.blur(frame1,(7,7))
    cv2.rectangle(frame,(200,300),(30,60),(255,255,255))
    if cv2.waitKey(1)&0xFF==ord('r'):        
        roi=frame[30:300,60:200]
        hsv_roi=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
        roi_hist=cv2.calcHist([hsv_roi],[0],None,[180],[0,180])
        roi_hist=cv2.normalize(roi_hist, roi_hist,0,255,cv2.NORM_MINMAX)
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
    _,track_window=cv2.meanShift(mask,(30,60,170,240),term_criteria)
    x,y,width,hight=track_window
    cv2.rectangle(frame1,(x,y),(x+width,y+hight),(255,0,0))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.imshow('FRAME',frame1)
    cv2.imshow('ROI',roi)
    cv2.imshow('mask',mask)
cap.release()
cv2.destroyAllWindows()


# In[3]:


cap.release()


# In[ ]:




