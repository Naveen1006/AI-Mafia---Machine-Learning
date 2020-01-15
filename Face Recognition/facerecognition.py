# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import numpy as np
# colored Image
# Black and White (gray scale)
cap=cv2.VideoCapture(0)

#Face Detection
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

face_data=[]
dataset_path="D:/"
file_name=input("Enter the name of person:")

while True:
    ret,frame=cap.read()
    
    if ret== False:
        continue
    
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    faces= face_cascade.detectMultiScale(gray_frame,1.3,5)
    if len(faces)==0:
        continue
    
    faces = sorted(faces, key=lambda f:f[2]*f[3])
    
    for face in faces[-1:]:
        #draw bounding box or the rectangle
        
        x,y,w,h=face
        cv2.rectangle(gray_frame, (x,y),(x+w,y+h), (0,255,255),2)
       #extract or crop out the required face:region of interest
        offset=10
        face_section=gray_frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section=cv2.resize(face_section,(100,100))
        face_data.append(face_section)
        print(len(face_section))
        
    #cv2.imshow("Frame",frame)
    cv2.imshow("gray_frame",gray_frame)
    key_pressed = cv2.waitKey(2) & 0xFF
    if key_pressed==ord('q'):
        break
 
#convert face data list to numpy array
face_data=np.asarray(face_data)
face_data=face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)
np.save(dataset_path+file_name+'.npy',face_data)
print("Data Saved Successfully")
        
cap.release()
cv2.destroyAllWindows()   