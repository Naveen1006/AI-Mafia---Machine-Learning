# -*- coding: utf-8 -*-
import cv2

import numpy as np

cap=cv2.VideoCapture(0)

face_cascade=cv2.CascadeClassifier("C:/Users/BHAWNAVEEN/face recognition snapchat filter/haarcascade_frontalface_alt.xml")
f3 = cv2.CascadeClassifier("C:/Users/BHAWNAVEEN/face recognition snapchat filter/Nose18x15.xml")
eyes_cascade = cv2.CascadeClassifier("C:/Users/BHAWNAVEEN/face recognition snapchat filter/haarcascade_eye_tree_eyeglasses.xml")
imgg=cv2.imread('C:/Users/BHAWNAVEEN/Downloads/Week 4 - Face Recognition Assignment/Train/mustache.jpg')

img2=cv2.imread('C:/Users/BHAWNAVEEN/Downloads/Week 4 - Face Recognition Assignment/Train/glasses.jpg')


while True:

    ret,frame=cap.read()

    if(ret==False):

        continue

    #frame=cv2.resize(frame,(700,400))

    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces=face_cascade.detectMultiScale(gray_frame,1.3,5)

    if(len(faces)==0):

       continue

    faces=sorted(faces,key=lambda x:x[2]*x[3])

    #print(faces)



    for face in faces[-1:]:

        x,y,w,h=face

        a=face

        #cv2.rectangle(gray_frame,(x,y),(x+w,y+h),(10,15,205),-1)

    roi_gray = gray_frame[y:y+h, x:x+w]

    roi_color = frame[y:y+h, x:x+w]

    

    eyes = eyes_cascade.detectMultiScale(gray_frame)

    eyes=sorted(eyes,key=lambda x:x[0])

    #print(eyes)

    for eye in eyes[:1]:

                ex,ey,ew,eh=eye

                #cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

                X1=ex-20

                Y1=ey-20

                X2=X1+2*ew+40

                Y2=ey+eh+20

                img2=cv2.resize(img2,(X2-X1,Y2-Y1),interpolation=cv2.INTER_AREA)

                frame[Y1:Y2,X1:X2]=cv2.bitwise_and(frame[Y1:Y2,X1:X2],img2,mask=None)



    nose = f3.detectMultiScale(gray_frame)

    #print(eyes)

    for (ex,ey,ew,eh) in nose[:1]:

        print(ex,ey,ew,eh)

        #cv2.rectangle(gray_frame,(ex-15,ey+55),(ex+ew+15,ey+eh),(0,255,0),2)

        x1,x2,y1,y2=ex,ey,ex+ew,ey+eh

        if(ex>15):

            x1=ex-15

        if(ey>23):    

            y1=ey+23

        if(ex+ew+15<730):

            x2=ex+ew+15

        if(ey+eh+20<730):

            y2=ey+eh+20

        imgg=cv2.resize(imgg,(x2-x1,y2-y1),interpolation=cv2.INTER_AREA)

        #frame[y1:y2,x1:x2]=cv2.addWeighted(frame[y1:y2,x1:x2],-5,imgg,-10,0,frame)

        frame[y1:y2,x1:x2]= cv2.bitwise_and(frame[y1:y2,x1:x2],imgg , mask=None)

    cv2.imshow("gray_frame",frame)

    key_pressed=cv2.waitKey(1)

    if(key_pressed==ord('q')):

        break

    #break

cap.release()

cv2.destroyAllWindows()

print("hello")
