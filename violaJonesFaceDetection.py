import cv2 
import os
from os.path import isfile, join
import numpy as np


vidcap = cv2.VideoCapture('EVMvideo.avi')
def facedetect(dirpath,fps):
    pathOut = 'FaceDetectVideo.avi'
    frame_array = []
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    files = [f for f in os.listdir(dirpath) if isfile(join(dirpath, f))]
    for i in range(len(files)):
        files[i] = int(files[i][5:-4])
    files.sort()
    print(files)
    for i in range(len(files)):
        filename=pathIn +'\\'+'image'+str(files[i])+'.jpg'
        #reading each files
        print(filename)
        img = cv2.imread(filename)
     
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:  
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  
        height, width, layers = img.shape
        size = (width,height)
        
        #inserting the frames into an image array
        frame_array.append(img)
    directory = r'D:\Majour _Project'
    os.chdir(directory)
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
    # writing to a image array
        out.write(frame_array[i])
    out.release()
    
def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    directory = r'D:\Majour _Project\temp1'
    os.chdir(directory)
    if hasFrames:
        cv2.imwrite("image"+str(count)+".jpg", image)     # save frame as JPG file
    return hasFrames


sec = 0
frameRate = 0.03
count=1
success = getFrame(sec)
while success:
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec)

pathIn = r'D:\Majour _Project\temp1'
fps = vidcap.get(cv2.CAP_PROP_FPS)
facedetect(pathIn,fps)
