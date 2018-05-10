####################################################
# Modified by Roikoh                          #
# Original code: http://thecodacus.com/            #
# All right reserved to the respective owner       #
####################################################

# Import
import cv2
import numpy as np
import os 

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

# Create Patterns for face recognization
recognizer = cv2.face.LBPHFaceRecognizer_create()

assure_path_exists("resualt/")

recognizer.read('resualt/resualt.yml')
cascadePath = "haarcascade_frontalface_default.xml"

# Create classifier from prebuilt model
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize and start the video frame capture
cam = cv2.VideoCapture(0)

while True:
    ret, im =cam.read()
    # Convert to grayscale
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.2,5)

    for(x,y,w,h) in faces:
        cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,0,200), 4)
        Id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        if(Id == 1):
            Id = "Roikoh {0:.2f}%".format(round(100 - confidence, 2))

        cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (255,0,0), -1)
        cv2.putText(im, str(Id), (x,y-40), font, 1, (255,255,255), 3)

    cv2.imshow('im',im) 

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Stop the camera
cam.release()
cv2.destroyAllWindows()
