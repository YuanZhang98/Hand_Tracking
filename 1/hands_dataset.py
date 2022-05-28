import cv2
import sys
import logging as log
import datetime as dt
from time import sleep

cascPath = "haarcascades_hands.xml"
handCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log', level=log.INFO)

video_cap = cv2.VideoCapture(0)
anterior = 0


count = 0

while True:
    if not video_cap.isOpened():
        print('Cant load cam')
        sleep(5)
        pass

    ret, frame = video_cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    hands = handCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=2,
        minSize=(20, 20)
    )

    for (x, y, w, h) in hands:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        count += 1
        cv2.imwrite("dataset/User." + str(count) + ".jpg", gray[y:y+h, x:x+w])

        cv2.imshow('image', frame)
    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    # Collect 30 sample images of each face
    elif count >= 50:
        break


# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")


# When everything is done, release the capture
video_cap.release()
cv2.destroyAllWindows()