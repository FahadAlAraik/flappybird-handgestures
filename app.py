import cv2
import mediapipe as mp
import time
import numpy as np
import HandTrackingModule as htm
import math
import pyautogui



######################################

wCam,hCam = 640, 480

######################################
pTime = 0
detector = htm.handDetector(min_detection_confidence=0.7)


cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
while True:
    success,img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img,draw=False)
    # we need lmList[4] -> thumb lmList[8] -> index
    if len(lmList) != 0:
        x1,y1 = lmList[4][1],lmList[4][2]
        x2,y2 = lmList[8][1], lmList[8][2]
        cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)
        cv2.circle(img,(x2,y2),15,(255,0,255),cv2.FILLED)
        cv2.line(img,(x1,y1),(x2,y2),(255,0,255),3)
        center_x_line,center_y_line = (x1+x2)//2,(y1+y2)//2
        cv2.circle(img,(center_x_line,center_y_line),15,(255,0,255),cv2.FILLED)
        length = math.hypot(x2-x1,y2-y1)
        # max = 350
        # min = 50

        if length<50:
            cv2.circle(img,(center_x_line,center_y_line),15,(0,255,0),cv2.FILLED)
            pyautogui.press('space')


    cTime = time.time()
    fps = 1 / (cTime-pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_COMPLEX,3,(255,0,255),2)
    cv2.imshow("Img", img)


    cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()