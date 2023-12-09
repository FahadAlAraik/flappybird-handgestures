import cv2
import mediapipe as mp
import time


class handDetector():

    def __init__(self,static_image_mode=False,max_num_hands=2,model_complexity=1,min_detection_confidence=0.5,min_tracking_confidence=0.5):
        self.mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.max_num_hands,self.model_complexity,self.min_detection_confidence,self.min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self,img,draw=True):

        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # by default, imgs in opencv are BGR and mediapipe uses RGB
        results = self.hands.process(imgRGB)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self,img,landmark_id=0,draw=True):
        landmark_list = []
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # by default, imgs in opencv are BGR and mediapipe uses RGB
        results = self.hands.process(imgRGB)
        if results.multi_hand_landmarks:
            wanted_landmark = results.multi_hand_landmarks[landmark_id]
            for handLms in results.multi_hand_landmarks:
                for id, landmark in enumerate(wanted_landmark.landmark):
                    height,width,channels = img.shape
                    cx, cy = int(landmark.x*width), int(landmark.y*height)
                    landmark_list.append([id,cx,cy])
                    if draw:
                        cv2.circle(img,(cx,cy),7,(255,0,0),cv2.FILLED)

        return landmark_list




    

def main():

    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()


    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img = detector.findHands(img)
 
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])


        cTime = time.time()
        fps = 1 / (cTime-pTime)
        pTime = cTime
        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_COMPLEX,3,(255,0,255),2)

        cv2.imshow("Image", img)
        
        # Wait for 1 millisecond for a key event, and check if the pressed key is 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()