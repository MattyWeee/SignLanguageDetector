import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands = 1)

offset = 20
imgSize = 300

folder = "Data/C"
counter = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        if x-offset > 0 and y-offset > 0 and y+h+offset < img.shape[0] and x+w+offset < img.shape[1]: 
            imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            aspectRatio = h/w
            if aspectRatio > 1: 
                k = imgSize/h
                wCal = math.ceil(k*w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize-wCal)/2)
                if 0 <= wGap <= imgWhite.shape[1] and 0 <= imgResize.shape[1]+wGap <= imgWhite.shape[1]:
                    imgWhite[0:imgResize.shape[0], 0+wGap:imgResize.shape[1]+wGap] = imgResize
            else:
                k = imgSize/w
                hCal = math.ceil(k*h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize-hCal)/2)
                if 0 <= hGap <= imgWhite.shape[0] and 0 <= imgResize.shape[0]+hGap <= imgWhite.shape[0]:
                    imgWhite[0+hGap:imgResize.shape[0]+hGap, 0:imgResize.shape[1]] = imgResize
        
            cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    k = cv2.waitKey(1)
    if k == 27: break
    if k == ord('s'): 
        counter += 1 
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
cap.release()
cv2.destroyAllWindows()