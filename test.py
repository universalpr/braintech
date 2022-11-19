import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import math

import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

plus = 20
imgsize = 300

folder = "Data/U"
counter = 0

labels = ['A', 'B', 'C', 'I', 'E', 'L', 'O', 'V', 'Y', 'U']

while True:
    success, img = cap.read()
    final = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imagewhite = np.ones((imgsize,imgsize, 3), np.uint8)*255
        imgCrop = img[y-plus:y + h+plus, x-plus:x + w+plus]

        imgcropshape = imgCrop.shape




        operator = h/w

        if operator >1:
            k = imgsize/h
            calcw = math.ceil(k*w)
            imgresize = cv2.resize(imgCrop,(calcw, imgsize))
            imgresshape = imgresize.shape
            wGap = math.ceil((imgsize-calcw)/2)
            imagewhite[0:imgresshape[0], wGap:calcw+wGap] = imgresize
            prediction, index = classifier.getPrediction(imagewhite, draw=False)
            print(prediction, index)


        else:
            k = imgsize/w
            calch = math.ceil(k*h)
            imgresize = cv2.resize(imgCrop,(imgsize,calch))
            imgresshape = imgresize.shape
            hgap = math.ceil((imgsize-calch)/2)
            imagewhite[hgap:calch+hgap, :] = imgresize
            prediction, index = classifier.getPrediction(imagewhite, draw=False)
            print(prediction, index)

        cv2.rectangle(final, (x-plus, y-plus-50),
                      (x-plus+90, y-plus-50+50), (255, 0, 255), cv2.FILLED)
        cv2.putText(final, labels[index], (x, y-30), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 3)
        cv2.rectangle(final, (x-plus, y-plus),
                      (x+w+plus, y+h+plus), (255, 0, 255), 4)

        cv2.imshow("CropImage", imgCrop)
        cv2.imshow("Numpy dock", imagewhite)



    cv2.imshow("Result", final)

    cv2.waitKey(1)

