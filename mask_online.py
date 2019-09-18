from random import randint
import itertools
import numpy as np
import cv2
from math import *


def continuous_mask(height, width,num,maxAngle,maxLength,maxBrushWidth,channels=3):
    """Generates a continuous mask with lines, circles and elipses"""

    img = np.zeros((height, width, channels), np.uint8)

    for j in range(1):
        startX = randint(0, width)
        startY = randint(0, height)
        for i in range(0,randint(1,num)):
            angle = randint(0,maxAngle)
            if i%2==0:
                angle = 360 - angle
            length = randint(1,maxLength)
            brushWidth = randint(1, maxBrushWidth)
            endX   = startX + int(length * sin(angle))
            endY   = startY + int(length * cos(angle))
            if endX>255:
                endX = 255
            if endX<0:
                endX = 0
            if endY>255:
                endY = 255
            if endY<0:
                endY = 0        
            cv2.line(img, (startX,startY),(endX,endY),(255,255,255),brushWidth)
            cv2.circle(img, (endX,endY),brushWidth//2,(255,255,255),-1)
            startY = endY
            startX = endX


    img2 = np.zeros((height, width,1))
    img2[:, :,0] = img[:, :, 0]
    img2[img2>1] = 1

    return 1-img2



def discontinuous_mask(height, width,num,low,high,channels=3):
    """Generates a discontinuous mask with lines, circles and elipses
       When we were training, we generated more elipses
    """
    img = np.zeros((height, width, channels), np.uint8)

    # Set size scale
    size = int((width + height) * 0.1) 
    if width < 64 or height < 64:
        raise Exception("Width and Height of mask must be at least 64!")

    # Draw random lines
    for _ in range(randint(1, num)):                     
        x1, x2 = randint(1, width), randint(1, width)   
        y1, y2 = randint(1, height), randint(1, height) 
        thickness = randint(3, size)                    
        cv2.line(img,(x1,y1),(x2,y2),(1,1,1),thickness) 

    # Draw random circles
    for _ in range(randint(1, num)):                     
        x1, y1 = randint(1, width), randint(1, height)  
        radius = randint(3, size)                       
        cv2.circle(img,(x1,y1),radius,(1,1,1), -1)      

    # Draw randow rectangle
    for _ in range(randint(1, num)):
        x1,y1 = randint(1, width), randint(1, height)
        x2,y2 = randint(1, width), randint(1, height)
        cv2.rectangle(img, (x1, y1), (x2, y2), (1, 1, 1), -1)

    # Draw random ellipses
    for _ in range(randint(1, num)):                   
        x1, y1 = randint(1, width), randint(1, height)  
        s1, s2 = randint(low, high), randint(low, high)
        a1, a2, a3 = randint(3, 180), randint(3, 180), randint(3, 180)  
        thickness = randint(3, size)                                    
        cv2.ellipse(img, (x1,y1), (s1,s2), a1, a2, a3,(1,1,1), thickness)  

    img2 = np.zeros((height, width,1))
    img2[:, :,0] = img[:, :, 0]

    return 1-img2



if __name__ == '__main__':

    mask = continuous_mask(256,256,60,360,32,50)
    cv2.namedWindow('dasd')
    cv2.imshow('dasd',mask)
    cv2.waitKey(0)
