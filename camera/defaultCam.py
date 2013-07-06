import cv2.cv as cv
#import numpy as np
#import matplotlib.pyplot as plt
import time


def getPhoto():

    cv.NamedWindow("camera", 1)
    
    capture = cv.CaptureFromCAM(0)
    
    cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_WIDTH, 1024)
    cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_HEIGHT, 1024)
    time.sleep(0.02)
    cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_EXPOSURE, 2)
    time.sleep(0.02)

    img = cv.QueryFrame(capture)
    cv.SaveImage("pleasework.png",img)

if __name__=='__main__':
#    getPhoto()
    import pygame
    import pygame.camera
    from pygame.locals import *

    pygame.init()
    pygame.camera.init()
    
    camlist = pygame.camera.list_cameras()
    print camlist
    if camlist:
        cam = pygame.camera.Camera(camlist[0],(640,480))
    
    cam.start()
    image = cam.get_image()