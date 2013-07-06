import pygame
import pygame.camera
import pygame.image
#import Image as im
#import matplotlib.pyplot as plt
#import numpy as np

try:
#    pygame.init()
    white = (255, 64, 64)
    w = 640
    h = 480
    screen = pygame.display.set_mode((w, h))
    screen.fill((white))
    running = 0

    while running<10:     
        pygame.camera.init()
        cam = pygame.camera.Camera(pygame.camera.list_cameras()[0])
        cam.start()
        img = cam.get_image()
        pygame.image.save(img,'test.jpg')
        cam.stop()
        screen.fill((white))
        screen.blit(img,(0,0))
        pygame.display.flip()
        
        running+=1

    cam.stop()
    pygame.display.quit()
    

except:
    pygame.display.quit()
    cam.stop()


    
#plt.imshow()
#plt.show()