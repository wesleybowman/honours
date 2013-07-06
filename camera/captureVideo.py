import cv2.cv as cv


capture=cv.CaptureFromCAM(0)
temp=cv.QueryFrame(capture)
writer=cv.CreateVideoWriter("test.avi", 1, 15, cv.GetSize(temp), 1)
count=0
while count<250:
    image=cv.QueryFrame(capture)
    cv.WriteFrame(writer, image)
    cv.ShowImage('Image_Window',image)
    cv.WaitKey(2)
    count+=1