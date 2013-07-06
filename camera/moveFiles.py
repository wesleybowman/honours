import os

x=raw_input('Rename img* to: ')
y=raw_input('Numbers to start at: ')

y=int(y)
y1=str(y)
y2=str(y+1)
y3=str(y+2)

x1='goodImages/'+x+y1+'.png'
x2='goodImages/'+x+y2+'.png'
x3='goodImages/'+x+y3+'.png'

os.renames("img1.png",x1)
os.renames("img2.png",x2)
os.renames("img3.png",x3)
