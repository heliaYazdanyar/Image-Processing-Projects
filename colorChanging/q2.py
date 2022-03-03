import numpy as np
import cv2
import matplotlib.pyplot as plt

# processing pink image
pink_pic = cv2.imread('Pink.jpg')
rr = cv2.cvtColor(pink_pic, cv2.COLOR_BGR2RGB)

hsv = cv2.cvtColor(pink_pic, cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(hsv)

newH1= np.where(h>=80,h-60,h)
newH2=np.where(newH1<=13,newH1+115,newH1)
newH3=np.where((newH2>14) &(newH2<=25)  & (s>148) &(v>100) ,newH2+70,newH2)

result = cv2.merge([newH3,s,v])
blue = cv2.cvtColor(result, cv2.COLOR_HSV2RGB)
plt.imshow(blue)
plt.show()


# processing yellow image
yellow = cv2.imread('Yellow.jpg')
rr = cv2.cvtColor(yellow, cv2.COLOR_BGR2RGB)
plt.imshow(rr)
hsv=cv2.cvtColor(yellow ,cv2.COLOR_BGR2HSV)
h,s,v=cv2.split(hsv)
newH1=np.where((h>20) & (h<=24) ,h+150,h)
newH2 = np.where((newH1>=25)& (newH1<=30) & (s<160) & (v>180) ,newH1+150,newH1 )
result=cv2.merge([newH2,s,v])
red=cv2.cvtColor(result,cv2.COLOR_HSV2RGB)
plt.imshow(red)
plt.show()


# second result for yellow image
newH1=np.where((h>=18) & (h<=24) & (v>100) ,h+150,h)
newH2 = np.where((newH1>=25)& (newH1<=30) & (s<180),newH1+150,newH1 )
result=cv2.merge([newH2,s,v])
red=cv2.cvtColor(result,cv2.COLOR_HSV2RGB)
plt.imshow(red)
plt.show()