import numpy as np
import matplotlib.pyplot as plt
import cv2

def gaussian_kernel(size):
    size = int(size)
    x, y = np.mgrid[-size:size+1, -size:size+1]
    g = np.exp(-(x**2/float(size)+y**2/float(size)))
    return g / g.sum()

def smoothed_image(image):
    image=np.array(image)
    guass=gaussian_kernel(3)
    smoothed=cv2.filter2D(image,-1,guass)
    return smoothed

def laplac_kernel():
    kernel = np.zeros((3, 3))
    kernel[0][1] = 1
    kernel[1][0] = 1
    kernel[2][1] = 1
    kernel[1][2] = 1
    kernel[1][1] = -4

    return kernel

def laplacian(image):
    kernel=laplac_kernel()
    res=cv2.filter2D(image,-1,kernel)
    return res

def unsharp():
    gauss=gaussian_kernel(3)
    LG=laplacian(gauss)
    return LG


def sharpening(signal,unsharped,alfa):
    res= signal+np.multiply(alfa,unsharped)
    return res




image=cv2.imread('flowers_blur.png')
mask=unsharp()

b,g,r=cv2.split(image)
b_unsharped=cv2.filter2D(b,-1,mask)
g_unsharped=cv2.filter2D(g,-1,mask)
r_unsharped=cv2.filter2D(r,-1,mask)

unsharped=cv2.merge([b_unsharped,g_unsharped,r_unsharped])
res1=cv2.cvtColor(unsharped, cv2.COLOR_BGR2RGB)
cv2.imwrite('res01.jpg',res1)

b_final=sharpening(b,b_unsharped,0.7).astype(int)
g_final=sharpening(g,g_unsharped,0.7).astype(int)
r_final=sharpening(r,r_unsharped,0.7).astype(int)


result_image=cv2.merge([b_final,g_final,r_final])
cv2.imwrite('first5.jpg',result_image)