import cv2 as cv
# import tensorflow as tf
# import numpy as np
# import matplotlib as plt




#灰度直方图  统计每个像素灰度出现的概率


#等比例缩放
img = cv.imread("D:\img7.jpg",1)
imginfo = img.shape
print(list(imginfo))
height = imginfo[0]
width = imginfo[1]
mode = imginfo[2]   #通道数默认为3 gbr
scaleheight = int(height*0.5)       #长宽乘以等比例的系数为等比例缩放
scalewidth = int(width*0.5)
#图像缩放方法  1.最近临域插值 2.双线性插值 3.像素关系重采样 4.立方插值
resizeimg = cv.resize(img,(scalewidth,scaleheight))
cv.imshow("newimg01",resizeimg)
cv.waitKey(0)