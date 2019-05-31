import cv2 as cv
# import tensorflow as tf
import numpy as np
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

# 最近临域插值  双线性插值 原理
# src 10*20  dst 5*10
# dst<- src
# (1,2) <- (2,4)
# newx = x*(src 行/目标 行)
# newy = y*(src 列/目标 列)
#
#
# 双线性插值
# newx = 20% *x(上)+80% *x(下)
# newy = 20% *y(上)+80% *y(下)


#1读取图片属性
img = cv.imread("D:\img7.jpg",1)
imginfo = img.shape
print(list(imginfo))
height = imginfo[0]
width = imginfo[1]
dstHeight = int(height/2)
dstWith = int(width/2)
#2创建空白模板
dstimg = np.zeros((dstHeight,dstWith,3),np.uint8)  #uint8范围为0-255
for i in range(0,dstHeight):
    for j in range(0,dstWith):
        newi = int(i*(height*1.0/dstHeight))
        newj = int(j*(width*1.0/dstWith))
        dstimg[i,j] = img[newi,newj]

cv.imshow("dst",dstimg)
cv.waitKey(0)



#图片移位
img = cv.imread("D:\img9.jpg",1)
cv.imshow("src",img)
imginfo = img.shape
height = imginfo[0]
width = imginfo[1]
matfloat = np.float32([[1,0,100],[0,1,200]])  #设置偏移量
dst = cv.warpAffine(img,matfloat,(height,width))
cv.imshow("new",dst)
cv.waitKey(0)