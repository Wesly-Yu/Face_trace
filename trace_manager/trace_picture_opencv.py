import cv2 as cv
import numpy as np
import matplotlib as plt




#灰度直方图  统计每个像素灰度出现的概率


# #等比例缩放
# img = cv.imread("D:\img7.jpg",1)
# imginfo = img.shape
# print(list(imginfo))
# height = imginfo[0]
# width = imginfo[1]
# mode = imginfo[2]   #通道数默认为3 gbr
# scaleheight = int(height*0.5)       #长宽乘以等比例的系数为等比例缩放
# scalewidth = int(width*0.5)
# #图像缩放方法  1.最近临域插值 2.双线性插值 3.像素关系重采样 4.立方插值
# resizeimg = cv.resize(img,(scalewidth,scaleheight))
# cv.imshow("newimg01",resizeimg)
# cv.waitKey(0)
#
# # 最近临域插值  双线性插值 原理
# # src 10*20  dst 5*10
# # dst<- src
# # (1,2) <- (2,4)
# # newx = x*(src 行/目标 行)
# # newy = y*(src 列/目标 列)
# #
# #
# # 双线性插值
# # newx = 20% *x(上)+80% *x(下)
# # newy = 20% *y(上)+80% *y(下)
#
#
# #1读取图片属性
# img = cv.imread("D:\img7.jpg",1)
# imginfo = img.shape
# print(list(imginfo))
# height = imginfo[0]
# width = imginfo[1]
# dstHeight = int(height/2)
# dstWith = int(width/2)
# #2创建空白模板
# dstimg = np.zeros((dstHeight,dstWith,3),np.uint8)  #uint8范围为0-255
# for i in range(0,dstHeight):
#     for j in range(0,dstWith):
#         newi = int(i*(height*1.0/dstHeight))
#         newj = int(j*(width*1.0/dstWith))
#         dstimg[i,j] = img[newi,newj]
#
# cv.imshow("dst",dstimg)
# cv.waitKey(0)
#
#
#
# # 图片移位
# img = cv.imread("D:\img9.jpg",1)
# cv.imshow("src",img)
# imginfo = img.shape
# height = imginfo[0]
# width = imginfo[1]
# matfloat = np.float32([[1,0,100],[0,1,200]])  #设置偏移量
# dst = cv.warpAffine(img,matfloat,(height,width))
# cv.imshow("new",dst)
# cv.waitKey(0)
#
#
# #图片直方图均衡化rgb
# img =cv.imread("D:\hua.jpg",1)
# cv.imshow("src",img)
# (b,g,r)=cv.split(img)
# bH=cv.equalizeHist(b)
# gH=cv.equalizeHist(g)
# rH=cv.equalizeHist(r)
# result=cv.merge((bH,gH,rH))
# cv.imshow('dst',result)
# cv.waitKey(0)
#
#
#
# #图片直方图均衡化yuv
# img =cv.imread("D:\hua.jpg",1)
# cv.imshow("src",img)
# imgYUV = cv.cvtColor(img,cv.COLOR_BGR2YCrCb)
# channelYUV= cv.split(imgYUV)
# channelYUV[0]=cv.equalizeHist(channelYUV[0])
# channels = cv.merge(channelYUV)
# result = cv.cvtColor(channels,cv.COLOR_YCrCb2BGR)
# cv.imshow('dst',result)
# cv.waitKey(0)
#
#
#
# #美白  双边滤波
# img =cv.imread("D:\woman.jpg",1)
# cv.imshow("src",img)
# dst = cv.bilateralFilter(img,50,40,40)
# cv.imshow('dst',dst)
# cv.waitKey(0)
#
#
# #高斯均值滤波
# img =cv.imread("D:\zaodian.jpg",1)
# cv.imshow("src",img)
# dst = cv.GaussianBlur(img,(5,5),1.5)
# cv.imshow('dst',dst)
# cv.waitKey(0)
#
#
#
# #均值滤波
# img =cv.imread("D:\zaodian.jpg",1)
# cv.imshow("src",img)
# dst = cv.GaussianBlur(img,(5,5),1.5)
# cv.imshow('dst',dst)
# cv.waitKey(0)
#
#
# #读取视频,分解成图片
#
# cap = cv.VideoCapture("D:\\1.mkv")
# isopen = cap.isOpened
# print(isopen)
# fps = cap.get(cv.CAP_PROP_FPS)
# width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
# print(fps,width,height)




# #读取图片
# face_xml = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
# eye_xml = cv.CascadeClassifier('haarcascade_eye.xml')
# img =cv.imread("D:\man.jpg",1)
# cv.imshow("src",img)
# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# faces = face_xml.detectMultiScale(gray,1.3,5)  #1.3为缩放系数，5为像素大小
# for(x,y,w,h) in faces:
#     cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)  #2表示线条宽度,这一部为画出人脸的范围
#     roi_gray = gray[y:y+h,x:x+w]           #人脸灰度图像范围
#     roi_color = img[y:y+h,x:x+w]            #人脸彩色图像范围
#     eyes = eye_xml.detectMultiScale(roi_gray)    #识别眼睛范围
#     print('eye=',len(eyes))
#     for (ex,ey,ew,eh) in eyes:
#         cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
# cv.imshow('dst',img)
# cv.waitKey(0)


#根据身高和体重判断性别


rand1= np.array([[155,48],[159,50],[164,53],[168,56],[172,57]])
rand2 = np.array([[152,54],[156,56],[160,58],[172,67],[176,65]])
label = np.array([[0],[0],[0],[0],[0],[1],[1],[1],[1],[1]])
data = np.vstack((rand1,rand2))
data = np.array(data,dtype='float32')
#训练样本,svm即向量机,属性设置
svm = cv.ml.SVM_create()     #ml:machine learning
svm.setType(cv.ml.SVM_C_SVC)  #svm type
svm.setKernel(cv.ml.SVM_LINEAR)  #线性内核
svm.setC(0.01)
#训练
result = svm.train(data,cv.ml.ROW_SAMPLE,label)
pt_data = np.vstack([[167,55],[162,57]])
pt_data = np.array(pt_data,dtype='float32')
(para1,para2)=svm.predict(pt_data)
print(para2)

