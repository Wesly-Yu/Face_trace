import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import  cm
from PIL import Image,ImageDraw,ImageFont,ImageFilter
import random
from captcha.image import ImageCaptcha
import math,string
import os
import io

number = ['0','1','2','3','4','5','6','7','8','9']
alpha=['a','b','c','d','e','f','g','h','i','g','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
Alpha=['A','B','C','D','E','F','G','H','I','G','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

#验证码图片的高度和宽度
size = (160,60)
#背景颜色
bgcolor = (255,255,255)
#干扰线颜色 灰色
line_color = (169,169,169)
#是否加入干扰线
draw_lines = True
#干扰线的上下限
line_numb = (1,5)
width, height = size





#创建4位验证码数字
def radom_captcha_text(char_set=number+alpha+Alpha,captsize=4):
	captcha_text = []
	for i in range(captsize):
		char = random.choice(char_set)
		captcha_text.append(char)
	return captcha_text


#绘制干扰线
def draw_line(draw,width,height):
	begin = (random.randint(0,width),random.randint(0,height))
	end = (random.randint(0,width),random.randint(0,height))
	draw.line([begin,end],fill=line_color)


def gen_captcha_text_image():
	width,height = size
	image = ImageCaptcha()
	captcha_text = radom_captcha_text()   #list格式
	captcha_text = ''.join(captcha_text)  #转换为string格式
	captcha = image.generate(captcha_text)
	# image.write(captcha_text,captcha_text+'.jpg')
	captcha_image = Image.open(captcha)
	captcha_image = np.array(captcha_image)  #转换成np.arry
	return captcha_text,captcha_image


#换换成灰度图像
def convert_gray(img):
	if len(img.shape)>2:
		gray = np.mean(img, -1)
		# 上面的转法较快，正规转法如下
		# r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
		# gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
		return gray
	else:
		return img




def train_robot():
	max_captcha = len(text)  # 验证码长度
	char_set = number + alpha + Alpha
	char_set_len = len(char_set)
	x= tf.placeholder(tf.float32,[None,height*width])
	y = tf.placeholder(tf.float32,[None,max_captcha*char_set_len])
	keep_prob = tf.placeholder(tf.float32)



#定义cnn
def crack_captcha_cnn(w_alpha=0.01,b_alpha=0.1):
	x = tf.reshape(X,shape=[-1,height,width,1])
	#3层卷积神经网络
	w_c1 =



def train_crack_captcha_cnn():



def train_robot():
	max_captcha = len(text)  # 验证码长度
	char_set = number + alpha + Alpha
	char_set_len = len(char_set)
	x= tf.placeholder(tf.float32,[None,height*width])
	y = tf.placeholder(tf.float32,[None,max_captcha*char_set_len])
	keep_prob = tf.placeholder(tf.float32)

if __name__ == '__main__':
	text,image = gen_captcha_text_image()
	path  =os.path.abspath(os.path.dirname(__file__))  # 获取当前工程目录
	image_name = text+'.png'
	f = plt.figure()
	ax = f.add_subplot(111)
	ax.text(0.1,0.9,text,ha='center',va='center')   #,transform=ax.transAxes
	plt.imshow(image)
	plt.savefig(image_name)
	plt.show()
	max_captcha = len(text)  #验证码长度





