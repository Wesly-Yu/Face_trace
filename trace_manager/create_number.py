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
size = (240,60)
#背景颜色
bgcolor = (255,255,255)
#干扰线颜色 灰色
line_color = (169,169,169)
#是否加入干扰线
draw_lines = True
#干扰线的上下限
line_numb = (1,5)






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
	captcha_text = radom_captcha_text()
	captcha_text = ''.join(captcha_text)
	captcha = image.generate(captcha_text)
	# image.write(captcha_text,captcha_text+'.jpg')
	captcha_image = Image.open(captcha)
	captcha_image = np.array(captcha_image)
	return captcha_text,captcha_image
if __name__ == '__main__':
	text,image = gen_captcha_text_image()
	path  =os.path.abspath(os.path.dirname(__file__))  # 获取当前工程目录
	image_path = path+'\\'+'images'+'\\'+text
	f = plt.figure()
	ax = f.add_subplot(111)
	ax.text(0.1,0.9,text,ha='center',va='center')   #,transform=ax.transAxes
	plt.imshow(image)
	plt.show()
	plt.savefig(image_path)



