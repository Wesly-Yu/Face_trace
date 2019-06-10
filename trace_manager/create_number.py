import tensorflow as tf
import numpy as np
import matplotlib as plt
from PIL import Image
import random
from captcha.image import ImageCaptcha

number = ['0','1','2','3','4','5','6','7','8','9']
alpha=['a','b','c','d','e','f','g','h','i','g','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
Alpha=['A','B','C','D','E','F','G','H','I','G','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

#验证码图片的高度和宽度
size = (120,40)
#背景颜色
bgcolor = (255,255,255)
#干扰线颜色 灰色
line_color = (169,169,169)





#创建4位验证码数字
def radom_captcha_text(char_set=number+alpha+Alpha,captsize=4):
	captcha_text = []
	for i in range(captsize):
		char = random.choice(char_set)
		captcha_text.append(char)
	return captcha_text


#绘制干扰线
# def draw_line():