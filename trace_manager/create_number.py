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
# alpha=['a','b','c','d','e','f','g','h','i','g','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
# Alpha=['A','B','C','D','E','F','G','H','I','G','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

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
def radom_captcha_text(char_set=number,captsize=4):
	captcha_text = []
	for i in range(captsize):
		char = random.choice(char_set)
		captcha_text.append(char)
	return captcha_text


#将验证码写入图片中
def gen_captcha_text_image():
	width,height = size
	image = ImageCaptcha(width,height)
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
		# 上面的转法较快
		return gray
	else:
		return img


def text2vec(text):
	text_len = len(text)
	if text_len > MAX_CAPTCHA:
		raise ValueError('验证码最长4个字符')

	vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)  #4x4
	for i, c in enumerate(text):
		idx = i * CHAR_SET_LEN + int(c)
		vector[idx] = 1
	return vector


# 向量转回文本
def vec2text(vec):
	text = []
	char_pos = vec.nonzero()[0]
	for i, c in enumerate(char_pos):
		number = i % 10
		text.append(str(number))
	return "".join(text)





# 定义cnn
def defin_cnn(w_alpha=0.01,b_alpha=0.1):
	model = tf.keras.Sequential()
	x = tf.reshape(X,shape=[-1,Image_height,Image_width,1])   #-1表示让tensorflow  自动计算batch的值,最后的1表示图像的通道数，由于已经转为灰度图像，所以通道数为1
	#3层卷积神经网络--01
	w_c1 =tf.Variable(w_alpha*tf.random_normal([3,3,1,32]))
	b_c1 =tf.Variable(b_alpha*tf.random_normal([32]))
	conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x,w_c1,strides=[1,1,1,1],padding='SAME'),b_c1))  #relu:激活函数，padding:填充算法.conv2d:2d卷积,strides = [1, stride, stride, 1]
	conv1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')  #池化像素数据
	conv1 = tf.nn.dropout(conv1,keep_prob)    #keep_prob: float类型，每个元素被保留下来的概率,dropout防止或减轻过拟合而使用的函数，它一般用在全连接层
	# 3层卷积神经网络--02
	w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
	b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
	conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'),b_c2))  # relu:激活函数，padding:填充算法.conv2d:2d卷积,strides = [1, stride, stride, 1]
	conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 池化像素数据
	conv2 = tf.nn.dropout(conv2, keep_prob)
	# 3层卷积神经网络--03
	w_c3 = tf.Variable(w_alpha * tf.random_normal([3,3,64,128]))
	b_c3 = tf.Variable(b_alpha * tf.random_normal([128]))
	conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'),b_c3))  # relu:激活函数，padding:填充算法.conv2d:2d卷积,strides = [1, stride, stride, 1]
	conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 池化像素数据
	conv3 = tf.nn.dropout(conv3, keep_prob)

	#全连接层--001
	w_d = tf.Variable(w_alpha*tf.random_normal([8*20*128,1024])) #8是height=60 进行3次卷积，每次卷积为1/2,3次之后根据SAME函数 变为8.160根据3次卷积后变为20。128为生成的特征图，1024为期望的向量
	b_d = tf.Variable(b_alpha*tf.random_normal([1024]))
	dense = tf.reshape(conv3,[-1,w_d.get_shape().as_list()[0]])
	dense = tf.nn.relu(tf.add(tf.matmul(dense,w_d),b_d))
	dense = tf.nn.dropout(dense,keep_prob)
	# 全连接层--002
	w_out = tf.Variable(w_alpha*tf.random_normal([1024,max_captcha*char_set_len]))
	b_out = tf.Variable(b_alpha*tf.random_normal([max_captcha*char_set_len]))
	out = tf.add(tf.matmul(dense,w_out),b_out)
	return out
#训练
def train_cnn():
	output = defin_cnn()
	loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(output, Y))  #定义损失函数,计算sigmoid的交叉熵,衡量的是分类任务中的概率误差,reduce_mean:降维或者计算tensor（图像）的平均值
	optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)    #自适应矩阵优化,定义优化器
	predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])    #定义预测函数
	max_idx_p = tf.argmax(predict, 2)  #tf.argmax返回每行或者每列最大值的索引
	max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
	correct_pred = tf.equal(max_idx_p, max_idx_l)
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))   #定义正确率,tf.cast()函数的作用是执行 tensorflow 中张量数据类型转换
	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		step = 0
		while True:
			batch_x, batch_y = get_train_batch(64)
			_, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})
			print(step, loss_)

			# 每100 step计算一次准确率
			if step % 100 == 0:
				batch_x_test, batch_y_test = get_train_batch(100)
				acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
				print(step, acc)
				# 如果准确率大于99%,保存模型,完成训练
				if acc > 0.990:
					saver.save(sess, "./model/crack_capcha.model", global_step=step)
					break

			step += 1




# # 声明学习率为不可训练
# learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
# # 学习率递减操作,这里表示每次学习率变成上一次的0.9倍
# learning_rate_decay_op = learning_rate.assign(learning_rate * 0.9)
#
# #训练过程中根据loss情况来执行学习率递减操作
# if #这里设置一个需要衰减学习率的条件
#     sess.run(learning_rate_decay_op)



def train_robot():
	max_captcha = len(text)  # 验证码长度
	char_set = number + alpha + Alpha
	char_set_len = len(char_set)
	X= tf.placeholder(tf.float32,[None,height*width])   #计算有多少个像素点
	Y = tf.placeholder(tf.float32,[None,max_captcha*char_set_len])  #计算有多少位数字
	keep_prob = tf.placeholder(tf.float32)


def get_train_batch(batch_size=128):
	batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
	batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])

	#判断图像大小不是(60, 160, 3)  3表示图像的通道数
	def wrap_gen_captcha_text_and_image():
		while True:
			text, image = gen_captcha_text_and_image()
			if image.shape == (60, 160, 3):
				return text, image

	for i in range(batch_size):
		text, image = wrap_gen_captcha_text_and_image()
		image = convert2gray(image)

		batch_x[i, :] = image.flatten() / 255  # (image.flatten()-128)/128  mean为0
		batch_y[i, :] = text2vec(text)

	return batch_x, batch_y


if __name__ == '__main__':
	train = 0
	if train == 0:
		number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
		text,image = gen_captcha_text_image()
		print(image.shape)
		print(text)
		Image_height = 60
		Image_width = 160
		max_captcha = len(text)
		char_set = number
		char_set_len = len(number)
		X = tf.placeholder(tf.float32,[None, height*width])  # 计算有多少个像素点
		Y = tf.placeholder(tf.float32,[None, max_captcha*char_set_len])  # 计算有多少位数字   4x10
		keep_prob = tf.placeholder(tf.float32)
		train_cnn()
	if train ==1:
		number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
		IMAGE_HEIGHT = 60
		IMAGE_WIDTH = 160
		char_set = number
		CHAR_SET_LEN = len(char_set)

		text, image = gen_captcha_text_and_image()

		f = plt.figure()
		ax = f.add_subplot(111)
		ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
		plt.imshow(image)

		plt.show()

		MAX_CAPTCHA = len(text)
		image = convert2gray(image)
		image = image.flatten() / 255   #将像素转换为0-1之间

		X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])# 计算有多少个像素点
		Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])# 计算有多少位数字
		keep_prob = tf.placeholder(tf.float32)  # dropout

		predict_text = crack_captcha(image)
		print("正确: {}  预测: {}".format(text, predict_text))



