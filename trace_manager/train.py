from  tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



# #装载数据,one_hot 表示如果有一个数据为1，别的数据就为0
minst = input_data.read_data_sets('E:\MNIST_data',one_hot=True)
