from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.utils import plot_model
import string
from tqdm import tqdm
from tensorflow.python.keras.utils.np_utils import to_categorical



#定义字符的类型以及个数
characters = string.digits + string.ascii_uppercase+string.ascii_lowercase
width, height, n_len, n_class = 160, 60, 4, len(characters)
SAVE_PATH="./model/learing_capcha.model"

#防止tensorflow占用所有显存
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

def gen_captcha(batch_size=32):
    X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
    y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(n_len)]
    generator = ImageCaptcha(width=width, height=height)
    while True:
        for i in range(batch_size):
            random_str = ''.join([random.choice(characters) for j in range(4)])
            X[i] = generator.generate_image(random_str)
            for j, ch in enumerate(random_str):
                y[j][i, :] = 0
                y[j][i, characters.find(ch)] = 1
        yield X, y




def decode(y):
    y = np.argmax(np.array(y), axis=2)[:, 0]
    return ''.join([characters[x] for x in y])
X, y = next(gen_captcha(1))
plt.imshow(X[0])
plt.title(decode(y))



#定义网络结构,4层卷积+1层全连接
def train_cnn():
    input_tensor = Input((height, width, 3))
    x = input_tensor
    for i, n_cnn in enumerate([2, 2, 2, 2, 2]):
        for j in range(n_cnn):
            x = Conv2D(32 * 2 ** min(i, 3), kernel_size=3, padding='same', kernel_initializer='he_uniform')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
        x = MaxPooling2D(2)(x)

    x = Flatten()(x)
    x = [Dense(n_class, activation='softmax', name='c%d' % (i + 1))(x) for i in range(n_len)]
    model = Model(inputs=input_tensor, outputs=x)
    return model
    # input_tensor = Input((height, width, 3))
    # x = input_tensor
    # for i in range(4):
    #     x = Convolution2D(32*2**i, 3, 3, activation='relu')(x)
    #     x = Convolution2D(32*2**i, 3, 3, activation='relu')(x)
    #     x = MaxPooling2D((2, 2))(x)
    #
    # x = Flatten()(x)
    # x = Dropout(0.25)(x)
    # x = [Dense(n_class, activation='softmax', name='c%d'%(i+1))(x) for i in range(4)]
    # model = Model(input=input_tensor, output=x)
    #
    # model.compile(loss='categorical_crossentropy',
    #               optimizer='adadelta',
    #               metrics=['accuracy'])
    # return  model



def run_cnn():
    try:
        model = tf.keras.models.load_model(SAVE_PATH + 'model')
    except Exception as e:
        print('#######Exception', e)
        model = train_cnn()
        model.fit_generator(gen_captcha(),samples_per_epoch=102400, nb_epoch=20,
                            validation_data=gen_captcha(), nb_val_samples=1280)


#计算模型总体准确率
def evaluate(model, batch_num=20):
    batch_acc = 0
    step = 0
    generator = gen_captcha()
    for i in tqdm(range(batch_num)):
        X, y = next(generator)
        y_pred = model.predict(X)
        y_pred = np.argmax(y_pred, axis=2).T
        y_true = np.argmax(y, axis=2).T
        batch_acc += np.mean(map(np.array_equal, y_true, y_pred))
    print('准确率：%d,训练次数:%d',(batch_acc,batch_num))
    model.save(SAVE_PATH)
    return batch_acc / batch_num



if __name__ == '__main__':
    model = train_cnn()
    evaluate(model)
    run_cnn()