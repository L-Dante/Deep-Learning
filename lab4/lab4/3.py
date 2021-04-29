# -*- coding: utf-8 -*-
#coding=utf-8
import numpy as np             #导入模块，numpy是扩展链接库
import pandas as pd             #类似一个本地的excel，偏向现在的非结构化的数据库
import tensorflow as tf
import keras
from keras.utils import np_utils
np.random.seed(10)            #设置seed可以产生的随机数据
from keras.datasets import mnist  #导入模块，下载读取mnist数据
(x_train_image,y_train_label),\
(x_test_image,y_test_label)=mnist.load_data() #下载读取mnist数据

import matplotlib.pyplot as plt

x_Train=x_train_image.reshape(60000,784).astype('float32') #以reshape转化成784个float
x_Test=x_test_image.reshape(10000,784).astype('float32')
x_Train_normalize=x_Train/255    #将features标准化
x_Test_normalize=x_Test/255
y_Train_OneHot=np_utils.to_categorical(y_train_label)#将训练数据和测试数据的label进行one-hot encoding转化
y_Test_OneHot=np_utils.to_categorical(y_test_label)

#2.建立模型
from keras.models import Sequential #可以通过Sequential模型传递一个layer的list来构造该模型,序惯模型是多个网络层的线性堆叠
from keras.layers import Dense    #全连接层
model=Sequential()
#建立输入层、隐藏层
model.add(Dense(units=1000,
                input_dim=784,
                kernel_initializer='normal',
                activation='relu'))
#建立输出层
model.add(Dense(units=10,
                kernel_initializer='normal',
                activation='softmax'))
print(model.summary())
#3、进行训练
#对训练模型进行设置，损失函数、优化器、权值
model.compile(loss='categorical_crossentropy',
              optimizer='adam',metrics=['accuracy'])
# 设置训练与验证数据比例，80%训练，20%测试，执行10个训练周期，每一个周期200个数据，显示训练过程2次
train_history=model.fit(x=x_Train_normalize,
                        y=y_Train_OneHot,validation_split=0.2,
                        epochs=100,batch_size=200,verbose=2)
#显示训练过程

def show_train_history(train_history,train,validation,hiddenUnits):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title(hiddenUnits)
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train','test'],loc='upper left')    #显示左上角标签
    plt.show()
show_train_history(train_history,'accuracy','val_accuracy', '10 hidden units')   #画出准确率评估结果
show_train_history(train_history,'loss','val_loss', '10 hidden units') #画出误差执行结果

