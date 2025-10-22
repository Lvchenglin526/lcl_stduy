# coding:utf8

# 解决 OpenMP 库冲突问题
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
# import random
# import json
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
1：规律：x是一个5维向量，如果第1个数>第5个数，则为正样本，反之为负样本
2：作业：x是一个5维向量， 
核心注意事项
(1)CrossEntropyLoss()函数内置了softmax(函数)
(2)CrossEntropyLoss()函数调用数据集时无需x和y为同维度，会自动将[0,1,2]转换成one-hot形式
[1,0,0]
[0,1,0]
[0,0,1]
但是MSE均方差loss函数需要注意x,y为同维度,因为其公式为Ln={xn - yn}*2，必须同维度才可以做向量运算
(3)nn.Linear(input_size, output_size),input_size为输入特征的维度，output_size为输出特征的维度
本实列中，input_size为输入向量维度，output_size为输出向量维度

"""


class TorchModel(nn.Module):  # 定义模型类，继承于nn.modeul
    def __init__(self, input_size):  # 定义构造器，输入input_size维度的向量
        super(TorchModel, self).__init__()  # 调用父类 nn.Module 的构造函数，确保模型能够正确初始化
        self.linear = nn.Linear(input_size, 5)  # 线性层函数，输入input_size维度的向量，输出五个特征（五个标签分类）
        self.activation = torch.softmax  # 激活函数为softmax函数
        self.loss = nn.CrossEntropyLoss()  # loss函数采用交叉熵

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):  # y为空时只输入x的值，这段定义为y初始值为空
        lx = self.linear(x)  # 调用线性层函数，输入五维向量，根据当前权重输出预测的五维向量
        # print(lx)
        y_pred = self.activation(lx,1)  # 赋予非线性能力，输入五维向量，根据当前权重输出预测的五维向量
        if y is not None:
            return self.loss(lx, y)  # 预测值和真实值计算损失，交叉熵自带softmax函数，所以计算loss时不需要再对数据进行激活函数处理
        else:
            return y_pred  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，并获得最大值索引
def build_sample():
    x = np.random.random(5)
    Max = np.argmax(x,axis=None)
    return x,Max
'''
numpy.argmax(axis=None)函数
一维数组：对于一维数组，np.argmax()直接返回最大值对应的索引。
多维数组：对于多维数组，可以通过axis参数指定沿哪个轴查找最大值索引。
axis=None：将整个数组展平为一维数组，返回最大值索引。类型为int形
axis=0：按列查找最大值索引，返回一个包含每列最大值索引的数组。
axis=1：按行查找最大值索引，返回一个包含每行最大值索引的数组。

torch.argmax()函数
torch.argmax(input, dim=None, keepdim=False)  函数定义
input：输入的张量。
keepdim = False
dim=None：将整个张量展平，返回最大值索引。类型为张量
dim=0：按列查找最大值索引，返回一个包含每列最大值索引的张量
dim=1：按行查找最大值索引，返回一个包含每列最大值索引的张量
keepdim = True
dim=None：将整个张量展平，返回最大值索引。类型为张量，保持二维
dim=0：按列查找最大值索引，返回一个包含每列最大值索引的张量，保持二维
dim=1：按行查找最大值索引，返回一个包含每列最大值索引的张量，保持原来的维度
'''

# 生成total_sample_num个样本
# 获得原始数据和最大值索引
def build_dataset(total_sample_num):
    X = []
    Y = []# 定义两个名字分别为X和Y的空列表
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)  # 列表操作，在列表末尾添加数据
        Y.append(y)
        # 使用MSELoss时需要对数据集添加[]（即增加维度），而CrossEntropyLoss（CE）通常不需要，这是因为两者对输入张量的形状要求不同
        # MSEloss函数需要让输入和输出（目标张量）维度一致，Celoss函数在内部会自动把索引转化成one-hot,所以不需要
    # print(X)#type(X)为list
    # print(Y)#type(Y)为list
    return torch.FloatTensor(np.array(X)), torch.LongTensor(np.array(Y))  # 将build_sample函数创建的列表样本转换为张量的数据形式
'''
如果使用return torch.FloatTensor(X), torch.LongTensor(Y)
报错：Creating a tensor from a list of numpy.ndarrays is extremely slow.
 Please consider converting the list to a single numpy.ndarray with numpy.array()
 before converting to a tensor.
这个警告的原因是直接从包含多个numpy.ndarray的列表创建PyTorch张量效率极低，
因为PyTorch需要逐个处理每个数组并进行类型推断和内存分配，建议先将列表转换为单个numpy.ndarray再创建张量以提升性能。
本例中numpy更名为np
所以使用np.array(),先将X和Y转化成为单个多维数组
'''
# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()  # 测试模式
    test_sample_num = 100  # 生成测试样本数量
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0  # 对于判断正确和错误的结果进行计数
    count1,count2,count3,count4,count5 = 0,0,0,0,0  # 对于分类的结果进行计数，属于哪一类，哪一类计数
    with torch.no_grad():  # 不更新梯度
        y_pred = model(x)  # 模型预测输出的五维向量 model.forward(x)
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比，利用zip同时迭代两个变量
            if torch.argmax(y_p, dim=None, keepdim=False) == int(y_t):
                correct += 1  # 样本判断正确
            else:
                wrong += 1
            if int(y_t) == 0:  # 计数生成的测试样本中对应的正确类别的数量
                count1 += 1
            elif int(y_t) == 1:
                count2 += 1
            elif int(y_t) == 2:
                count3 += 1
            elif int(y_t) == 3:
                count4 += 1
            elif int(y_t) == 4:
                count5 += 1
    print("本次预测集中真实样本数为%d个一类样本，%d个二类样本,%d个三类样本,%d个四类样本,%d个五类样本" % (count1, count2, count3, count4, count5))
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 10  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 6000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率
    model = TorchModel(input_size)  # 建立模型
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 选择优化器
    log = []
    train_x, train_y = build_dataset(train_sample)  # 创建训练集，正常任务是读取训练集
    # 训练过程
    for epoch in range(epoch_num):  # epoch_num次训练
        model.train()  # 模型训练
        watch_loss = []  # 定义空列表用于存储每一次的loss率，方便计算均值
        for batch_index in range(train_sample // batch_size):  # 训练总样本除以每次训练的样本数，获得每轮训练中的训练次数
            #取出一个batch数据作为输入   train_x[0:20]  train_y[0:20] train_x[20:40]  train_y[20:40]
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]  # train_x[0*20：1*20]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model.forward(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())  # 每一次的loss率存入watch_loss列表中
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))  # epoch从0开始
        acc = evaluate(model)  # 测试本轮模型结果，存储了模型预测的正确率
        log.append([acc, float(np.mean(watch_loss))])  # 存储loss平均值
    # 保存模型
    torch.save(model.state_dict(), "model.bin")  # 通过torch内置的state_dict()函数通过词典的形式存储模型参数,""中为存储文件
    # 画图
    # print(log)
    # plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    # plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    # plt.legend()
    # plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5  # 设置输入向量特征的维度
    model = TorchModel(input_size)  # 实例化对象
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    # print(model.state_dict())  # 打印模型权重

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
        # print(result)  # 得到的预测结果为每个类的概率，所有类别的概率和为1
    for vec, res in zip(input_vec, result):
        # print(res)  # tensor([0.3815, 0.1342, 0.3477, 0.0925, 0.0441])
        # print(type(res))  # 张量tensor
        # print(type(np.argmax(res)))  # 张量tensor
        print("输入：%s,\n预测类别：第%d类\n" % (vec, np.argmax(res)+1))  # 打印结果


if __name__ == "__main__":
    main()
    test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.88920843],
                [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.90797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.99349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    predict("model.bin", test_vec)
