#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json

"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
这个模型用来寻找提前指定的字符位于输入的字符串中的什么位置
用RNN和交叉熵实现

"""

class TorchModel(nn.Module):
    def __init__(self, vocab, vector_dim, input_size, hidden_size):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # embedding层
        self.layer = nn.RNN(input_size, hidden_size, bias=False, batch_first=True)
        self.classify = nn.Linear(hidden_size, hidden_size)     # 线性层
        self.activation = torch.softmax     # softmax
        self.loss = nn.CrossEntropyLoss()  # loss函数采用交叉熵

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)     # 样本数*文本长度*字符维度
        output, lx = self.layer(x)   # output.shape=样本数*文本长度*输出维度 x.shape = 1*样本数*输出维度
        lx = lx.squeeze()  # 丢弃一维后为，lx.shape=样本数*输出维度
        lx = self.classify(lx)
        y_pred = self.activation(lx, 1)  # y_pred.shape=样本数*输出维度
        if y is not None:
            return self.loss(lx, y)   # 预测值和真实值计算损失,y在交叉熵中自动转化为RNN输出维度
        else:
            return y_pred                 # 输出预测结果

# 字符集随便挑了一些字，实际上还可以扩充
# 为每个字生成一个标号
# {"a":1, "b":2, "c":3...}
# abc -> [1,2,3]
def build_vocab():
    chars = "你我他defghijklmnopqrstuvwxyz"  # 字符集
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1   # 每个字对应一个序号
    vocab['unk'] = len(vocab) 
    return vocab

# 随机生成一个样本
# 从所有字中选取sentence_length个字
# 反之为负样本
def build_sample(vocab, sentence_length):
    # 随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    # print(x)
    y = 0
    for index, i in enumerate(x):
        if i == '我':
            # print(index)
            # print(i)
            y = index + 1
            break
        else:
            y = len(vocab) + 1
    x = [vocab.get(word, vocab['unk']) for word in x]   # 将字转换成序号，为了做embedding
    # print(x)
    # print(y)
    return x, y

# 建立数据集
# 输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

# 建立模型
def build_model(vocab, char_dim,input_size, hidden_size):
    model = TorchModel(vocab, char_dim, input_size, hidden_size)
    return model

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)   # 建立200个用于测试的样本
    # print("本次预测集中共有%d个包含‘你’，%d个包含‘我’"%(sum(Y1), sum(Y2)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)      # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if np.argmax(y_p) == int(y_t):
                correct += 1   # 样本判断正确
            # elif float(y_p) >= 0.5 and int(y_t) == 1:
            #     correct += 1   # 正样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)


def main():
    # 配置参数
    epoch_num = 20        # 训练轮数
    batch_size = 20       # 每次训练样本个数
    train_sample = 5000   # 每轮训练总共训练的样本总数
    char_dim = 20         # 每个字的维度
    sentence_length = 6   # 样本文本长度
    learning_rate = 0.005 # 学习率
    input_size = 20
    hidden_size = 30
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, input_size, hidden_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length) # 构造一组训练样本
            optim.zero_grad()    # 梯度归零
            loss = model.forward(x, y)   # 计算loss
            loss.backward()      # 计算梯度
            optim.step()         # 更新权重
            
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)   # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])

    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

# 使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    input_size = 20
    hidden_size = 30
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) # 加载字符表
    model = build_model(vocab, char_dim, input_size, hidden_size)     # 建立模型
    model.load_state_dict(torch.load(model_path))             # 加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab.get(char,vocab['unk']) for char in input_string])  # 将输入序列化
        # print(x)
    model.eval()   # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.LongTensor(x))  # 模型预测
    for i, input_string in enumerate(input_strings):
        if np.argmax(result[i]) <= sentence_length:
            print("输入：%s, 预测位置：第%d个字为“我”" % (input_string, np.argmax(result[i])))  # 打印结果
        else:
            print("输入：%s, 预测位置：该字符串中未找到指定字符" % input_string)


# 自定义输入字符串数量和字符串
def putstrings(x):
    counti = 0
    test_strings = []
    while True:
        print('第%d个六字字符串' % (counti + 1))
        strings = input('请输入需要预测的六字字符串：')
        test_strings.append(strings)
        counti += 1
        if counti == x:
            break
    return test_strings



if __name__ == "__main__":
    main()
    test_strings = ["fnvf我e", "wz你dfg", "我qwdeg", "n我kwww", "rqadeg"]
    # test_strings = putstrings(int(input('输入需要预测的六字字符串数量：')))  # 自定义需要预测的字符串和数量
    predict("model.bin", "vocab.json", test_strings)
