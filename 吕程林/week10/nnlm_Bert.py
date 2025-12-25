#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertModel, BertConfig, BertTokenizer
"""
基于pytorch的Bert语言模型
生成任务
"""


class LanguageModel(nn.Module):
    def __init__(self, input_dim, vocab):
        super(LanguageModel, self).__init__()
        self.bert_config = BertConfig.from_pretrained(r"D:\ANACONDA\envs\py312\Lib\site-packages\modelscope\hub\models\google-bert\bert-base-chinese")
        self.bert_config.num_hidden_layers = 6
        self.Bert = BertModel(config=self.bert_config)
        # self.embedding = nn.Embedding(len(vocab), input_dim)
        # self.layer = nn.LSTM(input_dim, input_dim, num_layers=1, batch_first=True)
        self.classify = nn.Linear(self.bert_config.hidden_size, len(vocab))
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, attention_mask=None, y=None):
        # x = self.embedding(x)       # output shape:(batch_size, sen_len, input_dim)
        if attention_mask is not None:
            output = self.Bert(x, attention_mask=attention_mask)        # output shape:(batch_size, sen_len, input_dim)
        else:
            output = self.Bert(x)
        # print(output.last_hidden_state)
        x_last = self.dropout(output.last_hidden_state)
        y_pred = self.classify(x_last)   # output shape:(batch_size, sen_len, vocab_size)
        if y is not None:
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            return torch.softmax(y_pred, dim=-1)

# 加载字表
def build_vocab(vocab_path):
    vocab = {"<pad>":0}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]         # 去掉结尾换行符
            vocab[char] = index + 1  # 留出0位给pad token
    return vocab

# 加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="GBK") as f:
        for line in f:
            corpus += line.strip()
    # print(corpus)
    return corpus

# 随机生成一个样本
# 从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(vocab, window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  # 输入输出错开一位
    # print(window, target)
    # x = [vocab.get(word, vocab["<UNK>"]) for word in window]   # 将字转换成序号
    # y = [vocab.get(word, vocab["<UNK>"]) for word in target]
    tokenizer = BertTokenizer("vocab.txt")  # 传入词汇表路径，利用现有词汇表做分词和句子向量化
    Max = window_size + 2
    x = tokenizer.encode_plus(window,
                              max_length=Max,
                              padding="max_length",
                              truncation=True,
                              return_attention_mask=True)
    # 向量化后返回出一个字典，字典中包含了input_ids，token_type_ids，attention_mask
    y = tokenizer.encode_plus(target,
                              max_length=Max,
                              padding="max_length",
                              truncation=True,
                              return_attention_mask=True
                              )
    x_input_ids = x["input_ids"]
    x_attention_mask = x["attention_mask"]
    x_token_type_ids = x["token_type_ids"]
    y_input_ids = y["input_ids"]
    y_attention_mask = y["attention_mask"]
    y_token_type_ids = y["token_type_ids"]
    # print(x_input_ids, len(x_attention_mask), len(y_input_ids))
    return x_input_ids, x_attention_mask, y_input_ids

# 建立数据集
# sample_length 输入需要的样本数量。需要多少生成多少
# vocab 词表
# window_size 样本长度
# corpus 语料字符串
def build_dataset(sample_length, vocab, window_size, corpus):
    dataset_x_input_ids = []
    dataset_x_attention_mask = []
    dataset_y_input_ids = []
    for i in range(sample_length):
        x_input_id, x_att_mask, y_input_id \
            = build_sample(vocab, window_size, corpus)
        dataset_x_input_ids.append(x_input_id)
        dataset_x_attention_mask.append(x_att_mask)
        dataset_y_input_ids.append(y_input_id)
    return (torch.LongTensor(dataset_x_input_ids),
            torch.LongTensor(dataset_x_attention_mask),
            torch.LongTensor(dataset_y_input_ids))

# 建立模型
def build_model(vocab, char_dim):
    model = LanguageModel(char_dim, vocab)
    return model

# 文本生成测试代码
def generate_sentence(openings, model, vocab, window_size):
    reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    with torch.no_grad():
        pred_char = ""
        # 生成了换行符，或生成文本超过30字则终止迭代
        while pred_char != "\n" and len(openings) <= 30:
            openings += pred_char
            # x = [vocab.get(char, vocab["<UNK>"]) for char in openings[-window_size:]]
            tokenizer = BertTokenizer("vocab.txt")
            x = tokenizer.encode_plus(openings,
                                      max_length=window_size,
                                      padding="max_length",
                                      truncation=True,
                                      return_attention_mask=True
                                      )
            x_input_ids = x["input_ids"]
            x_in = torch.LongTensor([x_input_ids])
            if torch.cuda.is_available():
                x_in = x_in.cuda()
            y = model(x_in)[0][-1]
            index = sampling_strategy(y)
            pred_char = reverse_vocab[index]
    return openings

def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"

    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)


# 计算文本ppl
def calc_perplexity(sentence, model, vocab, window_size):
    prob = 0
    model.eval()
    with torch.no_grad():
        for i in range(1, len(sentence)):
            start = max(0, i - window_size)
            window = sentence[start:i]
            # x = [vocab.get(char, vocab["<UNK>"]) for char in window]
            tokenizer = BertTokenizer("vocab.txt")
            x = tokenizer.encode_plus(window,
                                      max_length=window_size,
                                      padding="max_length",
                                      truncation=True,
                                      return_attention_mask=True
                                      )
            x_input_ids = x["input_ids"]
            x_in = torch.LongTensor([x_input_ids])
            target = sentence[i]
            target_index = vocab.get(target, vocab["<UNK>"])
            if torch.cuda.is_available():
                x_in = x_in.cuda()
            pred_prob_distribute = model(x_in)[0][-1]
            target_prob = pred_prob_distribute[target_index]
            prob += math.log(target_prob, 10)
    return 2 ** (prob * (-1 / len(sentence)))


def train(corpus_path, save_weight=False):
    epoch_num = 20         # 训练轮数
    batch_size = 32        # 每次训练样本个数
    train_sample = 10000   # 每轮训练总共训练的样本总数
    char_dim = 256         # 每个字的维度
    window_size = 128       # 样本文本长度
    vocab = build_vocab("vocab.txt")       # 建立字表
    corpus = load_corpus(corpus_path)      # 加载语料
    model = build_model(vocab, char_dim)   # 建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=2e-5)   # 建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, x_att, y = build_dataset(batch_size, vocab, window_size, corpus)  # 构建一组训练样本
            # print(type(x), y.shape)
            if torch.cuda.is_available():
                x, x_att, y = x.cuda(), x_att.cuda(), y.cuda()
            optim.zero_grad()    # 梯度归零
            loss = model(x, x_att, y)   # 计算loss
            loss.backward()      # 计算梯度
            optim.step()         # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("李清脚步一顿，似乎是明白了", model, vocab, window_size))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, vocab, window_size))
    if save_weight:
        model_path = os.path.join(r"D:\AI  Learning\课程内容\TEN_Week\TENWEEK\lstm语言模型生成文本", "DIY_nnlm_bert.pth")
        torch.save(model.state_dict(), model_path)
        print(f"模型权重已保存至 {model_path}")
        return
    else:
        return



if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train("corpus.txt", True)
    # window_size = 10
    # vocab = build_vocab("vocab.txt")
    # corpus = load_corpus("corpus.txt")
    # build_sample(vocab, window_size, corpus)
