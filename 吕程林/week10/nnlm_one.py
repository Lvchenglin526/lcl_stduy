# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
from transformers import BertModel, BertTokenizer, BertConfig

"""
基于pytorch的BERT语言模型
"""


class LanguageModel(nn.Module):
    def __init__(self, config_path, vocab_size):
        super(LanguageModel, self).__init__()
        # 加载预训练的 BERT 模型
        # 注意：这里我们加载的是 BERT 的配置和模型，不包括下游任务的头部
        self.bert_config = BertConfig.from_pretrained(config_path)
        self.bert_config.num_hidden_layers = 3
        self.bert = BertModel(config=self.bert_config)

        # BERT 的隐藏层维度
        hidden_dim = self.bert_config.hidden_size

        # 分类层：将 BERT 的输出映射到词汇表大小
        self.classify = nn.Linear(hidden_dim, vocab_size)

        # Dropout
        self.dropout = nn.Dropout(0.1)

        # 损失函数
        self.loss = nn.CrossEntropyLoss()

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, attention_mask=None, y=None):
        # x shape: (batch_size, seq_len)
        # attention_mask: 用于忽略 padding 的位置

        # 获取 BERT 的输出
        # last_hidden_state shape: (batch_size, seq_len, hidden_size)
        outputs = self.bert(input_ids=x, attention_mask=attention_mask)
        x = outputs.last_hidden_state

        # 应用 dropout
        x = self.dropout(x)

        # 预测 logits
        y_pred = self.classify(x)  # shape: (batch_size, seq_len, vocab_size)

        if y is not None:
            # 计算损失，将输出展平
            loss = self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
            return loss
        else:
            # 返回概率分布
            return torch.softmax(y_pred, dim=-1)


# 加载 tokenizer（替代原来的 build_vocab）
def build_tokenizer(model_path):
    """
    加载 BERT 的 tokenizer
    """
    tokenizer = BertTokenizer.from_pretrained(model_path)
    return tokenizer


# 加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:  # 建议统一使用 utf8 编码
        for line in f:
            corpus += line.strip()
    return corpus


# 构建训练样本（使用 tokenizer 编码）
def build_sample(tokenizer, window_size, corpus, all_special_tokens=True):
    # 随机选择起始位置
    start = random.randint(0, len(corpus) - window_size - 1)
    text = corpus[start:start + window_size + 1]  # 多取一个字符用于目标

    # 使用 tokenizer 编码
    # 添加 special tokens 如 [CLS], [SEP] 可选，但在自回归语言模型中通常不加，或仅用于句子边界
    encoded = tokenizer(text,
                        truncation=True,
                        max_length=window_size + 1,
                        return_tensors=None,
                        add_special_tokens=all_special_tokens)

    input_ids = encoded['input_ids']

    # 输入是前 n 个 token，目标是后 n 个 token（错位）
    if len(input_ids) > window_size + 1:
        input_ids = input_ids[:window_size + 1]

    # 截断或填充
    if len(input_ids) < window_size + 1:
        # 填充
        padding_length = window_size + 1 - len(input_ids)
        input_ids = input_ids + [tokenizer.pad_token_id] * padding_length

    # 输入：前 window_size 个 token
    x = input_ids[:window_size]
    # 目标：从第2个开始的 window_size 个 token（错位预测）
    y = input_ids[1:window_size + 1]

    return x, y


# 建立数据集
def build_dataset(sample_length, tokenizer, window_size, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(tokenizer, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 建立模型
def build_model(tokenizer, model_path):
    # 加载 BERT 模型路径下的配置和权重
    model = LanguageModel(model_path, len(tokenizer))
    return model


# 文本生成测试代码
def generate_sentence(openings, model, tokenizer, window_size):
    model.eval()
    with torch.no_grad():
        # 编码输入
        tokens = tokenizer.tokenize(openings)
        input_ids = tokenizer.convert_tokens_to_ids(tokens[-window_size:])

        while len(tokens) < 30:  # 最大生成长度
            x = torch.LongTensor([input_ids])
            if torch.cuda.is_available():
                x = x.cuda()

            # 注意：BERT 模型在生成时通常不使用 auto-regressive mask
            # 但在自回归语言模型训练中，我们假设模型是 causal 的
            # 这里简化处理，实际中可能需要使用 causal mask 或使用 GPT 类模型
            y_pred = model(x)

            # 获取最后一个 token 的预测
            last_pred = y_pred[0, -1, :]  # shape: (vocab_size,)

            # 采样策略
            index = sampling_strategy(last_pred)

            # 转换为 token
            pred_token = tokenizer.convert_ids_to_tokens([index])[0]

            # 如果生成了结束符或特殊符号，可以提前结束
            if pred_token in ['[SEP]', '[CLS]', '</s>']:
                break

            tokens.append(pred_token)
            input_ids.append(index)

            if pred_token == '\n':
                break

        return openings + ''.join(tokens[len(tokenizer.tokenize(openings)):])


def sampling_strategy(prob_distribution):
    # 简单的采样策略
    prob_distribution = prob_distribution.cpu().numpy()
    prob_distribution = np.exp(prob_distribution) / np.sum(np.exp(prob_distribution))  # softmax
    index = np.random.choice(len(prob_distribution), p=prob_distribution)
    return int(index)


def train(corpus_path, bert_model_path="bert-base-chinese", save_weight=True):
    epoch_num = 20
    batch_size = 16  # BERT 显存占用大，减小 batch size
    train_sample = 10000
    window_size = 128  # BERT 支持更长序列

    # 加载 tokenizer
    tokenizer = build_tokenizer(bert_model_path)

    # 加载语料
    corpus = load_corpus(corpus_path)

    # 建立模型
    model = build_model(tokenizer, bert_model_path)

    if torch.cuda.is_available():
        model = model.cuda()

    # 优化器
    optim = torch.optim.Adam(model.parameters(), lr=2e-5)  # BERT 微调通常使用小学习率

    print("BERT 模型和 tokenizer 加载完毕，开始训练")

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []

        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, tokenizer, window_size, corpus)

            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()

            optim.zero_grad()
            loss = model(x, y=y)
            loss.backward()
            optim.step()

            watch_loss.append(loss.item())

        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))

        # 生成测试
        print(generate_sentence("让他在半年之前，就不能做出", model, tokenizer, window_size))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, tokenizer, window_size))

    if save_weight:
        model_path = os.path.join(r"D:\AI  Learning\课程内容\TEN_Week\TENWEEK\lstm语言模型生成文本", "bert_lm.pth")
        torch.save(model.state_dict(), model_path)
        print(f"模型权重已保存至 {model_path}")


if __name__ == "__main__":
    train("corpus.txt", bert_model_path=r"D:\ANACONDA\envs\py312\Lib\site-packages\modelscope\hub\models\google-bert\bert-base-chinese", save_weight=False)