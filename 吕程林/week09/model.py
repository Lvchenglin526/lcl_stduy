# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from TorchCRF import CRF
from transformers import BertModel
"""
建立网络模型结构
"""

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        # vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        class_num = config["class_num"]  # 输出特征维度
        num_layers = config["num_layers"]
        # self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.layer = BertModel.from_pretrained(r"D:\ANACONDA\envs\py312\Lib\site-packages\modelscope\hub\models\google-bert\bert-base-chinese", return_dict=False)
        self.classify = nn.Linear(hidden_size, class_num)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.crf_layer = CRF(class_num, batch_first=True)
        self.use_crf = config["use_crf"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  #loss采用交叉熵损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target=None):
        # x = self.embedding(x)  #input shape:(batch_size, sen_len)
        sentence_output, pooling_output = self.layer(x)  # input|output shape:(batch_size, max_length, hidden_size)
        # print(pooling_output.shape)
        predict = self.classify(sentence_output)
        predicts = self.softmax(predict)
        # print(predict.shape)
        # input:(batch_size, sen_len, hidden_size) -> (batch_size, sen_len, class_num)

        if target is not None:
            if self.use_crf:
                mask = target.gt(-1)  # 屏蔽无需参与训练的预测值
                # gt 是 "greater than"（大于）的缩写
                # target.gt(-1) 的意思是：检查 target 张量中的每一个元素是否大于 -1
                # 如果 target 中的某个元素大于 -1，对应位置的 mask 值为 True
                return - self.crf_layer(predicts, target, mask, reduction="mean")
            else:
                # (number, class_num), (number)
                return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
        else:
            if self.use_crf:
                return self.crf_layer.decode(predicts)
            else:
                return predicts


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config
    model = TorchModel(Config)
    # x = torch.randn(2,3,768)
    # y = model(x)