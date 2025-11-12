#!/usr/bin/env python3  
# coding: utf-8

# 基于训练好的词向量模型进行聚类
# 聚类采用Kmeans算法
import math
# import re
# import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

# 输入模型文件路径
# 加载训练好的模型
'''
1.计算每个簇的每个句子到质心的欧几里得距离
2.算出每个簇中所有欧几里得距离的均值
3.用这个距离代表这个簇的聚合度
4.按照簇的聚合度打印其中的内容，每个簇只打印十条内容
'''

def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

def load_sentence(path):
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    return sentences

# 将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  # sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)  # 生成全零的多维向量
        # 所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                # 部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)

def distance(p1, p2):
    #计算两点间距
    tmp = 0
    for i in range(len(p1)):
        tmp += pow(p1[i] - p2[i], 2)  # 所有标量进行对位
    return pow(tmp, 0.5)


def main():
    model = load_word2vec_model(r"model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)   # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)          # 进行聚类计算

    sentence_label_dict = defaultdict(list)
    Difflist = defaultdict(list)
    Descending_order = defaultdict(list)
    for sentence, label, vector in zip(sentences, kmeans.labels_, vectors):  # 取出句子和每个句子属于哪一个簇
        centers = kmeans.cluster_centers_  # 每个聚类中心坐标
        sentence_label_dict[label].append(sentence)  # 同标签的放到一起，句子属于哪一簇就放到哪一个里面去
        Difflist[label].append(distance(vector, centers[label]))  # 同标签的句子向量与该标签的质心坐标算欧几里得距离
    for label, diff in Difflist.items():
        avg = sum(diff) / len(diff)
        Descending_order[label].append(avg)
    sorted_Descending_order = dict(sorted(Descending_order.items(), key=lambda item: item[1]))
    for label_key, label_value in sorted_Descending_order.items():  # 遍历升序排列后的键值对
        print("cluster : %d" % label_key)
        for sentences in range(min(10, len(sentence_label_dict[label_key]))):  # 随便打印几个，太多了看不过来
            print(sentence_label_dict[label_key][sentences].replace(' ', ''))
        print("---------")


if __name__ == "__main__":
    main()

