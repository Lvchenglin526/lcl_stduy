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
1.实现基于kmeans结果类内距离的排序
欧几里得距离越短，说明越属于这个簇
2.打印距离质心最近的前十个
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
    diff = defaultdict(dict)
    Descending_order = defaultdict(list)
    for sentence, label, vector in zip(sentences, kmeans.labels_, vectors):  # 取出句子和每个句子属于哪一个簇
        centers = kmeans.cluster_centers_  # 每个聚类中心坐标
        sentence_label_dict[label].append(sentence)  # 同标签的放到一起，句子属于哪一簇就放到哪一个里面去
        distances_sentence = distance(vector, centers[label])  # 算出欧几里得距离
        Difflist[label].append(distances_sentence)  # 同标签的句子向量与该标签的质心坐标算欧几里得距离
        diff[label][distances_sentence] = sentence.replace(' ', '')  # 将句子和其到质心的欧几里得距离以字典的形式存储
    for label, dif in diff.items():  # 对每个簇中字典的key值排序（升序）
        up_dif = dict(sorted(dif.items()))  # 升序排序
        Descending_order[label].append(up_dif)
    for label_key, label_value in Descending_order.items():  # 遍历排列后每个簇的字典
        print("cluster : %d" % label_key)
        for sentences in label_value:  # 打印欧几里得距离距质心最小的前十个
            sent = list(dict(sentences).values())
            for i in range(min(10, len(sent))):
                print(sent[i])
        print("---------")


if __name__ == "__main__":
    main()

