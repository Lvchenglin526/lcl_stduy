# 词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
Dict = {"经常":0.1,
        "经":0.05,
        "有":0.1,
        "常":0.001,
        "有意见":0.1,
        "歧":0.001,
        "意见":0.2,
        "分歧":0.2,
        "见":0.05,
        "意":0.05,
        "见分歧":0.05,
        "分":0.1}

# 待切分文本
sentence = "经常有意见分歧"

# 目标输出;顺序不重要
target = [
    ['经常', '有意见', '分歧'],  #
    ['经常', '有意见', '分', '歧'],  #
    ['经常', '有', '意见', '分歧'],
    ['经常', '有', '意见', '分', '歧'],
    ['经常', '有', '意', '见分歧'],
    ['经常', '有', '意', '见', '分歧'],  #
    ['经常', '有', '意', '见', '分', '歧'],
    ['经', '常', '有意见', '分歧'],  #
    ['经', '常', '有意见', '分', '歧'],
    ['经', '常', '有', '意见', '分歧'],
    ['经', '常', '有', '意见', '分', '歧'],
    ['经', '常', '有', '意', '见分歧'],
    ['经', '常', '有', '意', '见', '分歧'],
    ['经', '常', '有', '意', '见', '分', '歧']
]

# 实现全切分函数，输出根据字典能够切分出的所有的切分方式
# def all_cut(sentence, Dict):
#     target = []
#     for i in range(len(sentence)):
#         if sentence[i] in Dict:
#             target.append(sentence[i])
#         else:
#             target = 0
#     return target


def all_cut(sentence, Dict):
    vocab = set(Dict.keys())
    max_len = max(len(word) for word in vocab)  # 取出字典中最长字
    n = len(sentence)  # 统计字符串长度

    # 初始化动态规划表：dp[i] 表示从位置i到最长字符间隔的所有切分方式
    dp = [[] for _ in range(n + 1)]  # 8个空列表
    dp[-1] = [[]]  # 空字符串的切分方式为空列表

    # 从右向左填充dp表
    for i in range(n - 1, -1, -1):  # i=6,5,4,3,2,1,0
        max_j = min(i + max_len, n)  # 设置最长边界
        for j in range(i + 1, max_j + 1):  # word从末尾往前开始取词
            word = sentence[i:j]  # 将字符切分切换成最长字符长度全切分
            # print(word)
            if word in vocab:
                for rest in dp[j]:  # 合并当前词与后续切分结果
                    # print(rest, 'rest')
                    dp[i].append([word] + rest)
                    # print(dp[i], 'dp[i]')
    return dp[0]

def main():
    x = all_cut(sentence, Dict)
    for i in x:
        print(i)

if __name__ == '__main__' :
    main()


