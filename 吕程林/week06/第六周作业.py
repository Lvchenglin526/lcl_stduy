import torch
import torch.nn as nn
import torch.nn.functional as F

'''
用pytorch实现Bert预训练模型
'''


class MultiHeadSelfAttention(nn.Module):  # 多头机制
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        assert hidden_size % num_heads == 0  # 用隐藏层除以头数，确保可以整除
        self.hidden_size = hidden_size  # 隐藏层维度
        self.num_heads = num_heads  # 头数
        self.head_dim = hidden_size // num_heads  # 用隐藏层除以头数，计算头维度

        self.q_linear = nn.Linear(hidden_size, hidden_size)  # 输入过线性层得到Q
        self.k_linear = nn.Linear(hidden_size, hidden_size)  # 输入过线性层得到K
        self.v_linear = nn.Linear(hidden_size, hidden_size)  # 输入过线性层得到V
        self.out_linear = nn.Linear(hidden_size, hidden_size)  # 过线性层得到输出

        self.dropout = nn.Dropout(dropout)  # dropout用于神经元随机丢弃

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)  # 输入的第一个维度也就是样本数

        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # QKV的形状变化过程如下
        # 经过线形层的形状为(batch_size, seq_len, embed_size)
        # 经过view重塑后的形状为(batch_size, seq_len, num_heads, head_dim)(-1 自动推断为序列长度 seq_len)
        # 经过transpose转换位置后(batch_size, num_heads, seq_len, head_dim)
        '''
        import torch
        # 定义张量 A 和 B
        A = torch.randn(2, 12, 10, 64)
        B = torch.randn(2, 12, 64, 10)
        # 使用 torch.matmul 进行矩阵乘法
        result = torch.matmul(A, B)
        # 打印结果形状
        print(result.shape)  # 输出: torch.Size([2, 12, 10, 10])
        '''

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # 自注意力得分


        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_probs = F.softmax(scores, dim=-1)  # 过softmax，将结果限制在0到1之间
        attention_probs = self.dropout(attention_probs)

        context = torch.matmul(attention_probs, V).transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        # 计算softmax后乘V，获得最终的Z，重塑形状为最开始的输入形状
        output = self.out_linear(context)  # 过线性层后输出结果

        return output


class PositionwiseFeedForward(nn.Module):  # feedforward部分
    def __init__(self, hidden_size, intermediate_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)  # gelu内线性层
        self.fc2 = nn.Linear(intermediate_size, hidden_size)  # gelu外线性层
        self.dropout = nn.Dropout(dropout)  # dropout层
        self.gelu = nn.GELU()  # 激活gelu

    def forward(self, x):
        x = self.fc1(x)  # 过gelu内线性层
        x = self.gelu(x)  # 过gelu
        x = self.dropout(x)  # 过dropout随机抛弃
        x = self.fc2(x)  # 过gelu外线性层后就是最终输出
        return x


class EncoderLayer(nn.Module):  # SelfAttention实现
    def __init__(self, hidden_size, num_heads, intermediate_size, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadSelfAttention(hidden_size, num_heads, dropout)  # 多头机制
        self.feed_forward = PositionwiseFeedForward(hidden_size, intermediate_size, dropout)  # feedforward
        self.norm1 = nn.LayerNorm(hidden_size)  # normalization
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        attn_output = self.self_attn(src, src, src, src_mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.norm1(src + attn_output)

        ff_output = self.feed_forward(out1)
        ff_output = self.dropout2(ff_output)
        out2 = self.norm2(out1 + ff_output)

        return out2


class BERTModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, intermediate_size, max_position_embeddings=512,
                 dropout=0.1):
        super(BERTModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)  # token_embedding
        self.position_encoding = nn.Parameter(torch.zeros(1, max_position_embeddings, hidden_size))
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(hidden_size, num_heads, intermediate_size, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask=None):
        seq_len = input_ids.size(1)
        pos_enc = self.position_encoding[:, :seq_len, :]

        embeddings = self.embedding(input_ids) + pos_enc
        embeddings = self.dropout(embeddings)
        # print(embeddings.shape,'a') barch_size*length*hidden_size

        for encoder_layer in self.encoder_layers:
            embeddings = encoder_layer(embeddings, attention_mask)

        outputs = self.norm(embeddings)
        return outputs


# 参数设置
vocab_size = 30000  # 假设词汇表大小为30000
hidden_size = 768  # 隐藏层大小
num_layers = 12  # 编码器层数
num_heads = 12  # 多头注意力头数
intermediate_size = 3072  # 前馈网络中间层大小
max_position_embeddings = 512  # 最大位置嵌入长度

# 创建BERT模型实例
model = BERTModel(vocab_size, hidden_size, num_layers, num_heads, intermediate_size, max_position_embeddings)

batch_size = 2  # 样本数
sequence_length = 10  # 样本长度
input_ids = torch.randint(0, vocab_size, (batch_size, sequence_length))
# input_ids = 生成的样本为batch_size*sequence_length，其中的元素都是0到vocab_size之间

# 前向传播
outputs = model(input_ids)
print("Input IDs Shape:", input_ids.shape)
print("Outputs Shape:", outputs.shape)

'''
Input IDs Shape: torch.Size([2, 10])
Outputs Shape: torch.Size([2, 10, 768])
'''
