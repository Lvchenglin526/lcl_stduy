def get_stats(ids):  # 将字符串中相邻两个字的词取出
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1  # 有则有的数量加一，没有则加一
    return counts

def merge(ids, pair, idx):  # 按照pair的大小限制取代数
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

def decoding(vocab, ids):  # 解码器
    # given ids (list of integers), return Python string
    tokens = b"".join(vocab[idx] for idx in ids)  # 按照字表映射回文本
    text = tokens.decode("utf-8", errors="replace")
    return text

def encoding(merges, text):  # 编码器
    # given a string, return list of integers (the tokens)
    tokens = list(text.encode("utf-8"))  # 自带的编码器编码
    while len(tokens) >= 2:
        stats = get_stats(tokens)  # 两两计数
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))  # 设置最大扩容
        if pair not in merges:
            break  # nothing else can be merged
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens

def main():
    text_1 = ("两人一高一矮，一胖一瘦，皆是身穿淡青色皂吏服，一路走到乱葬岗前，将一个破草席放下，其中的高个子长舒了口气，说道：“终于到了。”")
    # Convert text to a list of integers (byte values)
    tokens = list(text_1.encode("utf-8"))

    vocab_size = 276  # Desired final vocabulary size
    num_merges = vocab_size - 256
    ids = list(tokens)  # Copy so we don't destroy the original list

    merges = {}  # (int, int) -> int
    for i in range(num_merges):
        stats = get_stats(ids)
        pair = max(stats, key=stats.get)
        idx = 256 + i
        ids = merge(ids, pair, idx)
        merges[pair] = idx

    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]

    # Decode the tokenized ids back to text
    print(f"Text: {text_1}")  # 序列化后的结果
    print(f"Ids: {ids}")  # 序列化后的结果
    decoded_text = decoding(vocab, ids)  # 解码
    print(f"Decoded Text: {decoded_text}")

    # Encode the text again to verify
    encoded_ids = encoding(merges, decoded_text)  # 再编码
    print(f"Encoded IDs: {encoded_ids}")
    print(ids == encoded_ids)

if __name__ == "__main__":
    main()


