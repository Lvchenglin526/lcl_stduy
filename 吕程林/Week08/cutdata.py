import json
import random
import torch
from collections import defaultdict
from config import Config


class ReadData:
    def __init__(self, data_path):
        self.path = data_path
        self.schema = load_schema(Config["schema_path"])

    def load_data_cut(self):
        self.know = defaultdict(list)
        with open(self.path, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                questions = line['questions']
                labels = line['target']
                self.know[self.schema[labels]].append(questions)
            return self.know

    def train_sample(self):
        question_index = list(self.know.keys())
        p, n = random.sample(question_index, 2)
        if len(self.know[p]) < 2:
            return self.train_sample()
        else:
            sentence1, sentence2 = random.sample(self.know[p], 2)
            sentence3 = random.choice(self.know[n])
            return [sentence1, sentence2, sentence3, torch.LongTensor(p)]

def load_schema(path):
    with open(path, encoding='utf-8') as f:
        return json.loads(f.read())


if __name__ == "__main__":
    data = ReadData("../data/data.json")
    data.load_data_cut()