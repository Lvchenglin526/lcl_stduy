import json
import pandas as pd
import re
import os


class DialogueSystem:
    def __init__(self):
        self.load()
        self.last_memory = {}

    def load(self):
        self.all_node_info = {}
        self.load_scenario("scenario-买衣服.json")
        self.load_scenario("scenario-看电影.json")
        self.load_slot_templet("slot_fitting_templet.xlsx")

    def load_scenario(self, file):
        with open(file, 'r', encoding='utf-8') as f:
            scenario = json.load(f)
        scenario_name = os.path.basename(file).split('.')[0]
        for node in scenario:
            self.all_node_info[scenario_name + "_" + node['id']] = node
            if "childnode" in node:
                self.all_node_info[scenario_name + "_" + node['id']]['childnode'] = [scenario_name + "_" + x for x in
                                                                                     node['childnode']]

    def load_slot_templet(self, file):
        self.slot_templet = pd.read_excel(file)
        self.slot_info = {}
        for i in range(len(self.slot_templet)):
            slot = self.slot_templet.iloc[i]['slot']
            query = self.slot_templet.iloc[i]['query']
            values = self.slot_templet.iloc[i]['values']
            if slot not in self.slot_info:
                self.slot_info[slot] = {}
            self.slot_info[slot]['query'] = query
            self.slot_info[slot]['values'] = values

    def nlu(self, memory):
        memory = self.intent_judge(memory)
        memory = self.slot_filling(memory)
        return memory

    def intent_judge(self, memory):
        query = memory['query']
        max_score = -1
        hit_node = None
        for node in memory["available_nodes"]:
            score = self.calucate_node_score(query, node)
            if score > max_score:
                max_score = score
                hit_node = node
        memory["hit_node"] = hit_node
        memory["intent_score"] = max_score
        return memory

    def calucate_node_score(self, query, node):
        node_info = self.all_node_info[node]
        intent = node_info['intent']
        max_score = -1
        for sentence in intent:
            score = self.calucate_sentence_score(query, sentence)
            if score > max_score:
                max_score = score
        return max_score

    def calucate_sentence_score(self, query, sentence):
        query_words = set(query)
        sentence_words = set(sentence)
        intersection = query_words.intersection(sentence_words)
        union = query_words.union(sentence_words)
        return len(intersection) / len(union)

    def slot_filling(self, memory):
        hit_node = memory["hit_node"]
        node_info = self.all_node_info[hit_node]
        for slot in node_info.get('slot', []):
            if slot not in memory:
                slot_values = self.slot_info[slot]["values"]
                if re.search(slot_values, memory['query']):  # 修复：使用 memory['query'] 替代未定义的 query
                    memory[slot] = re.search(slot_values, memory['query']).group()
        return memory

    def dst(self, memory):
        hit_node = memory["hit_node"]
        node_info = self.all_node_info[hit_node]
        slot = node_info.get('slot', [])
        for s in slot:
            if s not in memory:
                memory["require_slot"] = s
                return memory
        memory["require_slot"] = None
        return memory

    def dpo(self, memory):
        if memory["require_slot"] is None:
            memory["policy"] = "reply"
            hit_node = memory["hit_node"]
            node_info = self.all_node_info[hit_node]
            memory["available_nodes"] = node_info.get("childnode", [])
        else:
            memory["policy"] = "request"
            memory["available_nodes"] = [memory["hit_node"]]
        return memory

    def nlg(self, memory):
        if memory["policy"] == "reply":
            hit_node = memory["hit_node"]
            node_info = self.all_node_info[hit_node]
            memory["response"] = self.fill_in_slot(node_info["response"], memory)
        else:
            slot = memory["require_slot"]
            memory["response"] = self.slot_info[slot]["query"]
        return memory

    def fill_in_slot(self, template, memory):
        node = memory["hit_node"]
        node_info = self.all_node_info[node]
        for slot in node_info.get("slot", []):
            if slot in memory:  # 修复：增加判断防止 KeyError
                template = template.replace(slot, memory[slot])
        return template

    def run(self, query, memory):
        '''
        query: 用户输入
        memory: 用户状态
        '''
        memory["query"] = query

        repeat_keywords = ["没听清", "没听见", "重复", "再说一遍", "重新说", "没听懂", "什么"]

        if any(keyword in query for keyword in repeat_keywords) and self.last_memory:
            memory["response"] = self.last_memory.get("response", "抱歉，我没有上一轮的问题可以重复。")
            if "policy" in self.last_memory:
                memory["policy"] = self.last_memory["policy"]
            if "require_slot" in self.last_memory:
                memory["require_slot"] = self.last_memory["require_slot"]
            if "hit_node" in self.last_memory:
                memory["hit_node"] = self.last_memory["hit_node"]
            if "available_nodes" in self.last_memory:
                memory["available_nodes"] = self.last_memory["available_nodes"]

            # 不执行标准的 nlu->dst->dpo->nlg 流程
            return memory

        # 标准流程，用户不是要求重复，则执行正常的对话逻辑
        memory = self.nlu(memory)
        memory = self.dst(memory)
        memory = self.dpo(memory)
        memory = self.nlg(memory)

        self.last_memory = memory.copy()

        return memory


if __name__ == '__main__':
    ds = DialogueSystem()
    memory = {"available_nodes": ["scenario-买衣服_node1", "scenario-看电影_node1"]}

    print("系统启动！你可以输入你的要求。没听清可以输入'没听清来告诉我'。输入 '退出' 结束对话。")

    while True:
        query = input("\n用户: ")
        if query == "退出":
            break

        memory = ds.run(query, memory)
        print("系统:", memory.get('response', '...'))
