import csv
import json
from sklearn.model_selection import train_test_split


def read_csv(path):
    # with open('../data/shuju.json', 'w', encoding='utf-8') as
    labels = []
    reviews = []
    with open(path, "r", newline='',  encoding='utf-8') as f:
        read = csv.DictReader(f)
        for i in read:
            # json.dump(i, save, ensure_ascii=False)
            # save.write(str(i))
            labels.append(int(i['label']))
            reviews.append(i['review'])
    # print(labels)
    return labels, reviews

def save_json(label, review, path):
    with open(path, 'w', encoding='utf-8') as save:
        for label, review in zip(label, review):
            json.dump({'label': label, 'review': review}, save, ensure_ascii=False)
            save.write('\n')

def split_data_and_save(train_path, test_path, test_size=0.2, random_state=42):
    labels, reviews = read_csv('../data/文本分类练习.csv')
    X_train, X_val, y_train, y_val = train_test_split(labels, reviews, test_size=test_size, random_state=random_state)
    # print(X_train, X_val, y_train, y_val)
    save_json(X_train, y_train, train_path)
    save_json(X_val, y_val, test_path)

if __name__ == "__main__":
    train_path = '../data/train_tag_news.json'
    test_path = '../data/test_tag_news.json'
    split_data_and_save(train_path, test_path)