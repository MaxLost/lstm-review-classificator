import pandas as pd
import torch
import random
import string
import spacy
import re
from sklearn.model_selection import train_test_split
from collections import Counter


class DataLoader:
    def __init__(self, input_size, path):
        self.input_size = input_size
        self.path = path
        self.tok = spacy.load("en_core_web_sm")
        self.vocab = {"":0, "UNK":1}
        return

    def load_data(self):
        data = pd.read_csv(self.path)
        data['Title'] = data['Title'].fillna('')
        data['Review Text'] = data['Review Text'].fillna('')
        data['review'] = data['Title'] + ' ' + data['Review Text']
        data = data[['Rating', 'review']]
        data.columns = ['mark', 'text']
        data['mark'] = data['mark'].apply(lambda x: x-1)
        self.create_vocab(data)
        return data

    def tokenize(self, text):
        text = re.sub(r"[^\x00-\x7F]+", " ", text)
        regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
        nopunct = regex.sub(" ", text.lower())
        return [token.text for token in self.tok.tokenizer(nopunct)]

    def create_vocab(self, data):
        counts = Counter()
        for index, row in data.iterrows():
            counts.update(self.tokenize(row['text']))

        words = ["", "UNK"]
        for word in counts:
            self.vocab[word] = len(words)
            words.append(word)
        return

    def get_sample(self, data):
        sample = data.loc[random.randint(1, len(data.index)-1)]
        sample = sample.values.tolist()
        sample[1] = self.tokenize(sample[1])
        return sample

    def encode_text(self, text):
        encoded = torch.zeros(self.input_size, dtype=torch.int)
        tmp = torch.tensor([self.vocab[word] for word in text])
        length = min(self.input_size, len(tmp))
        encoded[:length] = tmp[:length]
        return encoded

    def create_batch(self, data, batch_size):
        marks, text = [], []
        for i in range(batch_size):
            sample = self.get_sample(data)
            text.append(self.encode_text(sample[1]))
            marks.append(torch.tensor(sample[0]))

        text_train, text_valid, marks_train, marks_valid = train_test_split(text, marks, test_size=0.2)
        train_batch = [marks_train, text_train]
        valid_batch = [marks_valid, text_valid]
        return train_batch, valid_batch
