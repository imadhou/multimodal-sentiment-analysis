from torch.utils.data import Dataset
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import torch
import os
import pandas as pd
dir = os.path.dirname(__file__)
csv_file_path = os.path.join(dir, '..', '..', 'data', 'processed', 'text.csv')
image_path_prefix = os.path.join(dir, '..', '..', 'data', 'processed', 'text')
word2vec_path = os.path.join(dir, '..', '..', 'data', 'models', 'WORD2VEC.model')


class TextDataset(Dataset):
    def __init__(self, text_column, label_column):

        self.dataframe = pd.read_csv(csv_file_path)
        self.word2vec_model = Word2Vec.load(word2vec_path)

        self.labels = self.dataframe[label_column].tolist()
        self.SEQUENCE_LENGTH = 100
        self.TEXT_COL = text_column

        self.data = self._preprocess_text()


    def _preprocess_text(self):
        preprocessed_sequences = []
        for idx in range(len(self.dataframe)):
            text = self.dataframe.loc[idx, self.TEXT_COL]
            tokens = word_tokenize(text)
            sequence_indices = [self.word2vec_model.wv.key_to_index[word] for word in tokens if word in self.word2vec_model.wv]
            padded_sequence = torch.tensor(sequence_indices + [0] * (self.SEQUENCE_LENGTH - len(sequence_indices)))[:self.SEQUENCE_LENGTH]
            preprocessed_sequences.append(padded_sequence)
        return preprocessed_sequences
    
    def __len__(self):
        return len(self.data)

        
    def __getitem__(self, index):

        data = self.data[index]
        label = self.labels[index]
        
        return data, label