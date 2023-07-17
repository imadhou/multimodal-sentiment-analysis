import torch
from torch import nn
import os
import pandas as pd
from gensim.models import Word2Vec

dir = os.path.dirname(__file__)
word2vec_path = os.path.join(dir, '..', '..', 'data', 'models', 'WORD2VEC.model')

class TextModel(nn.Module):
    def __init__(self, NUM_CLASSES, is_for_multimodal=False):
        super(TextModel, self).__init__()

        w2v = Word2Vec.load(word2vec_path)

        EMBEDDING_MATRIX = w2v.wv.vectors

        self.is_for_multimodal = is_for_multimodal
        self.HIDDEN_SIZE = 50
        vocab_size, embedding_dim = EMBEDDING_MATRIX.shape
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(EMBEDDING_MATRIX), freeze=True)
        self.dropout = nn.Dropout(0.5)
        self.lstm1 = nn.LSTM(embedding_dim, self.HIDDEN_SIZE, batch_first=True, bidirectional=True)
        if not self.is_for_multimodal:
            self.fc = nn.Linear(self.HIDDEN_SIZE * 2, NUM_CLASSES)
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        output1, _ = self.lstm1(embedded)
        last_hidden_state = torch.cat((output1[:, -1, :self.HIDDEN_SIZE], output1[:, 0, self.HIDDEN_SIZE:]), dim=1)
        if self.is_for_multimodal:
            return last_hidden_state
        logits = self.fc(last_hidden_state)
        logits = self.softmax(logits)
        return logits