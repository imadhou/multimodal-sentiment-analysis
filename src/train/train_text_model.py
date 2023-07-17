if __name__ == '__main__' and  __package__ is None:
    import sys
    from os import path
    sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from dataset_generators import TextDataset
from models import TextModel
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import utils as ut
import numpy as np

# Data set

MODEL_NAME = "RETRAINED_TEXT_MODEL"
TEXT_COL = "text_emo"
LABEL_COL = "label"
NUM_CLASSES = 3

BATCH_SIZE = 1024
LR = 0.001
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 2048
EPOCHS = 20

data = TextDataset(TEXT_COL, LABEL_COL)
indices = np.arange(len(data))

# split
train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
train_data = Subset(data, train_indices)
test_data = Subset(data, test_indices)

# loader
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)


# model
model = TextModel(NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
model



print()
print(" ############# MODEL SUMMARY ############# ")
print()
print(model)
print()
print(" ############ ############# ############ ")
print()

ut.train(model, train_loader, test_loader, EPOCHS, MODEL_NAME, optimizer, criterion, DEVICE, False)



