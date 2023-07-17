if __name__ == '__main__' and  __package__ is None:
    import sys
    from os import path
    sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from sklearn.model_selection import train_test_split
from models import ImageModel, TextModel, MultimodalModel
from dataset_generators import MultimodalDataset
import utils as ut
import numpy as np
from torch.utils.data import Subset, DataLoader
import torch
import torch.nn as nn


MODEL_NAME = "MULTIMODAL_MODEL"
IMAGE_SHAPE = 224
TEXT_COL = "text_emo"
LABEL_COL = "label"
IMAGE_COL = "local_path"
NUM_CLASSES = 3
BATCH_SIZE = 258
LR = 0.001
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
EPOCHS = 20



data = MultimodalDataset(IMAGE_SHAPE, TEXT_COL, IMAGE_COL, LABEL_COL)

indices = np.arange(len(data))
train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)

train_data = Subset(data, train_indices)
test_data = Subset(data, test_indices)


# create data loaders for train and test sets
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)



text_model = TextModel(NUM_CLASSES, True).to(DEVICE)
image_model = ImageModel(NUM_CLASSES, IMAGE_SHAPE, True).to(DEVICE)

model = MultimodalModel(text_model, image_model, NUM_CLASSES).to(DEVICE)

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

ut.train(model, train_loader, test_loader, EPOCHS, MODEL_NAME, optimizer, criterion, DEVICE, True)



