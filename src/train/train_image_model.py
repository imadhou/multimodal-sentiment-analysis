if __name__ == '__main__' and  __package__ is None:
    import sys
    from os import path
    sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from sklearn.model_selection import train_test_split
from models import ImageModel
from dataset_generators import ImageDataset
import utils as ut
import os
import numpy as np
from torch.utils.data import Subset, DataLoader
from torchviz import make_dot
import torch
import torch.nn as nn


MODEL_NAME = "IMAGES_MODEL"
IMAGE_SHAPE = 224
IMAGE_COL = "local_path"
LABEL_COL = "label"
LR = 0.001
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 228
EPOCHS = 20
NUM_CLASSES = 3

data = ImageDataset(IMAGE_SHAPE, IMAGE_COL, LABEL_COL)
indices = np.arange(len(data))

train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)

train_data = Subset(data, train_indices)
test_data = Subset(data, test_indices)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

model = ImageModel(NUM_CLASSES, IMAGE_SHAPE).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)


print()
print(" ############# MODEL SUMMARY ############# ")
print()
print(model)
print()
print(" ############ ############# ############ ")
print()

ut.train(model, train_loader, test_loader, EPOCHS, MODEL_NAME, optimizer, criterion, DEVICE, False)












# text, _ = next(iter(train_loader))
# output = model(text.to(DEVICE))

# archi = make_dot(output, params=dict(model.named_parameters()))

# archi.save("images_model_archi", "C:/Users/MourtadaHouari/Desktop/sentiment-analysis/src/images")
# archi.render('C:/Users/MourtadaHouari/Desktop/sentiment-analysis/src/images/images_model_archi', format='png')
