import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

class ImageModel(nn.Module):
    def __init__(self, NUM_CLASSES, IMG_SHAPE, is_for_multimodal=False):
        super(ImageModel, self).__init__()

        self.NUM_CLASSES = NUM_CLASSES
        self.IMG_SHAPE = IMG_SHAPE
        self.is_for_multimodal = is_for_multimodal

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 14 * 14, 512)
        self.relu5 = nn.ReLU()
        if not self.is_for_multimodal:
            self.fc2 = nn.Linear(512, 3)
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu5(x)
        if not self.is_for_multimodal:
            x = self.fc2(x)
            x = self.softmax(x)
        return x

        # # Move tensor to CPU and convert to a numpy array
        # feature_maps = x.detach().cpu().numpy()
        # feature_maps = np.transpose(feature_maps, (0, 2, 3, 1))  # Rearrange dimensions for visualization
        
        # # Display each channel in the feature maps
        # for channel in range(feature_maps.shape[-1]):
        #     plt.imshow(feature_maps[0, :, :, channel], cmap='gray')  # Assuming batch size = 1
        #     plt.show()