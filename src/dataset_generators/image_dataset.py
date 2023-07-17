import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os

dir = os.path.dirname(__file__)
csv_file_path = os.path.join(dir, '..', '..', 'data', 'processed', 'train_images.csv')
image_path_prefix = os.path.join(dir, '..', '..', 'data', 'processed', 'train_images', 't4sa', '')


class ImageDataset(Dataset):
    def __init__(self, IMG_SHAPE, IMAGE_COL, LABEL_COL):
        self.IMG_SHAPE = IMG_SHAPE
        self.IMAGE_COL = IMAGE_COL
        self.LABEL_COL = LABEL_COL
        self.data = pd.read_csv(csv_file_path).head(10000)

        self.transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = self.data.loc[idx, self.IMAGE_COL][1:].split('/')[-1]
        image = Image.open(os.path.join(dir, image_path_prefix, image_path)).convert('RGB')
        image = image.resize((self.IMG_SHAPE, self.IMG_SHAPE))
        image = self.transform(image)

        label = self.data.loc[idx, self.LABEL_COL]
        return image, label
    
    # def plot_model_archi(graphviz_path):
    #     os.environ["PATH"] += os.pathsep + graphviz_path
    #     data = ImageDataset(LABELS_DATA_PATH)
    #     indices = np.arange(len(data))
    #     train_indices, _ = train_test_split(indices, test_size=0.2, random_state=42)
    #     train_data = Subset(data, train_indices)
    #     train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
    #     model = ImageClassifier().to(DEVICE)
    #     criterion = nn.CrossEntropyLoss()
    #     optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    #     text, _ = next(iter(train_loader))
    #     output = model(text.to(DEVICE))
    #     make_dot(output, params=dict(model.named_parameters()))