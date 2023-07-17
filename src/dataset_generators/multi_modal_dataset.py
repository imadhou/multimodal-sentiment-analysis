import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
from torch.utils.data import Dataset
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import torch
import os
import pandas as pd
dir = os.path.dirname(__file__)
csv_file_path = os.path.join(dir, '..', '..', 'data', 'processed', 'multimodal.csv')
image_path_prefix = os.path.join("C:/Users/MourtadaHouari/Desktop/ps/mvsa/train_imagesuu/images")
word2vec_path = os.path.join(dir, '..', '..', 'data', 'models', 'WORD2VEC.model')


class MultimodalDataset(Dataset):
    def __init__(self, IMAGE_SHAPE, TEXT_COL, IMAGE_COL, LABEL_COL):
        self.transform = transforms.Compose([
                        transforms.Resize((IMAGE_SHAPE, IMAGE_SHAPE)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
        
        self.IMG_SHAPE = IMAGE_SHAPE
        self.IMAGE_COL = IMAGE_COL
        self.LABEL_COL = LABEL_COL
        self.TEXT_COL = TEXT_COL

        self.dataframe = pd.read_csv(csv_file_path)
        self.word2vec_model = Word2Vec.load(word2vec_path)
        self.SEQUENCE_LENGTH = 100
        
        self.preprocessed_sequences = self._preprocess_text()


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
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = self.dataframe.loc[idx, self.IMAGE_COL]
        label = self.dataframe.loc[idx, self.LABEL_COL]

        image = Image.open(os.path.join(image_path_prefix,image_path)).convert('RGB')
        image = self.transform(image)

        preprocessed_sequence = self.preprocessed_sequences[idx]

        return preprocessed_sequence, image, label
