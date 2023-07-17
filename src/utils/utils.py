import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import pandas as pd
from nltk.tokenize import word_tokenize


import re
import pickle
import string
from nltk.tokenize import TweetTokenizer
import nltk
from nltk.corpus import stopwords 

import torchvision.transforms as transforms
import os
if __name__ == '__main__' and  __package__ is None:
    import sys
    from os import path
    sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )


dir = os.path.dirname(__file__)
SAVE_DIR = os.path.join(dir, '..', '..', 'data', 'models')
Emojis_Dict_path = os.path.join(dir, '')


dic = {0: "negative", 1: "neutral", 2: "positive"}


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def train_epoch(model, train_loader, optimizer, criterion, device, is_multimodal):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    if is_multimodal:
        for  batch_idx, (text, image, label) in enumerate(train_loader):
            print("Batch: ", batch_idx)
            text, image, label = text.to(device), image.to(device), label.to(device)

            optimizer.zero_grad()
            output = model(text, image)
            loss = criterion(output, label)
            acc = accuracy(output, label)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_acc += acc.item()

    else:
        for batch_idx, (data, label) in enumerate(train_loader):
            data, label = data.to(device), label.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            acc = accuracy(output, label)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_acc += acc.item()

    return running_loss / len(train_loader), running_acc / len(train_loader)


def test(model, predict_loader, criterion, device, is_multimodal):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    predictions = []
    true_labels = []

    with torch.no_grad():

        if is_multimodal:
            for batch_idx, (text, image, label) in enumerate(predict_loader):
                text, image, label = text.to(device), image.to(device), label.to(device)

                output = model(text, image)
                loss = criterion(output, label)
                acc = accuracy(output, label)

                running_loss += loss.item()
                running_acc += acc.item()

                _, preds = torch.max(output, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(label.cpu().numpy())
        else:
            for batch_idx, (data, label) in enumerate(predict_loader):
                data, label = data.to(device), label.to(device)

                output = model(data)
                loss = criterion(output, label)
                acc = accuracy(output, label)

                running_loss += loss.item()
                running_acc += acc.item()

                _, preds = torch.max(output, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(label.cpu().numpy())

    avg_loss = running_loss / len(predict_loader)
    avg_acc = running_acc / len(predict_loader)

    return avg_loss, avg_acc, predictions, true_labels


def val(model, test_loader, criterion, device, is_multimodal):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    with torch.no_grad():
        if is_multimodal:
            for batch_idx, (text, image, label) in enumerate(test_loader):
                text, image, label = text.to(device), image.to(device), label.to(device)

                output = model(text, image)
                loss = criterion(output, label)
                acc = accuracy(output, label)

                running_loss += loss.item()
                running_acc += acc.item()
        else:
            for batch_idx, (data, label) in enumerate(test_loader):
                data, label = data.to(device), label.to(device)

                output = model(data)
                loss = criterion(output, label)
                acc = accuracy(output, label)

                running_loss += loss.item()
                running_acc += acc.item()

    return running_loss / len(test_loader), running_acc / len(test_loader)

def save_model(model, filename):
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, filename))


def train(model, train_loader, test_loader, EPOCHS, model_name, optimizer, criterion, DEVICE, is_multimodal):
    best_val_acc = 0.0
    count = 0
    val_acc_list = []
    train_acc_list = []
    val_loss_list = []
    train_loss_list = []

    for epoch in range(EPOCHS):
        print("Epoch: "+ str(epoch))
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE, is_multimodal)
        val_loss, val_acc = val(model, test_loader, criterion, DEVICE, is_multimodal)

        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)
        
        print(
            f"Epoch: {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )
        

        if val_acc > best_val_acc:
            print(f"New best model! Saving...")
            save_model(model, model_name+".pt")
            best_val_acc = val_acc
            print(f"Best val acc: {best_val_acc:.4f}")
            count = 0
        else:
            count += 1
            print(f"Count: {count} of epochs have no improvement")
            if count == 5:
                print(f"Early stopping, best val acc: {best_val_acc:.4f}")
                break


emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

with open(Emojis_Dict_path+'Emoji_Dict.p', 'rb') as fp:
    Emoji_Dict = pickle.load(fp)
Emoji_Dict = {v: k for k, v in Emoji_Dict.items()}

PUNCUATION_LIST = list(string.punctuation)
tknzr = TweetTokenizer()
stpwrds = stopwords.words('english')
stemmer = nltk.PorterStemmer()

# Basic function to clean the text
def clean_text(text):
  text = str(text)

  # Remove identifications
  text = re.sub(r'[@#]\w+', '', text)
  # Remove links
  text = re.sub(r'http.?://[^/s]+[/s]?', '', text)
  # Remove html references
  text = re.sub(r'&[a-z]+;', '', text)
  # Replace repeated letters with only two occurences
  text = text.replace(r"(.)\1+", r"\1\1")
  #Remove single numeric terms
  text = re.sub(r'[0-9]', '', text)
  return text.strip().lower()

def remove_punctuation(word_list):
  return [w for w in word_list if w not in PUNCUATION_LIST]

def remove_stopwords(word_list):
  return [w for w in word_list if w not in stpwrds]

def stemm_words(word_list):
  return [stemmer.stem(w) for w in word_list]

def convert_emojis_to_word(word_list):
    word_list_second = []
    for word in word_list:
        if word in Emoji_Dict:
            word_list_second.append(Emoji_Dict[word])
        else:
            word_list_second.append(word)
    return word_list_second


def predict(model, device, text=None, word2vec_model=None, image=None, IMG_SHAPE=None, label=None):
    max_sequence_length = 100
    if image is not None:
        transform = transforms.Compose([
                        transforms.Resize((IMG_SHAPE, IMG_SHAPE)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
        image = transform(image).unsqueeze(0)

    if text is not None:
        text = clean_text(text)
        text = tknzr.tokenize(text)
        text = convert_emojis_to_word(text)
        text = remove_stopwords(text)
        text = remove_punctuation(text)
        text = stemm_words(text)
        tokens = text
        print(tokens)
        sequence_indices = [word2vec_model.wv.key_to_index[word] for word in tokens if word in word2vec_model.wv]
        padded_sequence = torch.tensor(sequence_indices + [0] * (max_sequence_length - len(sequence_indices)))[:max_sequence_length].unsqueeze(0)
    
    model.eval()
    with torch.no_grad():

        image_to_predict = None
        text_to_predict = None
        output = None

        if text is not None:
            text_to_predict = padded_sequence.to(device)
        if image is not None:
            image_to_predict = image.to(device)
        
        if text is not None and image is not None:
            output = model(text_to_predict, image_to_predict)
        if text is not None and image is None:
            output = model(text_to_predict)
        if image is not None and text is None:
            output = model(image_to_predict)
        result = None
        if text is not None and image is not None:
            result = torch.max(output, dim=0)
            result = result.indices.item()

        else:
            result = torch.max(output, dim=1)
            result = result.indices[0].cpu().item()


        
        print("predicted label:", dic[result])
        if label != None:
            print("actual label:", dic[label])
    return output, result
