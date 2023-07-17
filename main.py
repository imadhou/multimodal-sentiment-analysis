import base64
from src.models import MultimodalModel, ImageModel, TextModel
import src.utils as ut
import torch
from gensim.models import Word2Vec
from PIL import Image
from flask import Flask, request
from flask_cors import CORS
import json
from io import BytesIO


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
word2vec = Word2Vec.load("./data/models/WORD2VEC.model")

# text_model = TextModel(3).to(DEVICE)
# text_model.load_state_dict(torch.load("data/models/RETRAINED_TEXT_MODEL.pt"))


# multimodal_model = MultimodalModel(TextModel(3, True).to(DEVICE), ImageModel(3, 224, True), 3).to(DEVICE)
# multimodal_model.load_state_dict(torch.load("data/models/MULTIMODAL_MODEL.pt"))

# print("Text model prediction")
# prediction = ut.predict(model=text_model, device=DEVICE, text="", word2vec_model=word2vec, image=None, IMG_SHAPE=None)
# print(prediction)

image = Image.open('c:/Users/MourtadaHouari/Downloads/image.jpg').convert("RGB")
image_model = ImageModel(3, 48).to(DEVICE)
image_model.load_state_dict(torch.load("data/models/IMAGES_MODEL.pt"))

print("Image model prediction")
prediction = ut.predict(model=image_model, device=DEVICE, text=None, word2vec_model=None, image=image, IMG_SHAPE=48)
print(prediction)

# print("Multimodal model prediction")
# prediction = ut.predict(model=multimodal_model, device=DEVICE, text="", word2vec_model=word2vec, image=image, IMG_SHAPE=224)
# print(prediction)



# app = Flask(__name__)


# @app.route("/text", methods=['POST'])
# def text_analysis():
#     json_data = request.data.decode('utf-8')
#     data = json.loads(json_data)
#     text = data.get('text')
#     prediction = ut.predict(model=text_model, device=DEVICE, text=text, word2vec_model=word2vec, image=None, IMG_SHAPE=None)
#     prediction = prediction[0][0].cpu().numpy()
#     value = {'negative': str(prediction[0]), 'neutral': str(prediction[1]), 'positive': str(prediction[2])}
#     return {'response': value}

# @app.route("/image", methods=['POST'])
# def image_analysis():
#     json_data = request.data.decode('utf-8')
#     data = json.loads(json_data)
#     image = data.get('image')

#     decoded_file = base64.b64decode(image)
#     image = Image.open(BytesIO(decoded_file)).convert('RGB')

#     prediction = ut.predict(model=image_model, device=DEVICE, text=None, word2vec_model=None, image=image, IMG_SHAPE=48)
#     prediction = prediction[0][0].cpu().numpy()
#     value = {'negative': str(prediction[0]), 'neutral': str(prediction[1]), 'positive': str(prediction[2])}
#     return {'response': value}


# @app.route("/multimodal", methods=['POST'])
# def multimodal_analysis():
#     json_data = request.data.decode('utf-8')
#     data = json.loads(json_data)
#     image = data.get('image')
#     decoded_file = base64.b64decode(image)
#     image = Image.open(BytesIO(decoded_file)).convert('RGB')
#     text = data.get('text')
#     prediction = ut.predict(model=multimodal_model, device=DEVICE, text=text, word2vec_model=word2vec, image=image, IMG_SHAPE=224)
#     prediction = prediction[0].cpu().numpy()
#     value = {'negative': str(prediction[0]), 'neutral': str(prediction[1]), 'positive': str(prediction[2])}
#     return {'response': value}




# if __name__ == "__main__":  # There is an error on this line
#     app.run(host='127.0.0.1')
#     print("test")