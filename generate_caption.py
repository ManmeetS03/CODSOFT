import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences                                                                                                                                               # type: ignore
from tensorflow.keras.models import load_model, Model                                                                                                                                                           # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer                                                                                                                                                       # type: ignore
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input                                                                                                                                         # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array                                                                                                                                         # type: ignore
from PIL import Image
import matplotlib.pyplot as plt

Base_dir = 'C:/Users/manme/Desktop/CODSOFT/Task3/data' 
working_dir = 'C:/Users/manme/Desktop/CODSOFT/Task3'

with open(os.path.join(working_dir, 'features.pkl'), 'rb') as file:
    features = pickle.load(file)

model = load_model(os.path.join(working_dir, 'saved_model.keras'))

with open(os.path.join(Base_dir, 'captions.txt'), 'r') as File:
    next(File)
    captions_doc = File.read()

mapping = {}
for line in captions_doc.split('\n'):
    tokens = line.split(',')
    if len(tokens) < 2:
        continue
    image_id, caption = tokens[0], tokens[1:]
    image_id = image_id.split('.')[0]
    caption = " ".join(caption)
    if image_id not in mapping:
        mapping[image_id] = []
    mapping[image_id].append(caption)

def Clean(mapping):
    import re
    for key, captions in mapping.items():
        for i in range(len(captions)):
            caption = captions[i]
            caption = caption.lower()
            caption = re.sub(r'[^a-z ]+', '', caption)
            caption = re.sub(r'\s+', ' ', caption)
            caption = 'startseq ' + ' '.join([word for word in caption.split() if len(word) > 1]) + ' endseq'
            captions[i] = caption

Clean(mapping)

all_captions = []
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1

max_length = max(len(caption.split()) for caption in all_captions)

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
    return in_text

def extract_features(image_path):
    model = VGG16(weights='imagenet')
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    features = model.predict(image, verbose=0)
    return features

def generate_caption(image_path):
    image_id = os.path.basename(image_path).split('.')[0]
    print(f"Loading image from: {image_path}")
    image = Image.open(image_path)
    image.show()

    features = extract_features(image_path)

    y_pred = predict_caption(model, features, tokenizer, max_length)
    print('-------Predicted Caption---------')
    print(y_pred)
    plt.imshow(image)
    plt.show()

generate_caption('C:/Users/manme/Desktop/CODSOFT/Task3/bmw.jpg')
