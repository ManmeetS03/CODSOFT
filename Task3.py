import os
import pickle
import numpy as np
from tqdm import tqdm
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input                                                                                                                                                                                                                                         # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array                                                                                                                                                                                                                                         # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer                                                                                                                                                                                                                                                       # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences                                                                                                                                                                                                                                               # type: ignore
from tensorflow.keras.models import Model, load_model                                                                                                                                                                                                                                                           # type: ignore
from tensorflow.keras.utils import to_categorical, plot_model                                                                                                                                                                                                                                                   # type: ignore
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add                                                                                                                                                                                                                                 # type: ignore
from PIL import Image
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu
import sys

print(sys.executable)

Base_dir = 'C:/Users/manme/Desktop/CODSOFT/Task3/data' 
working_dir = 'C:/Users/manme/Desktop/CODSOFT/Task3'

vgg_model = VGG16(weights='imagenet')
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

features = {}
directory = os.path.join(Base_dir, 'Images')

if not os.path.exists(directory):
    raise FileNotFoundError(f"The directory {directory} does not exist.")

for img_name in tqdm(os.listdir(directory)):
    img_path = os.path.join(directory, img_name)
    image = load_img(img_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
    image = preprocess_input(image)
    feature = vgg_model.predict(image, verbose=0)
    image_id = img_name.split('.')[0]
    features[image_id] = feature

with open(os.path.join(working_dir, 'features.pkl'), 'wb') as file:
    pickle.dump(features, file)

with open(os.path.join(Base_dir, 'captions.txt'), 'r') as File:
    next(File)
    captions_doc = File.read()

mapping = {}
for line in tqdm(captions_doc.split('\n')):
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
print(f"Max caption length: {max_length}")

image_ids = list(mapping.keys())
split = int(len(image_ids) * 0.9)
train = image_ids[:split]
test = image_ids[split:]

def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
    X1, X2, y = list(), list(), list()
    n = 0
    while True:
        for key in data_keys:
            n += 1
            captions = mapping[key]
            for caption in captions:
                seq = tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    X1.append(features[key][0])
                    X2.append(in_seq)
                    y.append(out_seq)
            if n == batch_size:
                X1, X2, y = np.array(X1), np.array(X2), np.array(y)
                yield {"image": X1, "text": X2}, y
                n = 0
                X1, X2, y = list(), list(), list()

inputs1 = Input(shape=(4096,), name="image")
fe1 = Dropout(0.4)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)

inputs2 = Input(shape=(max_length,), name="text")
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.4)(se1)
se3 = LSTM(256, return_sequences=False, use_bias=False)(se2)

decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')

plot_model(model, show_shapes=True)

epochs = 38
batch_size = 32
steps = len(train) // batch_size

for i in range(epochs):
    generator = data_generator(train, mapping, features, tokenizer, max_length, vocab_size, batch_size)
    model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)

model.save(os.path.join(working_dir, 'saved_model.keras'))