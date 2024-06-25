# LSTM-CNN Sentiment Analysis on Rotten Tomatoes Reviews

Student ID: 23C11024

Student name: Chiêm Quốc Hùng

## Introduction

This is a homework for HCMUS' Deep Learning course. The homework is based on Dr. Nguyen Tien Huy's work on the paper Multi-channel LSTM-CNN model for Vietnamesesentiment analysis ([link here](https://www.researchgate.net/publication/321259272_Multi-channel_LSTM-CNN_model_for_Vietnamese_sentiment_analysis)).

The datasets can be found in the links below:

1. Base dataset (~400mb): [rotten_tomatoes_movie_reviews.csv](https://www.kaggle.com/datasets/andrezaza/clapper-massive-rotten-tomatoes-movies-and-reviews)
2. Additional testing dataset (~200mb): [rotten_tomatoes_critic_reviews.csv](https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset)
3. GloVe word embeddings (~1.3gb): [glove.6B.zip](https://nlp.stanford.edu/data/glove.6B.zip)

You can find the code in my [GitHub repository](https://github.com/cqhung1412/lstm_cnn_sentiment).

## Implementation

### Import dataset


```python
import os
import spacy
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Conv1D, GlobalMaxPooling1D, concatenate, Dense, Dropout, BatchNormalization
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor
```


```python
# Check if TensorFlow is using GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
```

Num GPUs Available:  1

```python
cpu_count = os.cpu_count()
print("Num CPUs Available: ", cpu_count)
```

Num CPUs Available:  12

```python
# Load spaCy model
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])

# Function to filter text by POS
def filter_text_by_pos(text):
    doc = nlp(text)
    filtered_text = " ".join([token.text for token in doc if token.pos_ in ['VERB', 'ADJ', 'ADV']])
    return filtered_text

# Function to apply POS filtering to a list of texts
def process_texts(texts):
    return [filter_text_by_pos(text) for text in texts]
```

```python
# Load pre-trained GloVe embeddings
def load_glove_embeddings(glove_path, tokenizer, embedding_dim):
    embeddings_index = {}
    with open(glove_path, 'r', encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, embedding_dim))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix
```

```python
data = pd.read_csv("./rotten_tomatoes_movie_reviews.csv")
data.head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>reviewId</th>
      <th>creationDate</th>
      <th>criticName</th>
      <th>isTopCritic</th>
      <th>originalScore</th>
      <th>reviewState</th>
      <th>publicatioName</th>
      <th>reviewText</th>
      <th>scoreSentiment</th>
      <th>reviewUrl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>beavers</td>
      <td>1145982</td>
      <td>2003-05-23</td>
      <td>Ivan M. Lincoln</td>
      <td>False</td>
      <td>3.5/4</td>
      <td>fresh</td>
      <td>Deseret News (Salt Lake City)</td>
      <td>Timed to be just long enough for most youngste...</td>
      <td>POSITIVE</td>
      <td>http://www.deseretnews.com/article/700003233/B...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>blood_mask</td>
      <td>1636744</td>
      <td>2007-06-02</td>
      <td>The Foywonder</td>
      <td>False</td>
      <td>1/5</td>
      <td>rotten</td>
      <td>Dread Central</td>
      <td>It doesn't matter if a movie costs 300 million...</td>
      <td>NEGATIVE</td>
      <td>http://www.dreadcentral.com/index.php?name=Rev...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>city_hunter_shinjuku_private_eyes</td>
      <td>2590987</td>
      <td>2019-05-28</td>
      <td>Reuben Baron</td>
      <td>False</td>
      <td>NaN</td>
      <td>fresh</td>
      <td>CBR</td>
      <td>The choreography is so precise and lifelike at...</td>
      <td>POSITIVE</td>
      <td>https://www.cbr.com/city-hunter-shinjuku-priva...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>city_hunter_shinjuku_private_eyes</td>
      <td>2558908</td>
      <td>2019-02-14</td>
      <td>Matt Schley</td>
      <td>False</td>
      <td>2.5/5</td>
      <td>rotten</td>
      <td>Japan Times</td>
      <td>The film's out-of-touch attempts at humor may ...</td>
      <td>NEGATIVE</td>
      <td>https://www.japantimes.co.jp/culture/2019/02/0...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>dangerous_men_2015</td>
      <td>2504681</td>
      <td>2018-08-29</td>
      <td>Pat Padua</td>
      <td>False</td>
      <td>NaN</td>
      <td>fresh</td>
      <td>DCist</td>
      <td>Its clumsy determination is endearing and some...</td>
      <td>POSITIVE</td>
      <td>http://dcist.com/2015/11/out_of_frame_dangerou...</td>
    </tr>
  </tbody>
</table>
</div>

### Data cleaning

```python
# Drop missing reviewText rows
data = data.dropna(subset=['reviewText'])

# Only use reviewState (result) and reviewText columns
data = data[['reviewState', 'reviewText']]

# Encode sentiment
data['sentiment'] = data['reviewState'].apply(lambda x: 1 if x == 'fresh' else 0)

data.head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>reviewState</th>
      <th>reviewText</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>fresh</td>
      <td>Timed to be just long enough for most youngste...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>rotten</td>
      <td>It doesn't matter if a movie costs 300 million...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>fresh</td>
      <td>The choreography is so precise and lifelike at...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>rotten</td>
      <td>The film's out-of-touch attempts at humor may ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>fresh</td>
      <td>Its clumsy determination is endearing and some...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>

### Training and testing data preparation

```python
# Extract texts and labels
texts = data['reviewText'].values
labels = data['sentiment'].values

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=69)

# Apply POS filtering to the training set only
# Split the data into chunks for parallel processing
num_chunks = cpu_count
chunks = np.array_split(x_train, num_chunks)

# Process the chunks in parallel
with ProcessPoolExecutor(max_workers=num_chunks) as executor:
    x_train_filtered_chunks = list(executor.map(process_texts, chunks))

# Combine the chunks back into a single list
x_train_filtered = [item for sublist in x_train_filtered_chunks for item in sublist]
```

```python
# Parameters
max_features = 1000
max_len = 100
embedding_dim = 100

# Text tokenizing
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(x_train_filtered)
x_train_sequences = tokenizer.texts_to_sequences(x_train_filtered)
# tokenizer.fit_on_texts(x_train)
# x_train_sequences = tokenizer.texts_to_sequences(x_train)
x_test_sequences = tokenizer.texts_to_sequences(x_test)

# Pad sequences
x_train_padded = pad_sequences(x_train_sequences, maxlen=max_len)
x_test_padded = pad_sequences(x_test_sequences, maxlen=max_len)
```

### Model layers

```python
glove_path = './glove.6B/glove.6B.100d.txt'
embedding_matrix = load_glove_embeddings(glove_path, tokenizer, embedding_dim)

# Embedding layer with pre-trained weights
embedding_layer = Embedding(len(tokenizer.word_index) + 1,
                            embedding_dim,
                            weights=[embedding_matrix],
                            trainable=False)
```

```python
# LSTM branch (with Batch Normalization and Dropout)
lstm_input = Input(shape=(max_len,))
embedded_sequences_lstm = embedding_layer(lstm_input)
lstm_out = LSTM(128, return_sequences=True)(embedded_sequences_lstm)
lstm_out = BatchNormalization()(lstm_out) # Batch Normalization helps to normalize activations and speed up convergence
lstm_out = Dropout(0.5)(lstm_out) # Dropout = 0.5 helps to prevent overfitting
lstm_out = LSTM(64, return_sequences=True)(lstm_out)
lstm_out = BatchNormalization()(lstm_out)
lstm_out = Dropout(0.5)(lstm_out)
lstm_out = LSTM(32)(lstm_out)
lstm_out = BatchNormalization()(lstm_out)
lstm_out = Dropout(0.5)(lstm_out)

# CNN layer branch (with Batch Normalization and Dropout)
cnn_input = Input(shape=(max_len,))
embedded_sequences_cnn = embedding_layer(cnn_input)
cnn_out = Conv1D(128, 5, activation='relu')(embedded_sequences_cnn)
cnn_out = BatchNormalization()(cnn_out)
cnn_out = Dropout(0.5)(cnn_out)
cnn_out = Conv1D(64, 5, activation='relu')(cnn_out)
cnn_out = BatchNormalization()(cnn_out)
cnn_out = Dropout(0.5)(cnn_out)
cnn_out = Conv1D(32, 5, activation='relu')(cnn_out)
cnn_out = BatchNormalization()(cnn_out)
cnn_out = Dropout(0.5)(cnn_out)
cnn_out = GlobalMaxPooling1D()(cnn_out)

# Concatenate LSTM and CNN outputs
merged = concatenate([lstm_out, cnn_out])
merged = Dense(32, activation='relu')(merged)
merged = BatchNormalization()(merged)
merged = Dropout(0.5)(merged)
pred = Dense(1, activation='sigmoid')(merged)

# Build model
model = Model(inputs=[lstm_input, cnn_input], outputs=pred)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "functional_17"</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)        </span>┃<span style="font-weight: bold"> Output Shape      </span>┃<span style="font-weight: bold">    Param # </span>┃<span style="font-weight: bold"> Connected to      </span>┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ input_layer_21      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">100</span>)       │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ -                 │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)        │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ input_layer_22      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">100</span>)       │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ -                 │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)        │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ embedding_1         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">100</span>, <span style="color: #00af00; text-decoration-color: #00af00">100</span>)  │  <span style="color: #00af00; text-decoration-color: #00af00">5,221,400</span> │ input_layer_21[<span style="color: #00af00; text-decoration-color: #00af00">0</span>… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Embedding</span>)         │                   │            │ input_layer_22[<span style="color: #00af00; text-decoration-color: #00af00">0</span>… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv1d_23 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv1D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">96</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)   │     <span style="color: #00af00; text-decoration-color: #00af00">64,128</span> │ embedding_1[<span style="color: #00af00; text-decoration-color: #00af00">9</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>] │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ lstm_37 (<span style="color: #0087ff; text-decoration-color: #0087ff">LSTM</span>)      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">100</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)  │    <span style="color: #00af00; text-decoration-color: #00af00">117,248</span> │ embedding_1[<span style="color: #00af00; text-decoration-color: #00af00">8</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>] │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">96</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)   │        <span style="color: #00af00; text-decoration-color: #00af00">512</span> │ conv1d_23[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">100</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)  │        <span style="color: #00af00; text-decoration-color: #00af00">512</span> │ lstm_37[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout_66          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">96</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)   │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)           │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout_63          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">100</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)  │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)           │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv1d_24 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv1D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">92</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)    │     <span style="color: #00af00; text-decoration-color: #00af00">41,024</span> │ dropout_66[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]  │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ lstm_38 (<span style="color: #0087ff; text-decoration-color: #0087ff">LSTM</span>)      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">100</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)   │     <span style="color: #00af00; text-decoration-color: #00af00">49,408</span> │ dropout_63[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]  │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">92</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)    │        <span style="color: #00af00; text-decoration-color: #00af00">256</span> │ conv1d_24[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">100</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)   │        <span style="color: #00af00; text-decoration-color: #00af00">256</span> │ lstm_38[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout_67          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">92</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)           │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout_64          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">100</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)   │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)           │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv1d_25 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv1D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">88</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)    │     <span style="color: #00af00; text-decoration-color: #00af00">10,272</span> │ dropout_67[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]  │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ lstm_39 (<span style="color: #0087ff; text-decoration-color: #0087ff">LSTM</span>)      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)        │     <span style="color: #00af00; text-decoration-color: #00af00">12,416</span> │ dropout_64[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]  │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">88</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)    │        <span style="color: #00af00; text-decoration-color: #00af00">128</span> │ conv1d_25[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)        │        <span style="color: #00af00; text-decoration-color: #00af00">128</span> │ lstm_39[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout_68          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">88</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)           │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout_65          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)        │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)           │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ global_max_pooling… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)        │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ dropout_68[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]  │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">GlobalMaxPooling1…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ concatenate_8       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)        │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ dropout_65[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>], │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Concatenate</span>)       │                   │            │ global_max_pooli… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_16 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)        │      <span style="color: #00af00; text-decoration-color: #00af00">2,080</span> │ concatenate_8[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)        │        <span style="color: #00af00; text-decoration-color: #00af00">128</span> │ dense_16[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]    │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout_69          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)        │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)           │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_17 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)         │         <span style="color: #00af00; text-decoration-color: #00af00">33</span> │ dropout_69[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]  │
└─────────────────────┴───────────────────┴────────────┴───────────────────┘
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">5,519,929</span> (21.06 MB)
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">297,569</span> (1.14 MB)
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">5,222,360</span> (19.92 MB)
</pre>

### Train and evaluate model

```python
# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True) # stop training when the validation loss stops improving to prevent overfitting
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0001) # reduce the learning rate when the validation loss plateaus, allowing finer convergence
```

```python
# Train model
history = model.fit([x_train_padded, x_train_padded], y_train, epochs=5, batch_size=32, validation_split=0.2)
```

    Epoch 1/5
    [1m27515/27515[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1463s[0m 53ms/step - accuracy: 0.6912 - loss: 0.6374 - val_accuracy: 0.7261 - val_loss: 0.5454
    Epoch 2/5
    [1m27515/27515[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1449s[0m 53ms/step - accuracy: 0.7261 - loss: 0.5406 - val_accuracy: 0.7444 - val_loss: 0.5263
    Epoch 3/5
    [1m27515/27515[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1438s[0m 52ms/step - accuracy: 0.7338 - loss: 0.5299 - val_accuracy: 0.7472 - val_loss: 0.5184
    Epoch 4/5
    [1m27515/27515[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1451s[0m 53ms/step - accuracy: 0.7359 - loss: 0.5244 - val_accuracy: 0.7425 - val_loss: 0.5132
    Epoch 5/5
    [1m27515/27515[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1452s[0m 53ms/step - accuracy: 0.7384 - loss: 0.5215 - val_accuracy: 0.7342 - val_loss: 0.5172

```python
# Evaluate model
score = model.evaluate([x_test_padded, x_test_padded], y_test)
print(f"Test accuracy: {score[1]}")
```

    [1m8599/8599[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m138s[0m 16ms/step - accuracy: 0.7329 - loss: 0.5285
    Test accuracy: 0.7337396740913391

```python
# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()
```


    
![png](rotten-tomatoes-lstm-cnn_files/rotten-tomatoes-lstm-cnn_21_0.png)
    


### Additional testing


```python
data = pd.read_csv("./rotten_tomatoes_critic_reviews.csv")
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rotten_tomatoes_link</th>
      <th>critic_name</th>
      <th>top_critic</th>
      <th>publisher_name</th>
      <th>review_type</th>
      <th>review_score</th>
      <th>review_date</th>
      <th>review_content</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>m/0814255</td>
      <td>Andrew L. Urban</td>
      <td>False</td>
      <td>Urban Cinefile</td>
      <td>Fresh</td>
      <td>NaN</td>
      <td>2010-02-06</td>
      <td>A fantasy adventure that fuses Greek mythology...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>m/0814255</td>
      <td>Louise Keller</td>
      <td>False</td>
      <td>Urban Cinefile</td>
      <td>Fresh</td>
      <td>NaN</td>
      <td>2010-02-06</td>
      <td>Uma Thurman as Medusa, the gorgon with a coiff...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>m/0814255</td>
      <td>NaN</td>
      <td>False</td>
      <td>FILMINK (Australia)</td>
      <td>Fresh</td>
      <td>NaN</td>
      <td>2010-02-09</td>
      <td>With a top-notch cast and dazzling special eff...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>m/0814255</td>
      <td>Ben McEachen</td>
      <td>False</td>
      <td>Sunday Mail (Australia)</td>
      <td>Fresh</td>
      <td>3.5/5</td>
      <td>2010-02-09</td>
      <td>Whether audiences will get behind The Lightnin...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>m/0814255</td>
      <td>Ethan Alter</td>
      <td>True</td>
      <td>Hollywood Reporter</td>
      <td>Rotten</td>
      <td>NaN</td>
      <td>2010-02-10</td>
      <td>What's really lacking in The Lightning Thief i...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Drop missing review_content rows
data = data.dropna(subset=['review_content'])

# Only use review_type (result) and review_content columns
data = data[['review_type', 'review_content']]

# Encode sentiment
data['sentiment'] = data['review_type'].apply(lambda x: 1 if x == 'Fresh' else 0)

data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review_type</th>
      <th>review_content</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Fresh</td>
      <td>A fantasy adventure that fuses Greek mythology...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Fresh</td>
      <td>Uma Thurman as Medusa, the gorgon with a coiff...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Fresh</td>
      <td>With a top-notch cast and dazzling special eff...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Fresh</td>
      <td>Whether audiences will get behind The Lightnin...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Rotten</td>
      <td>What's really lacking in The Lightning Thief i...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Extract texts and labels
texts = data['review_content'].values
labels = data['sentiment'].values

# Text tokenizing
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index

# Pad sequences
data = pad_sequences(sequences, maxlen=max_len)
```


```python
# Evaluate model
score = model.evaluate([data, data], labels)
print(f"Test accuracy: {score[1]}")
```

    [1m33257/33257[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m538s[0m 16ms/step - accuracy: 0.6134 - loss: 0.7961
    Test accuracy: 0.621994137763977



```python
model.save('./models/rotten_tomatoes_sentiment_model.keras', overwrite=True)
model.save_weights("./models/rotten_tomatoes_sentiment_model.weights.h5", overwrite=True)
```

## Results

| # | Model | Base Testing Accuracy | Base Testing Loss | Additional Testing Accuracy | Additional Testing Loss |
| --- | --- | --- | --- | --- | --- |
| 1 | Initial model (128/64) | _ | _ | 0.5530 | 1.0205 |
| 2 | Bigger model (256/128) with Batch Normalization and Dropout + Filtered content for training + 5 epochs | _ | _ | 0.5806 | 0.7253 |
| 3 | Node-reduced model (128/64) + Filtered content + GloVe 6B 100d word embedding + 10 epochs | 0.7412 | 0.5101 | 0.5973 | 0.8537 |
| 4 | Model (128/64) + Non-filtered content + GloVe 6B 300d word embedding + 10 epochs | **0.7987** | 0.4815 | 0.5765 | 1.3394 |
| 5 | Bigger model (256/128) + Filtered content + GloVe 6B 300d word embedding + 5 epochs | 0.7422 | 0.5109 | *0.6048* | 0.8003 |
| 6 | Extra-layer smaller model (128/64/32) + Corrected usage of binary_crossentropy loss function and sigmoid activation + GloVe 6B 100d word embedding + 5 epochs | 0.7329  | 0.5285 | **0.6134** | 0.7961 |

## Conclusion

In the 3rd training iteration, the model has achieved a 59.73% accuracy on the additional testing dataset, a 74.12% accuracy on the base testing dataset. The model has been trained with a reduced number of nodes (64/128) and filtered content. The model has also been trained with GloVe 6B 100d word embeddings and 10 epochs. The model has shown a significant improvement in accuracy compared to the initial model (55.30% accuracy on the additional testing dataset).

However, in the 4th training iteration, the model has achieved a 57.65% accuracy on the additional testing dataset, a 79.87% accuracy on the base testing dataset. The model has been trained with the same number of nodes (64/128) and non-filtered content. The model has also been trained with GloVe 6B 300d word embeddings and 10 epochs. The model has shown a decrease in accuracy compared to the 3rd training iteration while having ***a higher overfitting issue*** (increased 5.75% accuracy on base testing dataset).

Finally, in the 6th training iteration, the model has achieved a 61.34% accuracy on the additional testing dataset, a 73.29% accuracy on the base testing dataset. The model has been trained with an extra layer (64/128/32) and corrected usage of binary_crossentropy loss function and sigmoid activation (previously, the models used binary_crossentropy and softmax). The model has also been trained with GloVe 6B 100d word embeddings and 5 epochs. The model has shown a significant improvement in accuracy compared to the 4th and 5th training iteration.

The model can be further improved by:
- Adding more filters to the initial review content (remove special characters, stopwords, etc.).
- Using more data for training (k-fold cross validation).

## References

- [Multi-channel LSTM-CNN model for Vietnamesesentiment analysis](https://www.researchgate.net/publication/321259272_Multi-channel_LSTM-CNN_model_for_Vietnamese_sentiment_analysis)

Some interesting references for further development:
- [Kaggle Sentiment Analysis: Rotten Tomato Movie Reviews](https://www.kaggle.com/code/oragula/sentiment-analysis-rotten-tomato-movie-reviews)
