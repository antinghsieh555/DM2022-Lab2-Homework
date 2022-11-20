import tensorflow_hub as hub
import tokenization
import os
import pandas as pd 
import numpy as np
import keras
import codecs
from keras_bert import Tokenizer
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, TimeDistributed,Dense
from keras_contrib.layers import CRF
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

train_df = pd.read_csv('/kaggle/working/train_data.csv',
                    lineterminator='\n')
val_df  = train_df.sample(frac=0.2)
test_df = pd.read_csv('/kaggle/working/test_data.csv',
                   lineterminator='\n')                

module_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
bert_layer = hub.KerasLayer(module_url, trainable=True)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

labelencoder = LabelEncoder()
train_df['emotion'] = labelencoder.fit_transform(train_df['emotion'])
test_df['emotion'] = labelencoder.fit_transform(test_df['emotion'])

def label_decode(le, one_hot_label):
    dec = np.argmax(one_hot_label, axis=1)
    return le.inverse_transform(dec)

def bert_encode(texts, tokenizer, max_len=50):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
            
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

max_len = 50
train_input = bert_encode(train_df.text.values, tokenizer, max_len=max_len)
test_input  = bert_encode(val_df.text.values, tokenizer, max_len=max_len)

train_labels = tf.onclick=keras.utils.to_categorical(train_df.emotion.values, num_classes=8)
test_labels  =tf.onclick=keras.utils.to_categorical(val_df.emotion.values, num_classes=8)

def Bert_model(bert_layer, max_len=512):
    input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    net = tf.keras.layers.Dense(64, activation='relu')(clf_output)
    net = tf.keras.layers.Dropout(0.2)(net)
    net = tf.keras.layers.Dense(32, activation='relu')(net)
    net = tf.keras.layers.Dropout(0.2)(net)
    out = tf.keras.layers.Dense(8, activation='softmax')(net)
    
    model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(tf.keras.optimizers.Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

    
def train():
    model = Bert_model(bert_layer, max_len=max_len)

    checkpoint = tf.keras.callbacks.ModelCheckpoint('model/model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
    stopTrain = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)

    train_history = model.fit(
        train_input,
        train_labels,
        validation_split=0.2,
        epochs=3,
        callbacks=[checkpoint, stopTrain],
        batch_size=32)

def test():
    model = load_weights('model/model.h5')
    predict = model.predict(test_df,batch_size=32)

    result = label_decode(labelencoder, predict)
    test_df['emotion'] = result

    test_data=test_data.drop(['identification','text'],axis=1)
    output_df['emotion']=test_data['emotion']
    output_df.to_csv('result.csv') 


if __name__ == '__main__':
    # train()
    test()