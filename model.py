from transformers import TFAutoModel
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Concatenate
from aspect_base_vietnamese.util import argmax_label_matrix
from aspect_base_vietnamese.vlsp_process import PolarityMapping
import numpy as np
class VLSP2018Task(Model):
    def __init__(self, model, aspect_category_names, optimizer, **kwargs):
        super(VLSP2018Task, self).__init__(**kwargs)
        self.aspect_category_names = aspect_category_names
        self.pre_bert = TFAutoModel.from_pretrained(model, output_hidden_states=True)
        self.last_4_hidden_states = Concatenate(name='last_4_hidden_states')
        self.dropout = Dropout(0.2)
        self.dense_layers = [
            Dense(4, activation='softmax', name=label.replace('#', '-').replace('&', '_').replace('/', '-')) 
            for label in self.aspect_category_names
        ]

        self.flatten_onehot_labels = Concatenate(name='FlattenOneHotLabels')
        self.loss = 'binary_crossentropy'
        self.compile(optimizer=optimizer, loss=self.loss)
    

    def call(self, inputs):
        hidden_states = self.pre_bert(inputs).hidden_states 
        pooled_output = self.last_4_hidden_states(tuple([hidden_states[i] for i in range(-4, 0)])) 
        x = self.dropout(pooled_output[:, 0, :])       
        outputs = [dense_layer(x) for dense_layer in self.dense_layers]
        return self.flatten_onehot_labels(outputs)
     
    def acsa_predict(self, text_data, batch_size=1):
        y_pred = self.predict(text_data, batch_size=batch_size, verbose=1)
        return argmax_label_matrix(y_pred)

    def print_acsa_pred(self, y_pred):
        polarities = map(lambda x: PolarityMapping.INDEX_TO_POLARITY[x], y_pred)
        for aspect_category, polarity in zip(self.aspect_category_names, polarities): 
            if polarity: print(f'=> {aspect_category},{polarity}')
