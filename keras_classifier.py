import os
import glob

from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.utils import to_categorical
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer

from utility import get_complet_path
from pathlib import Path

from keras_classes import KerasClassifierWithVectorizer, KerasClassifierWithTokenizer, KerasClassifierWithWord2Vec, Vectorized, CountVectorizerDTO, KerasTokenizerDTO
from classifier import DataDTO

import numpy as np

# A lot of good stuff from here: https://realpython.com/python-keras-text-classification/

# Remove all saved models from the directory
def remove_saved_keras_models(remove_models):
    if (remove_models):
        files = glob.glob(get_complet_path("keras_models") + '/*')
        for f in files:
            os.remove(f)

# Get saved file from model if exists
def get_saved_model(name):
    model = None
    path = get_complet_path('keras_models') + "/" + name + ".h5"

    # We save/load model to improve performance
    if (Path(path).is_file()):
        model = load_model(path)

    return model, path

# Create a simple keras classifier with one layer
# This example does not have a hidden layer, just input and output
def get_simple_keras_classifier(name, data_dto, count_vectorizer_dto=None, verbose=False):

    if count_vectorizer_dto == None:
        count_vectorizer_dto = CountVectorizerDTO()

    vectorize_data = Vectorized(data_dto)
    vectorize_data.initialize_with_count_vectorizer(count_vectorizer_dto)

    model, path = get_saved_model(name)
    if (model == None):
        model = Sequential()
        model.add(Dense(10, input_dim=vectorize_data.input_dim, activation='relu'))
        model.add(Dense(len(vectorize_data.data_dto.target_names), activation='softmax', kernel_initializer='uniform')) 

        # categorial is the way to go for multiple possible categorys as results, instead of binary
        model.compile(loss='categorical_crossentropy',  
                    optimizer='adam', 
                    metrics=['accuracy'])
        
        if (verbose):
            model.summary()

        history = model.fit(vectorize_data.X_train, vectorize_data.y_train,
                        epochs=100,
                        verbose=verbose,
                        validation_data=(vectorize_data.X_test, vectorize_data.y_test),
                        batch_size=100)
    
        model.save(path)

        if (verbose):
            print (history)

    return KerasClassifierWithVectorizer(name, model, vectorize_data, verbose)

# this one create 5 layers with dropout to avoid overfitting
def get_denser_keras_classifier(name, data_dto, count_vectorizer_dto=None, verbose=False):

    if count_vectorizer_dto == None:
        count_vectorizer_dto = CountVectorizerDTO()

    vectorize_data = Vectorized(data_dto)
    vectorize_data.initialize_with_count_vectorizer(count_vectorizer_dto)

    model, path = get_saved_model(name)
    if (model == None):
        model = Sequential()
        model.add(Dense(512, input_dim=vectorize_data.input_dim, activation='relu'))
        model.add(Dropout(0.5)) # To avoid overfitting
        model.add(Dense(256, activation='sigmoid'))
        model.add(Dropout(0.5)) # To avoid overfitting
        model.add(Dense(len(vectorize_data.data_dto.target_names), activation='softmax', kernel_initializer='uniform')) 

        # categorial is the way to go for multiple possible categorys as results, instead of binary
        model.compile(loss='categorical_crossentropy',  
                    optimizer='adam', 
                    metrics=['accuracy'])
        
        if (verbose):
            model.summary()

        history = model.fit(vectorize_data.X_train, vectorize_data.y_train,
                        epochs=100,
                        verbose=verbose,
                        validation_data=(vectorize_data.X_test, vectorize_data.y_test),
                        batch_size=100)
    
        model.save(path)

        if (verbose):
            print (history)

    return KerasClassifierWithVectorizer(name, model, vectorize_data, verbose)

# this one uses 8 layers and the tokenizer from Keras library
def get_denser_keras_classifier_with_tokenizer(name, data_dto, keras_tokenizer_dto=None, verbose=False):

    if (keras_tokenizer_dto == None):
        keras_tokenizer_dto = KerasTokenizerDTO()

    vectorize_data = Vectorized(data_dto)
    vectorize_data.initialize_with_keras_tokenizer(keras_tokenizer_dto)

    model, path = get_saved_model(name)
    if (model == None):
        model = Sequential()
        
        model.add(Dense(512, input_shape=(vectorize_data.data_dto.vocab_size,)))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(Dense(len(vectorize_data.data_dto.target_names)))
        model.add(Activation('softmax'))

        # categorial is the way to go for multiple possible categorys as results, instead of binary
        model.compile(loss='categorical_crossentropy',  
                    optimizer='adam', 
                    metrics=['accuracy'])
        
        if (verbose):
            model.summary()

        history = model.fit(vectorize_data.X_train, vectorize_data.y_train,
                        epochs=30,
                        batch_size=100,
                        verbose=verbose,
                        validation_split=0.1)
    
        model.save(path)

        if (verbose):
            print (history)

    return KerasClassifierWithTokenizer(name, model, vectorize_data, verbose) 

# The two next methods using word2vec are inspired by: https://ahmedbesbes.com/sentiment-analysis-on-twitter-using-word2vec-and-keras.html
def get_keras_with_word2vec(name, data_dto, extra_param=None, verbose=False):
    
    vectorize_data = Vectorized(data_dto)
    vectorize_data.initialize_with_word2vec()

    model, path = get_saved_model(name)
    if (model == None):
    
        model = Sequential()
        model.add(Dense(32, activation='relu', input_dim=data_dto.vocab_size))
        model.add(Dense(len(vectorize_data.data_dto.target_names), activation='softmax'))
        
        model.compile(optimizer='rmsprop',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

        if (verbose):
            model.summary()

        model.fit(vectorize_data.X_train, vectorize_data.y_train, 
            epochs=9, 
            batch_size=32, 
            validation_data=(vectorize_data.X_test, vectorize_data.y_test),
            verbose=verbose)
        
        model.save(path)

    return KerasClassifierWithWord2Vec(name, model, vectorize_data, verbose)

# also using word2vec, but more layers, strange enough, its worse than simple layers
def get_keras_with_word2vec_denser(name, data_dto, extra_param=None, verbose=False):
    
    vectorize_data = Vectorized(data_dto)
    vectorize_data.initialize_with_word2vec()

    model, path = get_saved_model(name)
    if (model == None):
    
        model = Sequential()
        model.add(Dense(512, input_shape=(vectorize_data.data_dto.vocab_size,)))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(Dense(len(vectorize_data.data_dto.target_names)))
        model.add(Activation('softmax'))
        
        model.compile(optimizer='rmsprop',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

        if (verbose):
            model.summary()

        model.fit(vectorize_data.X_train, vectorize_data.y_train, 
            epochs=9, 
            batch_size=32, 
            verbose=verbose)
        
        model.save(path)

    return KerasClassifierWithWord2Vec(name, model, vectorize_data, verbose)
