import os
import pandas as pd
import numpy as np
from operator import itemgetter
import time
from tqdm import tqdm

from utility import get_complet_path, bcolors, clean_results
from sklearn_classifiers import SkLearnClassifier, ClassifierTestSet

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from keras_classifier import remove_saved_keras_models, get_simple_keras_classifier, get_denser_keras_classifier, get_denser_keras_classifier_with_tokenizer, get_keras_with_word2vec, get_keras_with_word2vec_denser
from keras_classes import CountVectorizerDTO, KerasTokenizerDTO, KerasClassifierTestSet
from classifier import DataDTO, PreTraitementDTO

from text_analyser import analyse_text, replace_keywords_in_text, replace_punctuation_in_text, lemmatize

# Loads the data for training, evaluation and prediction. Apply features if needed to all
def get_data(pre_traitement_dto):
    df = load_and_treat_data([1,2,3,4], pre_traitement_dto)

    texts = df["text"].values
    labels = df["label"].values

    target_names = df.label.unique()
    
    # Create our training and test data randomically
    data_train, data_test, target_train, target_test = train_test_split(
            texts,
            labels,
            test_size=0.20,
            train_size=0.80,
            random_state=1000)

    data_to_predict = get_predict_data(pre_traitement_dto)

    data_dto = DataDTO(data_train, data_test, target_train, target_test, target_names, data_to_predict) 

    return data_dto

# Loads the data do be predicted
def get_predict_data(pre_traitement_dto):
    df = load_and_treat_data([1,2,3], pre_traitement_dto)
    values = df["text"].values
    return values # np.ndenumerate( returns an enumerate to facilitate for

def load_and_treat_data(columns_to_load, pre_traitement_dto):
    # Read our train data from file
    df = pd.read_table(get_complet_path('data/train.txt'), sep='\t', header=0, usecols=columns_to_load)
    df.fillna('', inplace=True)
    
    # Threat all the columns as one
    if (pre_traitement_dto.apply_phrase_separation):
        df['text'] = df[['turn1', 'turn2', 'turn3']].apply(lambda x: "<p>" + "</p><p>".join(x) + "</p>" , axis=1)
    else:
        df['text'] = df[['turn1', 'turn2', 'turn3']].apply(lambda x: " ".join(x), axis=1)

    if pre_traitement_dto.replace_keywords:
        df['text'] = df['text'].apply(lambda x: replace_keywords_in_text(x))

    if pre_traitement_dto.remove_punctuation:
        df['text'] = df['text'].apply(lambda x: replace_punctuation_in_text(x))

    if pre_traitement_dto.apply_lemm:
        df['text'] = df['text'].apply(lambda x: lemmatize(x, pre_traitement_dto))

    df.drop('turn1', inplace=True, axis=1)
    df.drop('turn2', inplace=True, axis=1)
    df.drop('turn3', inplace=True, axis=1)

    return df 
    
# Train multiple SkLearn Classifiers differents, get the best result and predict the texts without label
def test_with_sklearn_classifiers(data_dto, pre_traitement_dto, name='sklearn', execute_quantity=0, verbose=False):

    start_time = time.time()

    # declaration of all tests
    classifiers = [
        ClassifierTestSet('MultinomialNB', MultinomialNB(), stop_words=None, max_df=1.0, min_df=1, use_Tfid=False, binary=False),
        ClassifierTestSet('MultinomialNB', MultinomialNB(), stop_words=None, max_df=0.5, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('MultinomialNB', MultinomialNB(), stop_words=None, max_df=0.7, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('MultinomialNB', MultinomialNB(), stop_words=None, max_df=0.8, min_df=0.1, use_Tfid=False, binary=False),
        ClassifierTestSet('MultinomialNB', MultinomialNB(), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=False),
        ClassifierTestSet('MultinomialNB', MultinomialNB(), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=True),
        ClassifierTestSet('MultinomialNB', MultinomialNB(), stop_words='english', max_df=1.0, min_df=1, use_Tfid=False, binary=False),
        ClassifierTestSet('MultinomialNB', MultinomialNB(), stop_words='english', max_df=0.5, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('MultinomialNB', MultinomialNB(), stop_words='english', max_df=0.7, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('MultinomialNB', MultinomialNB(), stop_words='english', max_df=1.0, min_df=1, use_Tfid=True, binary=False),
        ClassifierTestSet('MultinomialNB', MultinomialNB(), stop_words='english', max_df=1.0, min_df=1, use_Tfid=True, binary=True),
        ClassifierTestSet('MultinomialNB', MultinomialNB(), stop_words=None, max_df=1.0, min_df=1, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('MultinomialNB', MultinomialNB(), stop_words=None, max_df=0.5, min_df=0.05, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('MultinomialNB', MultinomialNB(), stop_words=None, max_df=0.7, min_df=0.05, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('MultinomialNB', MultinomialNB(), stop_words=None, max_df=0.8, min_df=0.1, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('MultinomialNB', MultinomialNB(), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('MultinomialNB', MultinomialNB(), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=True, ngram_range=(1,2)),
        ClassifierTestSet('MultinomialNB', MultinomialNB(), stop_words='english', max_df=1.0, min_df=1, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('MultinomialNB', MultinomialNB(), stop_words='english', max_df=0.5, min_df=0.05, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('MultinomialNB', MultinomialNB(), stop_words='english', max_df=0.7, min_df=0.05, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('MultinomialNB', MultinomialNB(), stop_words='english', max_df=1.0, min_df=1, use_Tfid=True, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('MultinomialNB', MultinomialNB(), stop_words='english', max_df=1.0, min_df=1, use_Tfid=True, binary=True, ngram_range=(1,2)),

        ClassifierTestSet('LogisticRegression', LogisticRegression(), stop_words=None, max_df=1.0, min_df=1, use_Tfid=False, binary=False),
        ClassifierTestSet('LogisticRegression', LogisticRegression(), stop_words=None, max_df=0.5, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('LogisticRegression', LogisticRegression(), stop_words=None, max_df=0.7, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('LogisticRegression', LogisticRegression(), stop_words=None, max_df=0.8, min_df=0.1, use_Tfid=False, binary=False),
        ClassifierTestSet('LogisticRegression', LogisticRegression(), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=False),
        ClassifierTestSet('LogisticRegression', LogisticRegression(), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=True),
        ClassifierTestSet('LogisticRegression', LogisticRegression(), stop_words='english', max_df=1.0, min_df=1, use_Tfid=False, binary=False),
        ClassifierTestSet('LogisticRegression', LogisticRegression(), stop_words='english', max_df=0.5, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('LogisticRegression', LogisticRegression(), stop_words='english', max_df=0.7, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('LogisticRegression', LogisticRegression(), stop_words='english', max_df=1.0, min_df=1, use_Tfid=True, binary=False),
        ClassifierTestSet('LogisticRegression', LogisticRegression(), stop_words='english', max_df=1.0, min_df=1, use_Tfid=True, binary=True),
        ClassifierTestSet('LogisticRegression', LogisticRegression(), stop_words=None, max_df=1.0, min_df=1, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('LogisticRegression', LogisticRegression(), stop_words=None, max_df=0.5, min_df=0.05, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('LogisticRegression', LogisticRegression(), stop_words=None, max_df=0.7, min_df=0.05, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('LogisticRegression', LogisticRegression(), stop_words=None, max_df=0.8, min_df=0.1, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('LogisticRegression', LogisticRegression(), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('LogisticRegression', LogisticRegression(), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=True, ngram_range=(1,2)),
        ClassifierTestSet('LogisticRegression', LogisticRegression(), stop_words='english', max_df=1.0, min_df=1, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('LogisticRegression', LogisticRegression(), stop_words='english', max_df=0.5, min_df=0.05, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('LogisticRegression', LogisticRegression(), stop_words='english', max_df=0.7, min_df=0.05, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('LogisticRegression', LogisticRegression(), stop_words='english', max_df=1.0, min_df=1, use_Tfid=True, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('LogisticRegression', LogisticRegression(), stop_words='english', max_df=1.0, min_df=1, use_Tfid=True, binary=True, ngram_range=(1,2)),

        ClassifierTestSet('SGD ', SGDClassifier(max_iter=1000), stop_words=None, max_df=1.0, min_df=1, use_Tfid=False, binary=False),
        ClassifierTestSet('SGD ', SGDClassifier(max_iter=1000), stop_words=None, max_df=0.5, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('SGD ', SGDClassifier(max_iter=1000), stop_words=None, max_df=0.7, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('SGD ', SGDClassifier(max_iter=1000), stop_words=None, max_df=0.8, min_df=0.1, use_Tfid=False, binary=False),
        ClassifierTestSet('SGD ', SGDClassifier(max_iter=1000), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=False),
        ClassifierTestSet('SGD ', SGDClassifier(max_iter=1000), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=True),
        ClassifierTestSet('SGD ', SGDClassifier(max_iter=1000), stop_words='english', max_df=1.0, min_df=1, use_Tfid=False, binary=False),
        ClassifierTestSet('SGD ', SGDClassifier(max_iter=1000), stop_words='english', max_df=0.5, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('SGD ', SGDClassifier(max_iter=1000), stop_words='english', max_df=0.7, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('SGD ', SGDClassifier(max_iter=1000), stop_words='english', max_df=1.0, min_df=1, use_Tfid=True, binary=False),
        ClassifierTestSet('SGD ', SGDClassifier(max_iter=1000), stop_words='english', max_df=1.0, min_df=1, use_Tfid=True, binary=True),
        ClassifierTestSet('SGD ', SGDClassifier(max_iter=1000), stop_words=None, max_df=1.0, min_df=1, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('SGD ', SGDClassifier(max_iter=1000), stop_words=None, max_df=0.5, min_df=0.05, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('SGD ', SGDClassifier(max_iter=1000), stop_words=None, max_df=0.7, min_df=0.05, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('SGD ', SGDClassifier(max_iter=1000), stop_words=None, max_df=0.8, min_df=0.1, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('SGD ', SGDClassifier(max_iter=1000), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('SGD ', SGDClassifier(max_iter=1000), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=True, ngram_range=(1,2)),
        ClassifierTestSet('SGD ', SGDClassifier(max_iter=1000), stop_words='english', max_df=1.0, min_df=1, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('SGD ', SGDClassifier(max_iter=1000), stop_words='english', max_df=0.5, min_df=0.05, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('SGD ', SGDClassifier(max_iter=1000), stop_words='english', max_df=0.7, min_df=0.05, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('SGD ', SGDClassifier(max_iter=1000), stop_words='english', max_df=1.0, min_df=1, use_Tfid=True, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('SGD ', SGDClassifier(max_iter=1000), stop_words='english', max_df=1.0, min_df=1, use_Tfid=True, binary=True, ngram_range=(1,2)),

        ClassifierTestSet('LinearSVC ', LinearSVC(random_state=0, tol=1e-5), stop_words=None, max_df=1.0, min_df=1, use_Tfid=False, binary=False),
        ClassifierTestSet('LinearSVC ', LinearSVC(random_state=0, tol=1e-5), stop_words=None, max_df=0.5, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('LinearSVC ', LinearSVC(random_state=0, tol=1e-5), stop_words=None, max_df=0.7, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('LinearSVC ', LinearSVC(random_state=0, tol=1e-5), stop_words=None, max_df=0.8, min_df=0.11, use_Tfid=False, binary=False),
        ClassifierTestSet('LinearSVC ', LinearSVC(random_state=0, tol=1e-5), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=False),
        ClassifierTestSet('LinearSVC ', LinearSVC(random_state=0, tol=1e-5), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=True),
        ClassifierTestSet('LinearSVC ', LinearSVC(random_state=0, tol=1e-5), stop_words='english', max_df=1.0, min_df=1, use_Tfid=False, binary=False),
        ClassifierTestSet('LinearSVC ', LinearSVC(random_state=0, tol=1e-5), stop_words='english', max_df=0.5, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('LinearSVC ', LinearSVC(random_state=0, tol=1e-5), stop_words='english', max_df=0.7, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('LinearSVC ', LinearSVC(random_state=0, tol=1e-5), stop_words='english', max_df=1.0, min_df=1, use_Tfid=True, binary=False),
        ClassifierTestSet('LinearSVC ', LinearSVC(random_state=0, tol=1e-5), stop_words='english', max_df=1.0, min_df=1, use_Tfid=True, binary=True),
        ClassifierTestSet('LinearSVC ', LinearSVC(random_state=0, tol=1e-5), stop_words=None, max_df=1.0, min_df=1, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('LinearSVC ', LinearSVC(random_state=0, tol=1e-5), stop_words=None, max_df=0.5, min_df=0.05, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('LinearSVC ', LinearSVC(random_state=0, tol=1e-5), stop_words=None, max_df=0.7, min_df=0.05, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('LinearSVC ', LinearSVC(random_state=0, tol=1e-5), stop_words=None, max_df=0.8, min_df=0.11, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('LinearSVC ', LinearSVC(random_state=0, tol=1e-5), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('LinearSVC ', LinearSVC(random_state=0, tol=1e-5), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=True, ngram_range=(1,2)),
        ClassifierTestSet('LinearSVC ', LinearSVC(random_state=0, tol=1e-5), stop_words='english', max_df=1.0, min_df=1, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('LinearSVC ', LinearSVC(random_state=0, tol=1e-5), stop_words='english', max_df=0.5, min_df=0.05, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('LinearSVC ', LinearSVC(random_state=0, tol=1e-5), stop_words='english', max_df=0.7, min_df=0.05, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('LinearSVC ', LinearSVC(random_state=0, tol=1e-5), stop_words='english', max_df=1.0, min_df=1, use_Tfid=True, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('LinearSVC ', LinearSVC(random_state=0, tol=1e-5), stop_words='english', max_df=1.0, min_df=1, use_Tfid=True, binary=True, ngram_range=(1,2)),

        ClassifierTestSet('SVC', SVC(kernel="linear", C=0.025), stop_words=None, max_df=1.0, min_df=1, use_Tfid=False, binary=False),
        ClassifierTestSet('SVC', SVC(kernel="linear", C=0.025), stop_words=None, max_df=0.5, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('SVC', SVC(kernel="linear", C=0.025), stop_words=None, max_df=0.7, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('SVC', SVC(kernel="linear", C=0.025), stop_words=None, max_df=0.8, min_df=0.1, use_Tfid=False, binary=False),
        ClassifierTestSet('SVC', SVC(kernel="linear", C=0.025), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=False),
        ClassifierTestSet('SVC', SVC(kernel="linear", C=0.025), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=True),
        ClassifierTestSet('SVC', SVC(kernel="linear", C=0.025), stop_words='english', max_df=1.0, min_df=1, use_Tfid=False, binary=False),
        ClassifierTestSet('SVC', SVC(kernel="linear", C=0.025), stop_words='english', max_df=0.5, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('SVC', SVC(kernel="linear", C=0.025), stop_words='english', max_df=0.7, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('SVC', SVC(kernel="linear", C=0.025), stop_words='english', max_df=1.0, min_df=1, use_Tfid=True, binary=False),
        ClassifierTestSet('SVC', SVC(kernel="linear", C=0.025), stop_words='english', max_df=1.0, min_df=1, use_Tfid=True, binary=True),
        ClassifierTestSet('SVC', SVC(kernel="linear", C=0.025), stop_words=None, max_df=1.0, min_df=1, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('SVC', SVC(kernel="linear", C=0.025), stop_words=None, max_df=0.5, min_df=0.05, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('SVC', SVC(kernel="linear", C=0.025), stop_words=None, max_df=0.7, min_df=0.05, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('SVC', SVC(kernel="linear", C=0.025), stop_words=None, max_df=0.8, min_df=0.1, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('SVC', SVC(kernel="linear", C=0.025), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('SVC', SVC(kernel="linear", C=0.025), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=True, ngram_range=(1,2)),
        ClassifierTestSet('SVC', SVC(kernel="linear", C=0.025), stop_words='english', max_df=1.0, min_df=1, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('SVC', SVC(kernel="linear", C=0.025), stop_words='english', max_df=0.5, min_df=0.05, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('SVC', SVC(kernel="linear", C=0.025), stop_words='english', max_df=0.7, min_df=0.05, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('SVC', SVC(kernel="linear", C=0.025), stop_words='english', max_df=1.0, min_df=1, use_Tfid=True, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('SVC', SVC(kernel="linear", C=0.025), stop_words='english', max_df=1.0, min_df=1, use_Tfid=True, binary=True, ngram_range=(1,2)),

        ClassifierTestSet('SVC', SVC(kernel="rbf", C=0.025), stop_words=None, max_df=1.0, min_df=1, use_Tfid=False, binary=False),
        ClassifierTestSet('SVC', SVC(kernel="rbf", C=0.025), stop_words=None, max_df=0.5, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('SVC', SVC(kernel="rbf", C=0.025), stop_words=None, max_df=0.7, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('SVC', SVC(kernel="rbf", C=0.025), stop_words=None, max_df=0.8, min_df=0.1, use_Tfid=False, binary=False),
        ClassifierTestSet('SVC', SVC(kernel="rbf", C=0.025), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=False),
        ClassifierTestSet('SVC', SVC(kernel="rbf", C=0.025), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=True),
        ClassifierTestSet('SVC', SVC(kernel="rbf", C=0.025), stop_words='english', max_df=1.0, min_df=1, use_Tfid=False, binary=False),
        ClassifierTestSet('SVC', SVC(kernel="rbf", C=0.025), stop_words='english', max_df=0.5, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('SVC', SVC(kernel="rbf", C=0.025), stop_words='english', max_df=0.7, min_df=0.05, use_Tfid=False, binary=False),
        ClassifierTestSet('SVC', SVC(kernel="rbf", C=0.025), stop_words='english', max_df=1.0, min_df=1, use_Tfid=True, binary=False),
        ClassifierTestSet('SVC', SVC(kernel="rbf", C=0.025), stop_words='english', max_df=1.0, min_df=1, use_Tfid=True, binary=True),
        ClassifierTestSet('SVC', SVC(kernel="rbf", C=0.025), stop_words=None, max_df=1.0, min_df=1, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('SVC', SVC(kernel="rbf", C=0.025), stop_words=None, max_df=0.5, min_df=0.05, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('SVC', SVC(kernel="rbf", C=0.025), stop_words=None, max_df=0.7, min_df=0.05, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('SVC', SVC(kernel="rbf", C=0.025), stop_words=None, max_df=0.8, min_df=0.1, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('SVC', SVC(kernel="rbf", C=0.025), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('SVC', SVC(kernel="rbf", C=0.025), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=True, ngram_range=(1,2)),
        ClassifierTestSet('SVC', SVC(kernel="rbf", C=0.025), stop_words='english', max_df=1.0, min_df=1, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('SVC', SVC(kernel="rbf", C=0.025), stop_words='english', max_df=0.5, min_df=0.05, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('SVC', SVC(kernel="rbf", C=0.025), stop_words='english', max_df=0.7, min_df=0.05, use_Tfid=False, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('SVC', SVC(kernel="rbf", C=0.025), stop_words='english', max_df=1.0, min_df=1, use_Tfid=True, binary=False, ngram_range=(1,2)),
        ClassifierTestSet('SVC', SVC(kernel="rbf", C=0.025), stop_words='english', max_df=1.0, min_df=1, use_Tfid=True, binary=True, ngram_range=(1,2)),
        
        # 6 of the best classifiers (previously seen) with extra-feature added (Emojis and positive/negative words)
        ClassifierTestSet('LinearSVC ', LinearSVC(random_state=0, tol=1e-5), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=False, ngram_range=(1,2), apply_count_features=True, apply_sentiment_features=True),
        ClassifierTestSet('LinearSVC ', LinearSVC(random_state=0, tol=1e-5), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=True, ngram_range=(1,2), apply_count_features=True, apply_sentiment_features=True),

        ClassifierTestSet('LogisticRegression', LogisticRegression(), stop_words=None, max_df=1.0, min_df=1, use_Tfid=False, binary=False, ngram_range=(1,2), apply_count_features=True, apply_sentiment_features=True),
        ClassifierTestSet('LogisticRegression', LogisticRegression(), stop_words=None, max_df=1.0, min_df=1, use_Tfid=False, binary=False, ngram_range=(1,1), apply_count_features=True, apply_sentiment_features=True),

        ClassifierTestSet('SGD ', SGDClassifier(max_iter=1000), stop_words=None, max_df=1.0, min_df=1, use_Tfid=False, binary=False, ngram_range=(1,2), apply_count_features=True, apply_sentiment_features=True),
        ClassifierTestSet('SGD ', SGDClassifier(max_iter=1000), stop_words=None, max_df=1.0, min_df=1, use_Tfid=True, binary=True, ngram_range=(1,2), apply_count_features=True, apply_sentiment_features=True),
    ]
    
    if (verbose):
        headerClassifier = ClassifierTestSet('Header', None)
        print(headerClassifier.str_keys())

    if (execute_quantity == 0):
        classifiers_to_execute = classifiers
    else:
        classifiers_to_execute = classifiers[:execute_quantity]

    # Execution of the tests
    results = []
    print("\nEvaluating {} classifiers\n".format(name))
    for classifier in tqdm(classifiers_to_execute):
        skLearnClassifier = SkLearnClassifier(data_dto, pre_traitement_dto)
        skLearnClassifier.train_classifier(classifier, False)
        
        write_classifier_result_to_file(name + '_classifiers.txt', skLearnClassifier)
        results.append(skLearnClassifier)

    print("\n{}# {:.2f} seconds to do {} {}".format(bcolors.WARNING, (time.time() - start_time), name, bcolors.ENDC))

    # prediction with the best classification
    return predict_with_best(results, name + '_prediction.txt', data_dto.data_to_predict)

# Train multiple keras classifiers differents, takes the best one and predict the texts without label
def test_with_keras_classifier(data_dto, verbose=False, remove_models=False, name_append='', execute_quantity=0):
    
    start_time = time.time()

    # clean the models on /keras_models. If done, program became slow because its gonna create it all
    remove_saved_keras_models(remove_models)

    # declaration of our tests
    results = []
    classifier_test = [
        KerasClassifierTestSet(name='simple' + name_append, creation_method=get_simple_keras_classifier, data_dto=data_dto, extra_param=None, verbose=verbose),
        KerasClassifierTestSet(name='denser' + name_append, creation_method=get_denser_keras_classifier, data_dto=data_dto, extra_param=None, verbose=verbose),

        KerasClassifierTestSet(name='denser_and_tokenizer_binary' + name_append, creation_method=get_denser_keras_classifier_with_tokenizer, data_dto=data_dto, extra_param=KerasTokenizerDTO(None, True, ' ', False, 'binary'), verbose=verbose),
        KerasClassifierTestSet(name='denser_and_tokenizer_count' + name_append, creation_method=get_denser_keras_classifier_with_tokenizer, data_dto=data_dto, extra_param=KerasTokenizerDTO(None, True, ' ', False, 'count'), verbose=verbose),
        KerasClassifierTestSet(name='denser_and_tokenizer_tfidf' + name_append, creation_method=get_denser_keras_classifier_with_tokenizer, data_dto=data_dto, extra_param=KerasTokenizerDTO(None, True, ' ', False, 'tfidf'), verbose=verbose),
        KerasClassifierTestSet(name='denser_and_tokenizer_freq' + name_append, creation_method=get_denser_keras_classifier_with_tokenizer, data_dto=data_dto, extra_param=KerasTokenizerDTO(None, True, ' ', False, 'freq'), verbose=verbose),

        KerasClassifierTestSet(name='word2vec' + name_append, creation_method=get_keras_with_word2vec_denser, data_dto=data_dto, extra_param=None, verbose=verbose),
        KerasClassifierTestSet(name='word2vec_denser' + name_append, creation_method=get_keras_with_word2vec_denser, data_dto=data_dto, extra_param=None, verbose=verbose),
    ]

    if (execute_quantity == 0):
        classifiers_to_execute = classifier_test
    else:
        classifiers_to_execute = classifier_test[:execute_quantity]

    # execution of the tests
    print("\nEvaluating Keras classifiers\n")
    for test in tqdm(classifiers_to_execute):
        classifier = test.execute()
        write_classifier_result_to_file('keras_classifiers'  + name_append + '.txt', classifier)
        results.append(classifier)

    print("\n{}# {:.2f} seconds to do keras{} {}".format(bcolors.WARNING, (time.time() - start_time), name_append, bcolors.ENDC))

    # prediction with the best classifier
    return predict_with_best(results, 'keras_prediction'  + name_append + '.txt', data_dto.data_to_predict)

# Finds the best classifier and use it to predict the texts
def predict_with_best(results, file_results_name, data_to_predict):
    results.sort(key=lambda x: x.accuracy, reverse=True)
    best_classifier = results[0]

    # Just show top 10
    print ("\n\n{}## The top 10 of classifiers: {}{}".format(bcolors.HEADER, type(best_classifier), bcolors.ENDC))
    print ("\nClassifier|accuracy")
    for classifier in results[:10]:
        print("{}|{}{}{}".format(
            classifier, 
            bcolors.WARNING, 
            classifier.accuracy, 
            bcolors.ENDC))

    print ("\n\n{}## 10 first predictions for: {}{}".format(bcolors.HEADER, type(best_classifier), bcolors.ENDC))
    print ("\nPrediction|Sentence")
    
    predictions = []
    index = 0
    # Predicting all talks with our best classifier and writting to file
    for text in data_to_predict:
        prediction = best_classifier.predict(text)[0] # 0 to remove from numpy array
        predictions.append(prediction)
    
        write_results_to_file(file_results_name, prediction, text)
        if (index < 10):
            index = index + 1
            print ("{}{}{}|{}".format(
                bcolors.WARNING,
                prediction,
                bcolors.ENDC,
                text))

    return np.array(predictions)

# Writes the result for classifier on file
def write_classifier_result_to_file(file, classifier):
    path = get_complet_path('results/' + file)
    if not os.path.exists(path):
        highscore = open(path, 'w')
        highscore.write("Classifier|stop_words|min_df|max_df|use_tfid|binary|ngram_range|emoji|sentiment|Accuracy\n")
        highscore.close()    

    highscore = open(path, 'a')
    highscore.write(str(classifier) + '|' + str(classifier.accuracy) + '\n')
    highscore.close()

# writes the results for prediction on file
def write_results_to_file(file, prediction, text):
    path = get_complet_path('results/' + file)
    if not os.path.exists(path):
        highscore = open(path, 'w')
        highscore.write("prediction|text\n")
        highscore.close()  

    highscore = open(path, 'a')
    highscore.write(prediction + '|' + text + '\n')
    highscore.close()

# evaluates the difference between our two best classifiers
def compare_classifiers_results(classifier_1, classifier_2):
    mean_between_results = np.mean(classifier_1 == classifier_2)
    print("\n\n### Difference between classifiers is: {}{:.4f}{}".format(
            bcolors.OKBLUE,
            mean_between_results,
            bcolors.ENDC
            ))

# Prepare data and call classifiers
def classify(verbose=False, remove_saved_keras_models=False):
    start_time = time.time()

    # Quantity of classifiers to execute for SkLearn. 0 does all
    execute_quantity = 0

    # First SkLearn without pre text treatment    
    pre_traitement_dto = PreTraitementDTO(apply_phrase_separation=True, replace_keywords=False, remove_punctuation=False, apply_lemm=False, filter_open_words=False)
    data_dto_1 = get_data(pre_traitement_dto)
    sk_predictions_1 = test_with_sklearn_classifiers(data_dto_1, pre_traitement_dto, name='sklearn', execute_quantity=execute_quantity, verbose=verbose)
    
    # Second SkLearn with all pre text treatment
    pre_traitement_dto = PreTraitementDTO(apply_phrase_separation=False, replace_keywords=True, remove_punctuation=True, apply_lemm=True, filter_open_words=True)
    data_dto_2 = get_data(pre_traitement_dto)
    sk_predictions_2 = test_with_sklearn_classifiers(data_dto_2, pre_traitement_dto, name='sklearn_2', execute_quantity=execute_quantity, verbose=verbose)

    ### THIS ONE IS THE BETTER ## Third SkLearn with separation, replacing words and remove punct pre text treatment 
    pre_traitement_dto = PreTraitementDTO(apply_phrase_separation=True, replace_keywords=True, remove_punctuation=True, apply_lemm=False, filter_open_words=False)
    data_dto_3 = get_data(pre_traitement_dto)
    sk_predictions_3 = test_with_sklearn_classifiers(data_dto_3, pre_traitement_dto, name='sklearn_3', execute_quantity=execute_quantity, verbose=verbose)

    # Fourth SkLearn with replacing words and remove punct pre text treatment
    pre_traitement_dto = PreTraitementDTO(apply_phrase_separation=False, replace_keywords=True, remove_punctuation=True, apply_lemm=False, filter_open_words=False)
    data_dto_4 = get_data(pre_traitement_dto)
    sk_predictions_4 = test_with_sklearn_classifiers(data_dto_4, pre_traitement_dto, name='sklearn_4', execute_quantity=execute_quantity, verbose=verbose)

    # Do all keras work related
    ke_predictions_1 = test_with_keras_classifier(data_dto_1, verbose, remove_saved_keras_models, execute_quantity=execute_quantity)

    ke_predictions_2 = test_with_keras_classifier(data_dto_2, verbose, remove_saved_keras_models, name_append='_dto2', execute_quantity=execute_quantity)

    ke_predictions_3 = test_with_keras_classifier(data_dto_3, verbose, remove_saved_keras_models, name_append='_dto3', execute_quantity=execute_quantity)

    ke_predictions_4 = test_with_keras_classifier(data_dto_4, verbose, remove_saved_keras_models, name_append='_dto4', execute_quantity=execute_quantity)

    print("\n{}### {:.2f} seconds to do it all {}".format(bcolors.WARNING, (time.time() - start_time), bcolors.ENDC))
    
    compare_classifiers_results(sk_predictions_1, ke_predictions_1)
    compare_classifiers_results(sk_predictions_2, ke_predictions_2)
    compare_classifiers_results(sk_predictions_3, ke_predictions_3)
    compare_classifiers_results(sk_predictions_4, ke_predictions_4)

if __name__ == '__main__':
    analyse_text()  # Build some graphs analysing the data we have for training
    clean_results() # Deletes the results folder to start over
    classify(verbose=False, remove_saved_keras_models=False) # TODO: Change this to receive args from command prompt
    