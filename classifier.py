import abc
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

# Little class used to help evaluation of SkLearn and Keras classifiers
class Classifier(object, metaclass=abc.ABCMeta):
    name = ""
    @abc.abstractmethod
    def predict(self, text):
        raise NotImplementedError('users must define __str__ to use this base class')

    def __str__(self):
        return self.name

class DataDTO():
    def __init__(self, data_train, data_test, target_train, target_test, target_names, data_to_predict, vocab_size=15000): 
        self.data_train = data_train 
        self.data_test = data_test 
        self.target_train = target_train
        self.target_test = target_test
        self.target_names = target_names
        self.vocab_size = vocab_size    
        self.data_to_predict = data_to_predict

class PreTraitementDTO():
    def __init__(self, apply_phrase_separation=True, replace_keywords=False, remove_punctuation=False, apply_lemm=False, filter_open_words=False):
        self.apply_phrase_separation = apply_phrase_separation
        self.replace_keywords = replace_keywords
        self.remove_punctuation = remove_punctuation
        self.apply_lemm = apply_lemm
        
        if (self.apply_lemm):
            self.filter_open_words = filter_open_words
            self.lemm = WordNetLemmatizer()
    
    def penn_to_wn(self, tag):
        if tag.startswith('J'):
            return wn.ADJ
        elif tag.startswith('N'):
            return wn.NOUN
        elif tag.startswith('R'):
            return wn.ADV
        elif tag.startswith('V'):
            return wn.VERB
        return None