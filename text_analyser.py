import os
import pandas as pd
import numpy as np
import string

import plotly
from plotly import graph_objs
import emoji

from nltk import FreqDist, re
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords 

from flashtext import KeywordProcessor

from nltk.tokenize import TweetTokenizer
from nltk import pos_tag
from nltk.corpus import wordnet as wn

# Custom imports
from utility import get_complet_path

keyword_processor = KeywordProcessor()
keyword_processor.add_keyword('ur', 'your')
keyword_processor.add_keyword('youre', 'you are')
keyword_processor.add_keyword('u', 'you')
keyword_processor.add_keyword('r', 'are')
keyword_processor.add_keyword('dont', r'''don't''')
keyword_processor.add_keyword('im', r'''i'm''')
keyword_processor.add_keyword('thats', r'''that's''')
keyword_processor.add_keyword('cant', r'''can't''')
keyword_processor.add_keyword('whats', r'''what's''')

def replace_punctuation_in_text(text):
    return text.replace(',', '').replace('..', ' ').replace('.','').replace('\' ',' ').replace('\"',"")

def replace_keywords_in_text(text):
    text = text.replace('İ', 'I') # Strange bug on the library can't work around this letter
    text = keyword_processor.replace_keywords(text)
    return text

def lemmatize(text, pre_traitement_dto):
    tokenizer = TweetTokenizer()
    
    tagged_sentence = pos_tag(tokenizer.tokenize(text))
    lemmatized_text = ""
    for word, tag in tagged_sentence:
        wn_tag = pre_traitement_dto.penn_to_wn(tag)
        if pre_traitement_dto.filter_open_words and wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV, wn.VERB):
            continue

        if (wn_tag != None):
            lemma = pre_traitement_dto.lemm.lemmatize(word, pos=wn_tag)
        else:
            lemma = pre_traitement_dto.lemm.lemmatize(word)

        if not lemma:
            lemmatized_text = lemmatized_text + word + " "
        else:
            lemmatized_text = lemmatized_text + lemma + " "
        
    return lemmatized_text.strip()

def analyse_text(verbose=False):
    
    # Read our train data from file
    df = pd.read_table(get_complet_path('data/train.txt'), sep='\t', header=0, usecols=[1,2,3,4])
    df.fillna('', inplace=True)
    
    # Threat all the columns as one
    df['text'] = df[['turn1', 'turn2', 'turn3']].apply(lambda x: " ".join(x) , axis=1)
    df.drop('turn1', inplace=True, axis=1)
    df.drop('turn2', inplace=True, axis=1)
    df.drop('turn3', inplace=True, axis=1)

    full_text = " ".join(df['text'].tolist()).lower()

    #get_sentiment_distribution(df, verbose)
    #get_word_count(full_text, verbose)
    #get_word_count(full_text, use_stop_words=True, verbose=verbose)
    #get_word_count(full_text, use_stop_words=True, replace_keywords=True, verbose=verbose)
    #get_word_count(full_text, use_stop_words=True, replace_keywords=True, remove_punctuation=True, verbose=verbose)
    #get_emoji_count(full_text)

def get_emoji_count(full_text):
    emojis = ''.join(c for c in full_text if c in emoji.UNICODE_EMOJI)

    tokenizer = TweetTokenizer()
    emoji_list = tokenizer.tokenize(emojis)

    fdist = FreqDist(emoji_list)
    most_common = fdist.most_common(50)

    zipped_fdist = list(zip(*most_common))
    dist = [
        graph_objs.Bar(
            x=zipped_fdist[0],
            y=zipped_fdist[1],
    )]
    plotly.offline.plot({"data":dist, "layout":graph_objs.Layout(title="Top emojis")}
        , filename='graphs/emoji_count.html')

# Creates a graph with top 50 word count - do some filtering if needed
def get_word_count(full_text, use_stop_words=False, replace_keywords=False, remove_punctuation=False, verbose=False):

    if (replace_keywords):
        full_text = replace_keywords_in_text(full_text)

    if (remove_punctuation):
        full_text = replace_punctuation_in_text(full_text)
        #table = str.maketrans({key: None for key in string.punctuation})
        #full_text = full_text.translate(table)

    tokenizer = TweetTokenizer()
    word_list = tokenizer.tokenize(full_text)

    if (use_stop_words):
        stop_words = set(stopwords.words('english')) 
        word_list = [w for w in word_list if not w in stop_words] 
    
    fdist = FreqDist(word_list)
    most_common = fdist.most_common(100)

    if (verbose):
        print(most_common)

    zipped_fdist = list(zip(*most_common))
    dist = [
        graph_objs.Bar(
            x=zipped_fdist[1],
            y=zipped_fdist[0],
            orientation="h"
    )]
    plotly.offline.plot({"data":dist, "layout":graph_objs.Layout(title="Top words in built wordlist")}
        , filename='graphs/word_count_stop' + str(use_stop_words) + '_replace' + str(replace_keywords) + '_punct' + str(remove_punctuation) + '.html')
    
# Get distribution by sentiment
def get_sentiment_distribution(df, verbose):

    target_names = df.label.unique()
    target_count = []
    
    for sentiment in target_names:
        count = len(df[df["label"] == sentiment])
        target_count.append(count)

        if (verbose):
            print("{}: {}".format(sentiment, count))
    
    dist = [
        graph_objs.Bar(
            x=target_names,
            y=target_count,
        )]

    plotly.offline.plot({"data":dist, "layout":graph_objs.Layout(title="Sentiment type distribution in training set")}
        , filename='graphs/sentiment_distribution.html')

