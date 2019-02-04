import pandas as pd
import numpy as np
import nltk
import re
import codecs
import vaderSentiment
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

def sentiment_analyzer_scores(text):
    score = analyser.polarity_scores(text)
    lb = score['compound']
    if lb >= 0.05:
        return 1
    elif (lb > -0.05) and (lb < 0.05):
        return 0
    else:
        return -1

import spacy
from collections import Counter
nlp=spacy.load('en')

def postag(text):
    doc=nlp(text)
    pos=[(i, i.pos_) for i in doc]
    counts=Counter(tag for word, tag in pos)
    return counts

def sent_word_tok(text):
    sents=nltk.sent_tokenize(text)
    words=nltk.word_tokenize(text)
    num_sents=len(sents)
    num_words=len(words)
    
    if num_words == 0:
        avg_word_sent == 0
    else:
        avg_word_sent = num_words/num_sents
    return {'num_word': num_words, 'num_sent': num_sents, 'avg_words_sent': avg_word_sent}

def standardize_text(text_field):
    text_field = text_field.replace(r"http\S+", "")
    text_field = text_field.replace(r"http", "")
    text_field = text_field.replace(r"@\S+", "")
    text_field = text_field.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    text_field = text_field.replace(r"@", "at")
    text_field = text_field.replace(r"\"", "")
    text_field = text_field.lower()
    return text_field

