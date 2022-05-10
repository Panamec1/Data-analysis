import pandas as pd
import matplotlib.pyplot as plt
from itertools import chain
from typing import List
from collections import Counter
from nltk.corpus import stopwords
import nltk

def build_vocab(texts: List[str]) -> Counter:
    words = list(chain.from_iterable([text.split() for text in texts]))
    vocab = Counter(words)
    return vocab

def preprocess_text(text: str) -> str:
    text = text.lower()
    is_allowed_char = lambda c: c.isalpha() or c == ' '
    text = ''.join(list(filter(is_allowed_char, text)))
    return text

def preprocess_text(text: str) -> str:
    text = text.lower()
    is_allowed_char = lambda c: c.isalpha() or c == ' '
    text = ''.join(list(filter(is_allowed_char, text)))
    
    is_stopword = lambda word: word not in eng_stopwords
    text = ' '.join(list(filter(is_stopword, text.split())))
    return text


df1 = pd.read_csv('shopee_reviews.csv')
texts = df1.text.tolist()
vocab = build_vocab(texts)
print(len(vocab))
print(list(vocab.items())[:30])

texts = [preprocess_text(text) for text in df.text.tolist()]
vocab = build_vocab(texts)
print(len(vocab))
print(list(vocab.items())[:30])
