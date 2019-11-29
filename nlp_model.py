from nltk.tokenize import sent_tokenize as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize as wt
from nltk.stem import SnowballStemmer
import numpy as np


class NLPModel(object):

    def __init__(self, language='russian'):
        self.language = language
        self.stopwords = stopwords.words(language)
        self.stopwords.extend(['что', 'это', 'так', 'вот', 'быть', 'как', 'в', '_', '–', 'к', 'на', '...', '(', ')',
                               '[', ']', '{', '}', ',', '.', '!', '@', '#', '$', '%', '^', '&', '*', '~', '\\', '|',
                               ':', ';', '\'', '\"', '+', '=', '№', '?', '<', '>', '«', '»'])
        self.stopwords = set(self.stopwords)
        self.stemmer = SnowballStemmer(language=language)
        self.vocab = []
        self.bows = []

    def fit(self, texts):
        for text in texts:
            self.vocab.extend(self.get_words(text))
        self.vocab = set(self.vocab)
        self.bows = [(text, self.get_bow(self.get_words(text))) for text in texts]

    def predict(self, text):
        words = self.get_words(text)
        bow = self.get_bow(words)
        table = []
        for self_text_bow in self.bows:
            text, self_bow = self_text_bow
            table.append((text, (np.array(list(bow.values())) * np.array(list(self_bow.values()))).sum()))

        max_score = 0
        max_text = []
        for text_score in table:
            text, score = text_score
            if score > max_score:
                max_score = score
                max_text = [text]
            elif score == max_score:
                max_text.append(text)

        return max_score, max_text

    def sent_tokenize(self, text):
        return st(text, self.language)

    def word_tokenize(self, text, preserve_line=False):
        return wt(text, self.language, preserve_line)

    def remove_stopwords(self, words):
        return [word for word in words if word not in self.stopwords]

    def stem_words(self, words):
        return [self.stemmer.stem(word) for word in words]

    def get_words(self, text):
        return self.stem_words(self.remove_stopwords(self.word_tokenize(text, preserve_line=False)))

    def get_bow(self, words):
        bow = {key: 0 for key in self.vocab}
        for word in words:
            if word in bow:
                bow[word] += 1
        return bow
