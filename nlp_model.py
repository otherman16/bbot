from nltk.tokenize import sent_tokenize as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize as wt
from nltk.stem import SnowballStemmer
import numpy as np


class SiteModel(object):

    def __init__(self, language='russian'):
        self.language = language
        self.stopwords = stopwords.words(language)
        self.stopwords.extend(['что', 'это', 'так', 'вот', 'быть', 'как', 'в', '_', '–', 'к', 'на', '...', '(', ')',
                               '[', ']', '{', '}', ',', '.', '!', '@', '#', '$', '%', '^', '&', '*', '~', '\\', '|',
                               ':', ';', '\'', '\"', '+', '=', '№', '?', '<', '>', '«', '»'])
        self.stopwords = set(self.stopwords)
        self.stemmer = SnowballStemmer(language=language)
        self.fmt = "{}.ру"

    def __call__(self, text):
        text_words = self.get_words(text)
        longest_word = max(text_words, key=len)
        return self.fmt.format(longest_word)

    def sent_tokenize(self, text):
        return st(text, self.language)

    def word_tokenize(self, text, preserve_line=False):
        return wt(text, self.language, preserve_line)

    def remove_stopwords(self, words):
        return [word for word in words if word not in self.stopwords]

    def stem_words(self, words):
        return [self.stemmer.stem(word) for word in words]

    def get_words(self, text):
        words = self.word_tokenize(text, preserve_line=False)
        clear_words = self.remove_stopwords(words)
        norm_words = self.stem_words(clear_words)
        return norm_words



class NLPModel(object):

    def __init__(self, language='russian'):
        self.language = language
        self.stopwords = stopwords.words(language)
        self.stopwords.extend(['что', 'это', 'так', 'вот', 'быть', 'как', 'в', '_', '–', 'к', 'на', '...', '(', ')',
                               '[', ']', '{', '}', ',', '.', '!', '@', '#', '$', '%', '^', '&', '*', '~', '\\', '|',
                               ':', ';', '\'', '\"', '+', '=', '№', '?', '<', '>', '«', '»'])
        self.stopwords = set(self.stopwords)
        self.stemmer = SnowballStemmer(language=language)
        self.vocab = set()
        self.bows = []
        self.texts = []

    def fit(self, texts):
        self.vocab = set()
        self.bows = []
        self.texts = texts
        for text in self.texts:
            self.vocab.update(self.get_words(text))
        self.bows = [self.get_bow(self.get_words(text)) for text in texts]

    def predict(self, text):
        text_words = self.get_words(text)
        text_bow = self.get_bow(text_words)
        scores = [(np.array(bow) * np.array(text_bow)).sum() for bow in self.bows]

        max_score = max(scores)
        if max_score == 0:
            return 0, []

        max_score_index = [i for i, score in enumerate(scores) if score == max_score]
        max_score_text = [self.texts[i] for i in max_score_index]

        return max_score, max_score_text

    def sent_tokenize(self, text):
        return st(text, self.language)

    def word_tokenize(self, text, preserve_line=False):
        return wt(text, self.language, preserve_line)

    def remove_stopwords(self, words):
        return [word for word in words if word not in self.stopwords]

    def stem_words(self, words):
        return [self.stemmer.stem(word) for word in words]

    def get_words(self, text):
        words = self.word_tokenize(text, preserve_line=False)
        clear_words = self.remove_stopwords(words)
        norm_words = self.stem_words(clear_words)
        return norm_words

    def get_bow(self, words):
        bow = {key: 0 for key in self.vocab}
        for word in words:
            if word in bow:
                bow[word] += 1
        return list(bow.values())
