import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.collocations import *

text = "Mary had a little lamb. Her fleece was white as snow"
sentences = sent_tokenize(text)
print(sentences)
words = [word_tokenize(sentence) for sentence in sentences]
print(words)
customStopWords = set(stopwords.words('english') + list(punctuation))
wordsWithoutStopWords = [word for word in word_tokenize(text) if word not in customStopWords]
print(wordsWithoutStopWords)

bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(wordsWithoutStopWords)
sorted(finder.ngram_fd.items())