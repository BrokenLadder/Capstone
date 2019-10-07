# This sentiment analysis is polarized, it is either Negative, or Positive
# Which is obviously much simpler than real semantic analysis

import nltk
import random
from nltk.tokenize import word_tokenize
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
            return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]



# similar and less complex than a one liner
# for category in movie_reviews.categories():
#     for fileid in movie_reviews.fileids(category):
#         documents.append(list(movie_reviews.words(fileid)), category))

# random.shuffle(documents)

# print(documents[1])

all_words = []
for w in movie_reviews.words():
    # lowercase words to standardize them
    all_words.append(w.lower())

# To use in nltk we have to turn the movie reviews into a
# nltk Frequency Distribution



all_words = nltk.FreqDist(all_words)
# now we can use nltk's most_common function to show the top
# 15 most common words, remember nltk by default consideres
# punctuation and junk words like the and a to be real words
# print(all_words.most_common(15))
# prints amount of occurances of "stupid"
# print(all_words["stupid"])

# Frequency distribution literally orders the words based on
# how many times the words show up

word_features = list(all_words.keys())[:3000]


def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
    # If the 3000 words is within this document this will be true
    # if not it will be false
        features[w] = (w in words)
    return features

# This gives us negative movie reviews and runs it through our method
# print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]

# training set has to be different from our testing set to remain unbiased
# we will take the first 1900 in the featuresets for our training set
#Positive data example
# training_set = featuresets[:1900]
# testing_set = featuresets[1900:]

#Negative data example
training_set = featuresets[100:]
testing_set = featuresets[:100]

# saving a classifier
# wb means write in bytes
# save_classifier = open("naivebayes.pickle", "wb")
# pickle.dump(classifier, save_classifier)
# save_classifier.close()

# loading A classifier
#rb = read in bytes
classifier_f = open("naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

# posterior = prior occurences x likely / evidence
# this is the likeliness that something will be negative or positive
# this is using the naiveBayes Algorithm
# [Posterior Probability]  P(c/x) = [likelihood] P(x/c)P(c)  [Class Prior Probability]
#                                               -------------
#                                                    P(x)  [Predictor Probability]

classifier = nltk.NaiveBayesClassifier.train(training_set)





print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set)) * 100)
classifier.show_most_informative_features(15)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier Algo accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set)) * 100)


BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier Algo accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set)) * 100)

# LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
# LogisticRegression_classifier.train(training_set)
# print("LogisticRegression_classifier Algo accuracy percent:", (nltk.classify.accuracy(LogisticRegression, testing_set)) * 100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier Algo accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set)) * 100)

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC_classifier Algo accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set)) * 100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier Algo accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set)) * 100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier Algo accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set)) * 100)

voted_classifier = VoteClassifier(classifier, MNB_classifier, BernoulliNB_classifier, LinearSVC_classifier, NuSVC_classifier)
# print("Voted_classifier Algo accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set)) * 100)
# print("Classification:", voted_classifier.classify(testing_set[0][0]), "Confidence %:", voted_classifier.confidence(testing_set[0][0])*100)
# print("Classification:", voted_classifier.classify(testing_set[0][0]), "Confidence %:", voted_classifier.confidence(testing_set[2][0])*100)
# print("Classification:", voted_classifier.classify(testing_set[0][0]), "Confidence %:", voted_classifier.confidence(testing_set[3][0])*100)
# print("Classification:", voted_classifier.classify(testing_set[0][0]), "Confidence %:", voted_classifier.confidence(testing_set[4][0])*100)
# print("Classification:", voted_classifier.classify(testing_set[0][0]), "Confidence %:", voted_classifier.confidence(testing_set[5][0])*100)




