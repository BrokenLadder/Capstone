import nltk


positiveReviewsFileName = "C:/Users/Gabe B/Desktop/NeumontWork/Q42018/Capstone/Resources/rt-polaritydata/rt-polarity.pos"
negativeReviewsFileName = "C:/Users/Gabe B/Desktop/NeumontWork/Q42018/Capstone/Resources/rt-polaritydata/rt-polarity.neg"
with open(positiveReviewsFileName, 'r') as f:
    positiveReviews = f.readlines()
with open(negativeReviewsFileName, 'r') as f:
    negativeReviews = f.readlines()
testTrainingSplitIndex = 2500
testNegativeReviews = negativeReviews[testTrainingSplitIndex + 1:]
testPositiveReviews = positiveReviews[testTrainingSplitIndex + 1:]
trainingNegativeReviews = negativeReviews[:testTrainingSplitIndex]
trainingPositiveReviews = positiveReviews[:testTrainingSplitIndex]


def getVocabulary():
    positiveWordList = [word for line in trainingPositiveReviews for word in line.split()]
    negativeWordList = [word for line in trainingNegativeReviews for word in line.split()]
    allWordList = [item for sublist in [positiveWordList, negativeWordList] for item in sublist]
    allWordSet = list(set(allWordList))
    vocabulary = allWordSet
    return vocabulary

def getTrainingData():
    negTaggedTrainingReviewList = [{'review' :oneReview.split(), 'label':'negative'} for oneReview in trainingNegativeReviews]
    posTaggedTrainingReviewList = [{'review':oneReview.split(),'label':'positive'} for oneReview in trainingPositiveReviews]
    fullTaggedTrainingData = [item for sublist in [posTaggedTrainingReviewList, negTaggedTrainingReviewList] for item in sublist]
    trainingData = [(review['review'], review['label']) for review in fullTaggedTrainingData]
    return trainingData


vocabulary = getVocabulary()
trainingData = getTrainingData()

def extract_features(review):
    review_words = set(review)
    features={}
    for word in vocabulary:
        features[word] = (word in review_words)
    return features


def getTrainedNaiveBayesClassifier(extract_features, trainingData):
    trainingFeatures=nltk.classify.apply_features(extract_features, trainingData)
    trainedNBClassifer = nltk.NaiveBayesClassifier.train(trainingFeatures)
    return trainedNBClassifer


trainedNBClassifier = getTrainedNaiveBayesClassifier(extract_features,trainingData)

def naiveBayesSentimentCalculator(review):
    problemInstance = review.split()
    problemFeatures = extract_features(problemInstance)
    return trainedNBClassifier.classify(problemFeatures)

#print(naiveBayesSentimentCalculator("What an awesome movie"))
#print(naiveBayesSentimentCalculator("What a terrible review"))

def getTestReviewSentiments(naiveBayesSentimentCalculator):
    testNegResults = [naiveBayesSentimentCalculator(review) for review in testNegativeReviews]
    testPosResults = [naiveBayesSentimentCalculator(review) for review in testNegativeReviews]
    labelToNum = {'positive':1, 'negative':-1}
    numericNegResults = [labelToNum[x] for x in testNegResults]
    numericPosResults = [labelToNum[x] for x in testPosResults]
    return {'results-on-positive':numericPosResults, 'results-on-negative':numericNegResults}

def runDiagnostics(reviewResult):
    positiveReviewsResult = reviewResult['results-on-positive']
    negativeReviewsResult = reviewResult['results-on-negative']
    numTruePositive = sum(x > 0 for x in positiveReviewsResult)
    numTrueNegative = sum(x < 0 for x in negativeReviewsResult)
    pctTruePositive = float(sum(x > 0 for x in positiveReviewsResult)) / len(positiveReviewsResult)
    pctTrueNegative = float(sum(x < 0 for x in negativeReviewsResult)) / len(negativeReviewsResult)
    totalAccurate = numTruePositive + numTrueNegative
    total = len(positiveReviewsResult) + len(negativeReviewsResult)
    print("Accuracy on positive reviews = " + "%.2f" % (pctTruePositive*100))
    print("Accuracy on negative reviews = " + "%.2f" % (pctTrueNegative*100))
    print("Overall Accuracy = " + "%.2f" % (totalAccurate*100/total) + "%")

runDiagnostics(getTestReviewSentiments(naiveBayesSentimentCalculator))