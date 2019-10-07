import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment import vader


positiveReviewsFileName = "C:/Users/Gabe B/Desktop/NeumontWork/Q42018/Capstone/Resources/rt-polaritydata/rt-polarity.pos"
with open(positiveReviewsFileName, 'r') as f:
    positiveReviews = f.readlines()

NegativeReviewsFileName = "C:/Users/Gabe B/Desktop/NeumontWork/Q42018/Capstone/Resources/rt-polaritydata/rt-polarity.neg"
with open(NegativeReviewsFileName, 'r') as f:
    negativeReviews = f.readlines()

# print(positiveReviews)
# print(negativeReviews)
# print(len(negativeReviews))

sia = vader.SentimentIntensityAnalyzer()


def vadersentiment(review):
    return sia.polarity_scores(review)


[vadersentiment(onePositiveReview) for onePositiveReview in positiveReviews]
[vadersentiment(oneNegativeReview) for oneNegativeReview in negativeReviews]


def getReviewSentiments(sentimentCalculator):
    negReviewResult = [sentimentCalculator(oneNegativeReview) for oneNegativeReview in negativeReviews]
    posReviewResult = [sentimentCalculator(onePositiveReview) for onePositiveReview in positiveReviews]
    return{'results-on-positive':posReviewResult, 'results-on-negative':negReviewResult}


def runDiagnostics(reviewResult):
    positiveReviewsResult = reviewResult['results-on-positive']
    negativeReviewsResult = reviewResult['results-on-negative']
    pctTruePositive = float(sum(x > 0 for x in positiveReviewsResult)) / len(positiveReviewsResult)
    pctTrueNegative = float(sum(x < 0 for x in negativeReviewsResult)) / len(negativeReviewsResult)
    totalAccurate = float(sum(x > 0 for x in positiveReviewsResult)) + float(sum(x < 0 for x in negativeReviewsResult))
    total = len(positiveReviewsResult) + len(negativeReviewsResult)
    print("Accuracy on positive reviews = " + "%.2f" % (pctTruePositive*100))
    print("Accuracy on negative reviews = " + "%.2f" % (pctTrueNegative*100))
    print("Overall Accuracy = " + "%.2f" % (totalAccurate*100/total) + "%")


vaderResults = getReviewSentiments(vadersentiment)
print(vaderResults.keys())
print(len(vaderResults['results-on-negative']))
runDiagnostics(getReviewSentiments(vadersentiment))

