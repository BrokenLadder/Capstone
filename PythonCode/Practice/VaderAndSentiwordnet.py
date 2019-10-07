# Vader is a rule based binary classifier
# Valence Aware Dictionary for sEnteiment Reasoning
import nltk
from nltk.sentiment import vader
sia = vader.SentimentIntensityAnalyzer()
sia.polarity_scores("What a terrible restaurant")
print(sia.polarity_scores("What a terrible restaurant"))
print("Smiley Face:" + str(sia.polarity_scores(":)")))
# Vader can tell Negation, the not is taken into affect
print(sia.polarity_scores("The food was good"))
print(sia.polarity_scores("The food not was good"))
# Vader is sensitive to capitilization
print(sia.polarity_scores("Food was good"))
print(sia.polarity_scores("Food was GOOD"))
# Vader understands depreciating comments
print(sia.polarity_scores("I usually hate seafood, but I liked this"))