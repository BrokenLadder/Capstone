from nltk.stem import WordNetLemmatizer

# lemmatizing is a bit better than stemming

lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize("cats"))
print(lemmatizer.lemmatize("geese"))
print(lemmatizer.lemmatize("cacti"))
print(lemmatizer.lemmatize("rocks"))
print(lemmatizer.lemmatize("python"))

# pos = a tells it is an adverb
print(lemmatizer.lemmatize("better", pos="a"))
# default pos or part of speech is defaulted at "n" or noun
print(lemmatizer.lemmatize("better", pos="a"))

