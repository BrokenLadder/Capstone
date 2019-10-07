import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
# tokenizing- word tokenizers and sentence tokenizers
# corpora- body of text. ex: medical journals, presidential speeches, twitter posts [English Language]
# lexicon- words and their meanings
# investor-speak vs regular english-speak
# investor speak 'bull' = someone who is positive in the market
# english speak 'bull' = an animal

example_text = "Hello Mr. Smith, how are you doing today? The weather" \
               " is great and python is awesome. The sky is pinkish blue," \
               " you should not eat cardboard"

# print(sent_tokenize(example_text))
# print(word_tokenize(example_text))

for i in word_tokenize(example_text):
    print(i)

