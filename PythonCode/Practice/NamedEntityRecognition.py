import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
tokenized = custom_sent_tokenizer.tokenize(sample_text)

"""
Named Entity Type Examples
ORGANIZATION                Georgia-Pacific Corp., WHO
PERSON                      President Obama, Eddy Bonte
LOCATION                    Murray River, Mount Everest
DATE                        June, 2008-06-29
TIME                        two fifty am, 1:30pm
MONEY                       175 million Canadian Dollars, GBP 10.40
PERCENT                     twenty pct, 18.75%
FACILITY                    Washington Monument, Stonehenge
GPE(GEOPOLITACAL ENTITY)    South East Asia, Midlothian
"""


def process_content():
    try:
        for i in tokenized[5:]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            namedEnt = nltk.ne_chunk(tagged)
            # This will just show that it is a named Entity and not what
            # it is affiliated with.
            # namedEntBinary = nltk.ne_chunk(tagged, binary=True)
            namedEnt.draw()

    except Exception as e:
        print(str(e))


process_content()
