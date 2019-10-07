
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk

for ss in wn.synsets('bass'):
    print(ss, ss.definition())

sensel = lesk(word_tokenize("Sing in a lower tone, along with the bass"), 'bass')
print(sensel, sensel.definition())

sense2 = lesk(word_tokenize("This sea bass was really hard to catch"), 'bass')
print("sense2")
print(sense2, sense2.definition())