from nltk.corpus import wordnet

syns = wordnet.synsets("program")

# synset
print(syns)
# just the word
print(syns[0].lemmas()[0].name())
#examples
print(syns[0].definition())


synonyms = []
antonyms = []
for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

print(set(synonyms))
print(set(antonyms))

# ship.n.01 = ship, noun, 1st in the dictionary
word1 = wordnet.synset("ship.n.01")
word2 = wordnet.synset("boat.n.01")

# it will check the similarity of the two words and give us a percentage
print(word1.wup_similarity(word2))
# these words are 90% similar

# I can add more to my capstone, by showing a "how similar is this paper to this paper" function