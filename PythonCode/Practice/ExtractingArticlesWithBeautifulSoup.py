from urllib.request import urlopen
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.probability import FreqDist
from heapq import nlargest
from collections import defaultdict

articleURL = "https://www.washingtonpost.com/news/the-switch/wp/2016/10/18/the-pentagons-massive-new-telescope-is-designed-to-track-space-junk-and-watch-out-for-killer-asteroids/?noredirect=on&utm_term=.33e12e03ecf6"


# print(soup.find('article').text)

def getText(url):
    page = urlopen(url).read().decode('utf-8', 'ignore')
    soup = BeautifulSoup(page, "lxml")
    text = ' '.join(map(lambda p: p.text, soup.find_all('article')))
    return text
    # text.encode('ascii', errors='replace').replace("?", " ")


myArticle = getText(articleURL)
print(myArticle)

def summarize(text, n):

    sentences = sent_tokenize(text)
    assert n <= len(sentences)
    words = word_tokenize(text.lower())

    _stopwords = set(stopwords.words('english') + list(punctuation))
    wordsInSentences = [word for word in words if word not in _stopwords]
    # print(wordsInSentences)

    freq = FreqDist(wordsInSentences)
    nlargest(10, freq, key=freq.get)

    ranking = defaultdict(int)

    for i, sent in enumerate(sentences):
        for w in word_tokenize(sent.lower()):
            if w in freq:
                ranking[i] += freq[w]
    # print(ranking)

    sentences_index = nlargest(4, ranking, key=ranking.get)
   # print(sentences_index)

    return [sentences[j] for j in sorted(sentences_index)]

print(summarize(myArticle, 3))

