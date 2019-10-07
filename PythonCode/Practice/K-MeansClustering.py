from sklearn.feature_extraction.text import TfidfVectorizer
import urllib.request
from urllib.request import urlopen
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from sklearn.neighbors import KNeighborsClassifier
from collections import defaultdict
from string import punctuation
from heapq import nlargest
import nltk

def getAllDoxyDonkeyPosts(url, links):
    request = urllib.request.Request(url)
    response = urlopen(request)
    soup = BeautifulSoup(response, "lxml")
    for a in soup.find_all('a'):
        try:
            url = a['href']
            title = a['title']
            if title == "Older Posts":
                print(title, url)
                links.append(url)
                getAllDoxyDonkeyPosts(url, links)
        except:
            title = ""
    return

def getDoxyDonkeyText(testURL):
    request = urllib.request.Request(testURL)
    response = urlopen(request)
    soup = BeautifulSoup(response, "lxml")
    mydivs = soup.find_all("div", {"class": 'post-body'})
    posts = []
    for div in mydivs:
        posts+=map(lambda p: p.text.encode('ascii', errors='replace').replace(b"?", b" "), div.findAll("li"))
    return posts

blogUrl = "http://doxydonkey.blogspot.in"
links = []
getAllDoxyDonkeyPosts(blogUrl, links)
doxyDonkeyPosts = []
for link in links:
    doxyDonkeyPosts += getDoxyDonkeyText(link)
print(doxyDonkeyPosts)
vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english')

x = vectorizer.fit_transform(doxyDonkeyPosts)
print(x)

print(x[0])
# n_clusters = number of clusters you want, or groups
# init = specifies the algorithm we are using, or method of choosing initial centroids
# Maximum number of iterations (in case of no convergence)
km = KMeans(n_clusters=3, init = 'k-means++', max_iter=100, n_init=1, verbose=True)
km.fit(x)

print(np.unique(km.labels_, return_counts=True))

text={}
for i, cluster in enumerate(km.labels_):
    oneDocument = doxyDonkeyPosts[i]
    if cluster not in text.keys():
        text[cluster] = oneDocument
    else:
        text[cluster] += oneDocument
_stopwords = set(stopwords.words('english') + list(punctuation)+["million","billion","year"])
keywords = {}
counts = {}
for cluster in range(3):
    word_sent = word_tokenize(text[cluster].lower())
    word_sent = [word for word in word_sent if word not in _stopwords]
    freq = FreqDist(word_sent)
    keywords[cluster] = nlargest(100,freq, key=freq.get)
    counts[cluster] = freq

unique_keys={}
for cluster in range():
    other_clusters=list(set(range(3))-set([cluster]))
    keys_other_clusters=set(keywords[other_clusters[0]]).union(set(keywords[other_clusters[1]]))
    unique = set(keywords[cluster])-keys_other_clusters
    unique_keys[cluster]=nlargest(10, unique, key=counts[cluster].get)

# classifier = KNeighborsClassifier()
# classifier.fit(X,km.labels_)