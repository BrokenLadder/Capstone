# feature extraction using the bag of words model
import urllib.request
from urllib.request import urlopen
from bs4 import BeautifulSoup

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

