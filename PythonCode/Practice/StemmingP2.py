import nltk
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import word_tokenize
text = "Marry closed on closing night when she was in the mood to close"
st = LancasterStemmer()
stemmedWords =[st.stem(word) for word in word_tokenize(text)]
print(stemmedWords)
print(nltk.pos_tag(word_tokenize(text)))
