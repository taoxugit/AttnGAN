'''
Info that we have form the original paper:

Textual Representation: We use the raw Wikipedia articles collected by [10] for both benchmark datasets. Text articles are first tokenized into words, the stop words are removed, and porter stemmor [31] is applied to reduce inflected words to their word stem. Then, Term FrequencyInverse Document Frequency(TF-IDF) feature vector [38] is extracted. The dimensionalities of TF-IDF features for CUB and NAB are 7551 and 13217.

So we need to:
-tokenize
-remove stop-words
-apply porter stemming
-calc tf-idf
'''

import pickle
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


class TextProcessor:
    def __init__(self, tfidf_vectorizer_path):
        # It's not the best idea because we split complex words like 'medium-sized' 
        # into two with loosing original meaning
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.stemmer = PorterStemmer()
        with open(tfidf_vectorizer_path, 'rb') as file:
            self.tfidf_vectorizer = pickle.load(file)

    def preprocess_text(self, text):
        tokens = self.tokenizer.tokenize(text)
        filtered_tokens = [token 
                           for token in tokens 
                           if token not in stopwords.words('english') and token.isalpha()]
        final_tokens = [self.stemmer.stem(token) for token in filtered_tokens]
        return ' '.join(final_tokens)
    
    def get_tfidf(self, sentence):
        return self.tfidf_vectorizer.transform([sentence]).toarray().flatten()
