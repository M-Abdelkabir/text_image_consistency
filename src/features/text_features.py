import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

def extract_text_features(texts, max_features=5000, vectorizer=None):
    stop_words = set(stopwords.words('english'))
    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            ngram_range=(1,2),
            max_features=max_features,
            stop_words=stop_words,
            lowercase=True
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
    else:
        tfidf_matrix = vectorizer.transform(texts)
    
    extra_features = []
    for doc in texts:
        words = nltk.word_tokenize(doc.lower())
        punct_ratio = sum(1 for c in doc if c in '!?.,;') / max(len(doc), 1)
        upper_count = sum(1 for w in words if w.isupper())
        extra_features.append([len(words), punct_ratio, upper_count])
    
    return tfidf_matrix, np.array(extra_features), vectorizer
