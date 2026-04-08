import pandas as pd
from tqdm import tqdm
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d', ' ', text)
    text = word_tokenize(text)
    stopwords_set = set(nltk.corpus.stopwords.words('english'))
    text = [t for t in text if t not in stopwords_set]
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(t) for t in text]
    text = [t for t in text if t not in stopwords_set]
    return ' '.join(text)


data = pd.read_csv('reviews.csv')
tqdm.pandas()
data['label'] = data['sentiment'].progress_apply(lambda label: 1 if label == 'positive' else 0)
data['processed'] = data['review'].progress_apply(preprocess_text)
data[['processed', 'label']].to_csv('reviews_preprocessed.csv', index=False, header=True)







