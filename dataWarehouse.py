import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from pymongo import MongoClient
from datetime import datetime

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

client = MongoClient('localhost', 27017)
db = client['comment_database']
collection = db['comments']

def insert_comment(comment_text, label):
    processed_text = preprocess_text(comment_text)
    comment = {
        "comment_text": processed_text,
        "label": label,
        "created_at": datetime.now()
    }
    collection.insert_one(comment)


text = "This is an example of a comment! It's quite offensive."
insert_comment(text, 1)  
