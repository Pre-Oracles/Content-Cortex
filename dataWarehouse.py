import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from pymongo import MongoClient
from datetime import datetime

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation and non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Lowercase
    text = text.lower()
    # Tokenize
    words = word_tokenize(text)
    # Remove stopwords and lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Connect to MongoDB
client = MongoClient('localhost', 27017)
db = client['comment_database']
collection = db['comments']

# Function to insert a comment into MongoDB
def insert_comment(comment_text, label):
    processed_text = preprocess_text(comment_text)
    comment = {
        "comment_text": processed_text,
        "label": label,
        "created_at": datetime.utcnow()
    }
    collection.insert_one(comment)

# Example usage
text = "This is an example of a comment! It's quite offensive."
insert_comment(text, 1)  # Assuming 1 indicates offensive and 0 indicates non-offensive
