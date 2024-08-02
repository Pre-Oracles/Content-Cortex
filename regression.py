import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
import matplotlib.pyplot as plt

nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    text = ' '.join(tokens)
    return text

df = pd.read_csv("dataset_1.csv")
vectorizer = TfidfVectorizer()
df['tweet'] = df['tweet'].apply(preprocess_text)
X = vectorizer.fit_transform(df['tweet'])

df['hate_speech'] = df['hate_speech'].apply(lambda x: 0 if x == 0 else 1)
y = df['hate_speech']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

cm = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix:\n{cm}')

feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]

feature_importance = list(zip(feature_names, coefficients))
sorted_features = sorted(feature_importance, key=lambda x: abs(x[1]), reverse=True)

top_positive = sorted_features[:20]
top_negative = sorted_features[-20:]

print("Top positive features indicating offensive content:")
for feature, coef in top_positive:
    print(f"{feature}: {coef:.4f}")

print("\nTop negative features indicating non-offensive content:")
for feature, coef in top_negative:
    print(f"{feature}: {coef:.4f}")

feature_df = pd.DataFrame(sorted_features, columns=['Feature', 'Coefficient'])

feature_df.head(20).plot(kind='barh', x='Feature', y='Coefficient', title='Top Positive Features')
plt.show()