import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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
X_sparse = vectorizer.fit_transform(df['tweet'])
X = X_sparse.toarray()
df['hate_speech'] = df['hate_speech'].apply(lambda x: 0 if x == 0 else 1)

y = df['hate_speech']
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

input_dim = X.shape[1]
model = SimpleNN(input_dim)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)



for epoch in range(10):
    for batch in dataloader:
        inputs, targets = batch
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    outputs = model(torch.tensor(X_test, dtype=torch.float32))
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == torch.tensor(y_test)).float().mean().item()
    print(f'Test Accuracy: {accuracy}')
