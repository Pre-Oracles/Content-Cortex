import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load your data
df = pd.read_csv('dataset_1.csv')  # Ensure this CSV has 'text' and 'label' columns
df = df.sample(10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(df['tweet'], df['hate_speech'], test_size=0.6, random_state=42)

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to convert text to BERT embeddings
def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Apply the function to the training and test sets
try:
    X_train_embeddings = np.array([get_bert_embeddings(text) for text in X_train])
    X_test_embeddings = np.array([get_bert_embeddings(text) for text in X_test])
except Exception as e:
    print(f"Error during embedding generation: {e}")
    exit()

# Train logistic regression model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

try:
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_embeddings, y_train)
    y_pred = model.predict(X_test_embeddings)
except Exception as e:
    print(f"Error during model training/prediction: {e}")
    exit()

# Evaluate the model
try:
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')
    print('Classification Report:')
    print(classification_report(y_test, y_pred))
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
except Exception as e:
    print(f"Error during evaluation: {e}")
    exit()