import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# Load and preprocess data
def load_and_preprocess_data(file_path):
   # Load data
   df = pd.read_csv(file_path)
  
   # Check for missing values and handle them
   df.dropna(inplace=True)
  
   # Basic text preprocessing
   df['tweet'] = df['tweet'].apply(lambda x: x.lower())  # Convert to lowercase
   df['tweet'] = df['tweet'].str.replace(r'\b\w{1,2}\b', '')  # Remove short words
   df['tweet'] = df['tweet'].str.replace(r'[^\w\s]', '')  # Remove punctuation
  
   # Tokenize text data
   tokenizer = Tokenizer(num_words=10000)
   tokenizer.fit_on_texts(df['tweet'])
   sequences = tokenizer.texts_to_sequences(df['tweet'])
  
   # Pad sequences
   data = pad_sequences(sequences, maxlen=100)
  
   # Encode labels
   label_encoder = LabelEncoder()
   labels = label_encoder.fit_transform(df['offensive_language'])
   labels = to_categorical(labels)
   # Split data into train and test sets
   train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
  
   return (train_data, train_labels), (test_data, test_labels), tokenizer.word_index, label_encoder


# Define the model architecture
def build_model(vocab_size, embedding_dim, input_length):
   model = Sequential()
   model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length))
   model.add(LSTM(units=128, return_sequences=True))
   model.add(Dropout(0.5))
   model.add(LSTM(units=64))
   model.add(Dropout(0.5))
   model.add(Dense(units=32, activation='relu'))
   model.add(Dense(units=train_labels.shape[1], activation='softmax'))
  
   # Compile the model
   model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
   return model


# Train the model
def train_model(model, train_data, train_labels, epochs, batch_size):
   early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
   model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping])
   return model


# Evaluate the model
def evaluate_model(model, test_data, test_labels):
   evaluation_metrics = model.evaluate(test_data, test_labels)
   return evaluation_metrics


# Main execution
file_path = 'dataset_1.csv'
embedding_dim = 128
input_length = 100
epochs = 10
batch_size = 32


(train_data, train_labels), (test_data, test_labels), word_index, label_encoder = load_and_preprocess_data(file_path)
vocab_size = min(10000, len(word_index) + 1)
model = build_model(vocab_size, embedding_dim, input_length)
trained_model = train_model(model, train_data, train_labels, epochs, batch_size)
evaluation_metrics = evaluate_model(trained_model, test_data, test_labels)


print(f"Test Loss: {evaluation_metrics[0]}")
print(f"Test Accuracy: {evaluation_metrics[1]}")

