import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re 
stop_words = ["a", "the", "is", "are","but","yet"]
def collect_historical_data():
    
    data_sources = ['dataset_1.csv']
    historical_data = []
    for source in data_sources:
        #Load data from each source
        data = pd.read_csv(source)
        historical_data.append(data)
    return historical_data
def clean_text(text):
    text = re.sub(r'[^\w\s]','',text)
    return text

def tokenize(text):
    clean_tokens = text.split(" ")
    for token in clean_tokens:
        if(token in stop_words):
            clean_tokens.remove(token)
    return clean_tokens
def stem_and_lemmatize(token):
    #use nltk?
    return token
def preprocess_data(data):
   processed_data = []
   for text in data:
       cleaned_text = clean_text(text)
       tokenized_text = tokenize(cleaned_text)
       stemmed_and_lemmatized_text = [stem_and_lemmatize(token) for token in tokenized_text]
       processed_data.append(stemmed_and_lemmatized_text)
   return processed_data


"""
def preprocess_data(data):
    return tokenize(data)

    processed_data = []
    for text in data:
       #if(data["class"]==0){}
       tokenized_text = tokenize(text["tweet"])
       # Stem and lemmatize the tokens
       stemmed_and_lemmatized_text = [stem_and_lemmatize(token) for token in tokenized_text]
       processed_data.append(stemmed_and_lemmatized_text)
   return processed_data
   
def create_model():
    # Select the appropriate architecture (RNN, CNN, or Transformer)
    model_architecture = select_architecture()
    # Initialize word embeddings (Word2Vec, GloVe)
    word_embeddings = initialize_word_embeddings()
    # Define the loss function (e.g., cross-entropy)
    loss_function = define_loss_function()
    # Define the optimizer (e.g., Adam, SGD)
    optimizer = define_optimizer()
    # Build the model with the selected architecture, embeddings, loss function, and optimizer
    model = build_model(model_architecture, word_embeddings, loss_function, optimizer)
    return model
 
   
def main():
    historical_data = collect_historical_data()
    #preprocess_data = preprocess_data(historical_data)
    X_text = historical_data['tweet']
    Y = historical_data['hate_speech']
    
    bow_transformer = CountVectorizer(analyzer=tokenize, max_features=800).fit(X_text)
    X = bow_transformer.transform(X_text)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=101)
    #print(bow_transformer.vocabulary_)
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train,Y_train)
    Y_pred = logistic_model.predict(X_test)
    accuracy = accuracy_score(Y_test,Y_pred)
    print (accuracy)
    prediction = logistic_model.predict(bow_transformer.transform(["hi im alex"]))
    if prediction:
        print ("This was a GOOD comment!")
    else:
        print ("This was a BAD comment!")
main()
"""