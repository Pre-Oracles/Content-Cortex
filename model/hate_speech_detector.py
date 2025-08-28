import math
from collections import defaultdict
import json
from text_preprocessor import TextPreprocessor
from sentiment_analyzer import SentimentAnalyzer
from linguistic_analyzer import LinguisticAnalyzer
from context_analyzer import ContextAnalyzer

class HateSpeechDetector:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.linguistic_analyzer = LinguisticAnalyzer()
        self.context_analyzer = ContextAnalyzer()
        
        self.feature_weights = defaultdict(float)
        self.vocab = set()
        
    def extract_features(self, text):
        #Extract numerical features from text
        processed = self.preprocessor.process_text(text)
        tokens = processed['tokens']
        
        # Get all analysis components
        sentiment_scores = self.sentiment_analyzer.analyze_sentiment(tokens)
        linguistic_features = self.linguistic_analyzer.extract_features(tokens)
        context_scores = self.context_analyzer.analyze_context(tokens)
        phrase_patterns = self.context_analyzer.analyze_phrase_patterns(tokens)
        
        # Convert analyses into numerical features
        features = {
            'sentiment_compound': sentiment_scores['compound'],
            'sentiment_negative': sentiment_scores['neg'],
            'avg_word_length': linguistic_features['text_complexity']['avg_word_length'],
            'vocab_size': linguistic_features['text_complexity']['vocab_size'],
            'identity_references': context_scores['identity_references'],
            'profanity_level': context_scores['profanity_level'],
            'target_groups': len(context_scores['target_groups']),
            'negative_associations': phrase_patterns['negative_associations'],
            'stereotyping': phrase_patterns['stereotyping'],
            'threatening': phrase_patterns['threatening']
        }
        
        # Add word presence features
        for token in set(tokens):
            self.vocab.add(token)
            features[f'word_{token}'] = 1
            
        return features

    def _predict_probability(self, features):
        z = sum(self.feature_weights[feature] * value 
                for feature, value in features.items())
        return 1 / (1 + math.exp(-z))

    def train(self, training_data):
        print("Starting training...")
        self.feature_weights = defaultdict(float)


        # process the training data
        processed_data = []
        for text, label in training_data:
            features = self.extract_features(text)
            processed_data.append((features, label))

    
        # Train using logistic regression
        learning_rate = 0.01
        epochs = 200
        for epoch in range(epochs):
            total_loss = 0
            
            for features, label in processed_data:
                prediction = self._predict_probability(features)
                target = 1.0 if label == 'hate' else 0.0
                
                # Calculate loss and update weights
                loss = target - prediction
                total_loss += abs(loss)
                
                for feature, value in features.items():
                    self.feature_weights[feature] += learning_rate * loss * value
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(processed_data)
                print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

    def predict(self, text):
        """Predict whether text contains hate speech"""
        features = self.extract_features(text)
        hate_probability = self._predict_probability(features)
        
        # Get feature importance
        feature_importance = {
            feature: abs(self.feature_weights[feature])
            for feature in features
            if abs(self.feature_weights[feature]) > 0.1
        }
        
        return {
            'hate_speech_probability': hate_probability,
            'prediction': 'hate' if hate_probability > 0.5 else 'normal',
            'confidence': abs(hate_probability - 0.5) * 2,
            'important_features': feature_importance
        }

    def save_model(self, filepath):
        model_data = {
            'feature_weights': dict(self.feature_weights),
            
            'vocab': list(self.vocab)
        }
        with open(filepath, 'w') as f:
            json.dump(model_data, f)

    def load_model(self, filepath):
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        self.feature_weights = defaultdict(float, model_data['feature_weights'])
        self.vocab = set(model_data['vocab'])