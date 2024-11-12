import nltk
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
import re

class HateSpeechContentAnalyzer:
    def __init__(self):
        # Initialize required NLTK downloads
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('maxent_ne_chunker')
        nltk.download('words')
        
        # Initialize analyzers
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.hate_speech_classifier = pipeline(
            "text-classification",
            model="facebook/roberta-hate-speech-dynabench-r4-target",
            return_all_scores=True
        )
        self.stop_words = set(stopwords.words('english'))
        
    def analyze_content(self, text):
        """
        Comprehensive content analysis for hate speech detection
        """
        # Basic text cleaning
        text = self._clean_text(text)
        
        # Get different analysis features
        sentiment_scores = self._analyze_sentiment(text)
        linguistic_features = self._extract_linguistic_features(text)
        contextual_features = self._analyze_context(text)
        target_analysis = self._analyze_targets(text)
        
        return {
            'sentiment_analysis': sentiment_scores,
            'linguistic_features': linguistic_features,
            'contextual_features': contextual_features,
            'target_analysis': target_analysis,
            'hate_speech_probability': self._get_hate_speech_score(text)
        }
    
    def _clean_text(self, text):
        """Clean and normalize text"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def _analyze_sentiment(self, text):
        """
        Multiple sentiment analysis approaches combined
        """
        # VADER sentiment
        vader_scores = self.sentiment_analyzer.polarity_scores(text)
        
        # TextBlob sentiment
        blob = TextBlob(text)
        textblob_sentiment = {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
        
        return {
            'vader': vader_scores,
            'textblob': textblob_sentiment
        }
    
    def _extract_linguistic_features(self, text):
        """
        Extract linguistic patterns and features using NLTK instead of spaCy
        """
        # Tokenize text
        tokens = word_tokenize(text)
        sentences = sent_tokenize(text)
        
        # Get POS tags
        pos_tags = pos_tag(tokens)
        pos_dist = {}
        for _, tag in pos_tags:
            pos_dist[tag] = pos_dist.get(tag, 0) + 1
        
        # Named Entity Recognition using NLTK
        named_entities = []
        chunks = ne_chunk(pos_tags)
        for chunk in chunks:
            if hasattr(chunk, 'label'):
                named_entities.append((chunk.label(), ' '.join(c[0] for c in chunk)))
        
        return {
            'pos_distribution': pos_dist,
            'named_entities': named_entities,
            'text_complexity': {
                'avg_word_length': np.mean([len(token) for token in tokens]),
                'sentence_count': len(sentences)
            }
        }
    
    def _analyze_context(self, text):
        """
        Analyze contextual elements using NLTK
        """
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        
        # Extract noun phrases (simplified)
        topics = []
        modifiers = []
        
        for i, (word, tag) in enumerate(pos_tags):
            # Collect nouns and noun phrases
            if tag.startswith('NN'):
                if i > 0 and pos_tags[i-1][1].startswith('JJ'):
                    topics.append(f"{pos_tags[i-1][0]} {word}")
                else:
                    topics.append(word)
                    
            # Collect modifiers (adjectives and adverbs)
            if tag.startswith('JJ') or tag.startswith('RB'):
                modifiers.append(word)
        
        return {
            'topics': list(set(topics)),  # Remove duplicates
            'modifiers': list(set(modifiers)),
            'has_modal_verbs': any(tag == 'MD' for _, tag in pos_tags)
        }
    
    def _analyze_targets(self, text):
        """
        Analyze potential targets of hate speech using NLTK NER
        """
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        chunks = ne_chunk(pos_tags)
        
        targets = []
        for chunk in chunks:
            if hasattr(chunk, 'label'):
                if chunk.label() in ['PERSON', 'ORGANIZATION', 'GPE']:
                    targets.append({
                        'text': ' '.join(c[0] for c in chunk),
                        'type': chunk.label()
                    })
        
        return {
            'potential_targets': targets,
            'target_count': len(targets)
        }
    
    def _get_hate_speech_score(self, text):
        """
        Get hate speech probability using pre-trained model
        """
        results = self.hate_speech_classifier(text)
        return results[0]

    def get_recommendations(self, analysis_results):
        """
        Provide recommendations based on analysis results
        """
        recommendations = []
        
        # Check sentiment intensity
        if analysis_results['sentiment_analysis']['vader']['compound'] < -0.5:
            recommendations.append("High negative sentiment detected - review for potential hate speech")
        
        # Check for targeted content
        if analysis_results['target_analysis']['target_count'] > 0:
            recommendations.append("Content appears to target specific groups or individuals")
        
        # Check linguistic complexity
        if analysis_results['linguistic_features']['text_complexity']['avg_word_length'] > 6:
            recommendations.append("Complex language detected - review for subtle forms of bias")
        
        return recommendations
    
# Initialize the analyzer
analyzer = HateSpeechContentAnalyzer()

# Test the analyzer
text = "Your are dumb"
results = analyzer.analyze_content(text)
recommendations = analyzer.get_recommendations(results)

# Print results
print("Analysis Results:", results)
print("Recommendations:", recommendations)