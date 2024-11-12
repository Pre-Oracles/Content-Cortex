# sentiment_analyzer.py

class SentimentAnalyzer:
    def __init__(self):
        # Initialize positive and negative word lists
        self.positive_words = {
            'good', 'great', 'awesome', 'excellent', 'happy', 'love', 'wonderful',
            'fantastic', 'nice', 'amazing', 'beautiful', 'kind', 'friendly',
            'positive', 'brilliant', 'perfect', 'pleasant', 'lovely', 'delightful'
        }
        
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'hate', 'angry', 'poor',
            'negative', 'ugly', 'wrong', 'worst', 'stupid', 'dumb', 'idiot',
            'useless', 'disgusting', 'evil', 'cruel', 'mean', 'vicious'
        }
        
        # Initialize intensity modifiers
        self.intensifiers = {
            'very': 1.5,
            'really': 1.3,
            'extremely': 2.0,
            'totally': 1.4,
            'completely': 1.4,
            'absolutely': 1.5,
            'highly': 1.3
        }

    def analyze_sentiment(self, tokens):
        """
        Analyze sentiment of text using a simple scoring system
        """
        score = 0
        max_score = len(tokens)  # Normalize based on text length
        current_intensifier = 1.0

        for i, token in enumerate(tokens):
            # Check for intensifiers
            if token in self.intensifiers:
                current_intensifier = self.intensifiers[token]
                continue
                
            # Score words
            if token in self.positive_words:
                score += (1 * current_intensifier)
            elif token in self.negative_words:
                score -= (1 * current_intensifier)
                
            # Reset intensifier
            current_intensifier = 1.0

        # Normalize score to range [-1, 1]
        normalized_score = score / max_score if max_score > 0 else 0

        # Calculate basic metrics
        return {
            'compound': normalized_score,
            'pos': max(0, normalized_score),
            'neg': abs(min(0, normalized_score)),
            'neu': 1 - abs(normalized_score)
        }