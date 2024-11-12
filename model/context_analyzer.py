# context_analyzer.py

class ContextAnalyzer:
    def __init__(self):
        # Initialize common target indicators
        self.target_indicators = {
            'identity': {
                'racial': ['race', 'ethnic', 'black', 'white', 'asian', 'latino', 'hispanic'],
                'gender': ['man', 'woman', 'male', 'female', 'gay', 'lesbian', 'trans'],
                'religious': ['muslim', 'christian', 'jewish', 'hindu', 'buddhist', 'atheist'],
                'nationality': ['american', 'chinese', 'indian', 'european', 'african']
            },
            'profanity': {
                'mild': ['stupid', 'dumb', 'idiot', 'fool'],
                'moderate': ['damn', 'hell'],
                'severe': ['*']  # Placeholder for actual profanity words
            }
        }

    def analyze_context(self, tokens):
        """
        Analyze contextual elements of the text
        """
        # Initialize counters
        context_scores = {
            'identity_references': 0,
            'profanity_level': 0,
            'target_groups': set()
        }
        
        # Analyze each token
        for token in tokens:
            token = token.lower()
            
            # Check for identity-based references
            for category, terms in self.target_indicators['identity'].items():
                if token in terms:
                    context_scores['identity_references'] += 1
                    context_scores['target_groups'].add(category)
            
            # Check for profanity levels
            for level, terms in self.target_indicators['profanity'].items():
                if token in terms:
                    if level == 'mild':
                        context_scores['profanity_level'] += 1
                    elif level == 'moderate':
                        context_scores['profanity_level'] += 2
                    else:  # severe
                        context_scores['profanity_level'] += 3

        return context_scores

    def analyze_phrase_patterns(self, tokens):
        """
        Analyze common phrase patterns that might indicate hate speech
        """
        patterns = {
            'negative_associations': 0,
            'stereotyping': 0,
            'threatening': 0
        }
        
        # Simple sliding window to detect patterns
        for i in range(len(tokens) - 2):
            three_gram = ' '.join(tokens[i:i+3]).lower()
            
            # Check for negative associations
            if "all" in three_gram and any(term in three_gram for term in self.target_indicators['identity']['racial']):
                patterns['stereotyping'] += 1
            
            # Check for threatening language
            if any(word in three_gram for word in ['kill', 'hurt', 'destroy', 'die']):
                patterns['threatening'] += 1
                
        return patterns