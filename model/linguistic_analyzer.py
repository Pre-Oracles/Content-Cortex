# linguistic_analyzer.py

class LinguisticAnalyzer:
    def __init__(self):
        # Basic parts of speech patterns
        self.pos_patterns = {
            'noun': ['ness', 'ment', 'ship', 'sion', 'tion', 'age', 'ity'],
            'verb': ['ate', 'ize', 'ify', 'ing', 'ed'],
            'adjective': ['able', 'ible', 'ful', 'ous', 'ive', 'est', 'er'],
            'adverb': ['ly']
        }

    def guess_pos(self, word):
        """
        Guess part of speech based on word endings
        Not as accurate as NLTK but works without external dependencies
        """
        word = word.lower()
        
        for pos, suffixes in self.pos_patterns.items():
            for suffix in suffixes:
                if word.endswith(suffix):
                    return pos
        
        return 'unknown'

    def extract_features(self, tokens):
        """
        Extract basic linguistic features from text
        """
        # Count word lengths
        word_lengths = [len(token) for token in tokens]
        avg_word_length = sum(word_lengths) / len(tokens) if tokens else 0
        
        # Count parts of speech
        pos_counts = {'noun': 0, 'verb': 0, 'adjective': 0, 'adverb': 0, 'unknown': 0}
        for token in tokens:
            pos = self.guess_pos(token)
            pos_counts[pos] += 1
        
        # Calculate sentence length (basic approximation)
        sentence_count = 1
        for token in tokens:
            if token.endswith(('.', '!', '?')):
                sentence_count += 1
                
        return {
            'text_complexity': {
                'avg_word_length': avg_word_length,
                'sentence_count': sentence_count,
                'vocab_size': len(set(tokens))
            },
            'pos_distribution': pos_counts
        }

    def find_repeated_patterns(self, tokens):
        """
        Find repeated word patterns that might indicate emphasis or aggression
        """
        repeated_patterns = []
        for i in range(len(tokens) - 1):
            if tokens[i] == tokens[i + 1]:
                repeated_patterns.append(tokens[i])
        
        return repeated_patterns