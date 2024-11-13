from hate_speech_detector import HateSpeechDetector

class TextAnalyzer:
    def __init__(self, model_path="hate_speech_model.json"):
        self.detector = HateSpeechDetector()
        self.detector.load_model(model_path)
    
    def analyze_text(self, text):
        result = self.detector.predict(text)
        
        print("\nText Analysis Results:")
        print("=" * 50)
        print(f"Text: {text}")
        print(f"Classification: {result['prediction'].upper()}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Hate Speech Probability: {result['hate_speech_probability']:.2%}")
        
        if result['important_features']:
            print("\nKey Features Detected:")
            for feature, importance in sorted(
                result['important_features'].items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]:
                print(f"- {feature}: {importance:.3f}")
        
        return result

def main():
    analyzer = TextAnalyzer()
    print("Hate Speech Analyzer")
    print("Enter q to exit")
    print("-" * 50)
    
    while True:
        text = input("\nEnter text to analyze: ")
        if text == 'q':
            break
        analyzer.analyze_text(text)

if __name__ == "__main__":
    main()