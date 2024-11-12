from hate_speech_detector import HateSpeechDetector
import random
import pandas as pd

def train_model():
    # Create detector
    detector = HateSpeechDetector()
    
    # Load and prepare the dataset
    print("Loading dataset...")
    df = pd.read_csv("dataset_1.csv")
    
    # Convert data to the required format
    # If hate_speech is not 0, label as "hate", otherwise "normal"
    training_data = [
        (row['tweet'], "hate" if row['hate_speech'] != 0 else "normal")
        for _, row in df.iterrows()
    ]
    
    # Split into training and validation
    random.shuffle(training_data)
    split_point = int(len(training_data) * 0.8)
    train_set = training_data[:split_point]
    val_set = training_data[split_point:]
    
    # Train
    print(f"Training with {len(train_set)} examples...")
    detector.train(train_set)
    
    # Validate
    print("\nValidating model...")
    correct = 0
    total = len(val_set)
    
    # Detailed metrics
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    
    for text, true_label in val_set:
        prediction = detector.predict(text)
        pred_label = prediction['prediction']
        
        if pred_label == true_label:
            correct += 1
            if true_label == "hate":
                true_pos += 1
            else:
                true_neg += 1
        else:
            if true_label == "hate":
                false_neg += 1
            else:
                false_pos += 1
    
    # Calculate metrics
    accuracy = correct / total
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Print metrics
    print(f"\nValidation Metrics:")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1 Score: {f1:.2%}")
    
    # Save model
    detector.save_model("hate_speech_model.json")
    print("\nModel saved to 'hate_speech_model.json'")

if __name__ == "__main__":
    train_model()